#!/usr/bin/env python3
"""代码沙箱：跑模型写的 solution 对上题目自带的 pytest。

这是 Step 1 的地基——奖励函数、scripted 修复探针、分解评测全部走这里。

两个返回值，用途不同，**不要混用**：
  - pass_ratio  通过测试的比例，训练奖励用。1.5B 在 hard 题上全 0 会让
                GRPO 一整组 advantage 归零、没有梯度，partial credit 治这个。
  - all_passed  严格全通过，评测用。混用会让数字和公开 pass@1 不可比，
                而且 partial credit 会诱导模型只挑简单测试满足。

隔离手段是 rlimit + 独立临时目录 + 进程组 kill，不是完整沙箱化
（seccomp/nsjail）。跑在容器里、输入是自己训的小模型，这个强度够用；
换成不可信输入就不够了。

**并行度按内存算，不是按核数算。** 无卡模式 cgroup 只给 2 GiB，而这台有
112 核——照核数开 112 个 pytest 进程会直接 OOM。
"""

from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 默认限额

TIMEOUT_S = 10.0        # wall-clock
MEM_MB = 512            # 单进程地址空间；python+pytest 本身约需 60–100 MB。
                        # **训练时保持 512**：跑的是模型生成的代码，1.5B 写不出
                        # 需要 GB 级内存的正确解，却完全写得出内存炸弹；而 112 路
                        # 并行 × 4 GB 上限 = 448 GB，远超 120 GiB 的 cgroup 上限。
VERIFY_MEM_MB = 4096    # 仅用于**验证官方参考解**：Mbpp/255 的官方解
                        # （combinations_with_replacement）实测需要约 3 GB，
                        # 512 MB 下会误报 MemoryError。一次性动作，并行度低，可以放宽。
MAX_OUTPUT = 1600       # 返回给模型的 stderr 字符数（约 400 token）
FSIZE_MB = 16           # 防止 solution 写爆磁盘
NPROC = 64              # 防 fork bomb
HEAD_TAIL = 100_000     # 输出过长时头尾各取的字节数（摘要行在尾部）


@dataclass
class Result:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    status: str = "ok"          # ok / timeout / crash / no_tests
    stderr: str = ""            # 已截断，喂给模型的那份
    duration: float = 0.0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def pass_ratio(self) -> float:
        """训练奖励用。没有测试跑起来一律 0。"""
        return self.passed / self.total if self.total else 0.0

    @property
    def all_passed(self) -> bool:
        """评测用，严格。"""
        return self.status == "ok" and self.total > 0 and self.passed == self.total


# ---------------------------------------------------------------------------
# 子进程限额

def _apply_limits(mem_mb: int, cpu_s: int) -> None:
    """在 fork 之后、exec 之前生效。

    CPU 时间和 wall-clock 是两道独立的防线：sleep 不烧 CPU 但耗 wall-clock，
    死循环反过来。只设一个都会漏。
    """
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_MB << 20, FSIZE_MB << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (NPROC, NPROC))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _kill_group(p: subprocess.Popen) -> None:
    """杀整个进程组。

    只 kill 父进程的话，pytest 派生出的子进程会活下来继续吃 CPU——
    这就是为什么要 start_new_session=True。
    """
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# 输出解析

# KodCode 的 test 有两种风格，必须分别处理：
#   A) `from solution import f` —— solution 与 test 分两个文件
#   B) 直接写 `assert f(...)`   —— test 假设 solution 已拼在自己前面
# 按单一风格处理会让 B 类全部 NameError（实测占 200 条样本的 16.5%）。
_IMPORTS_SOLUTION = re.compile(
    r"^\s*(?:from\s+solution\s+import|import\s+solution)\b", re.M)

_PAT = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) errors?\b"),
}


def _parse(text: str) -> tuple[int, int, int]:
    def grab(k: str) -> int:
        m = _PAT[k].search(text)
        return int(m.group(1)) if m else 0
    return grab("passed"), grab("failed"), grab("errors")


def _read_capped(path: Path) -> str:
    """读输出文件，过长时取头 + 尾——pytest 的摘要行在最后，不能只截头部。"""
    size = path.stat().st_size if path.exists() else 0
    if size == 0:
        return ""
    with open(path, "rb") as f:
        if size <= 2 * HEAD_TAIL:
            raw = f.read()
        else:
            head = f.read(HEAD_TAIL)
            f.seek(-HEAD_TAIL, os.SEEK_END)
            raw = head + b"\n...[output truncated]...\n" + f.read()
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------

def build_evalplus(rec: dict, solution: str | None = None) -> tuple[str, str]:
    """把一条 EvalPlus 记录拼成 (solution, test) 供 mode="script" 使用。

    两个 benchmark 的约定不同，实测得来：
      - HumanEval+：canonical_solution 只是**函数体**，要和 prompt 拼起来；
        test 里定义了 `def check(candidate)` 但**自己不调用**，末尾要补
        `check(entry_point)`。
      - MBPP+：code 是完整函数；test 是模块级代码且**直接引用函数名**
        （不是 candidate），拼上即可。

    solution 传 None 时用官方解；传入则替换（buggy 初稿或模型输出走这里）。
    """
    is_he = bool(rec.get("entry_point"))
    if solution is None:
        solution = (rec["prompt"] + rec["canonical_solution"]) if is_he \
            else rec["code"]
    suffix = f"\n\ncheck({rec['entry_point']})\n" if is_he else ""
    return solution, rec["test"] + suffix


def run_tests(solution: str, test: str, timeout: float = TIMEOUT_S,
              mem_mb: int = MEM_MB, max_output: int = MAX_OUTPUT,
              mode: str = "pytest", suffix: str = "") -> Result:
    """把 solution 与 test 落盘到独立临时目录，跑 pytest，返回结构化结果。

    mode="pytest"（训练集 KodCode）解析 passed/failed，给得出 partial credit。
    mode="script"（评测集 EvalPlus）直接跑脚本、只看退出码，得到严格 0/1——
    EvalPlus 没有 test_ 函数，pytest 跑它只会报 "no tests ran"。
    两种模式的分工正对应"训练用 partial credit、评测用严格全通过"。

    pytest 模式下两种 test 风格自动分流（见 _IMPORTS_SOLUTION）。写 `from solution import xxx`
    的走双文件——pytest 默认 prepend import mode 会把测试文件所在目录插到
    sys.path[0]，同目录的 solution.py 能被导入；不写 import 的则把 solution
    拼到 test 前面，因为那种 test 假设两者同处一个命名空间。
    """
    # 沙箱**总**执行次数。注意：reward_code.py 评测最终代码时也调用本函数，
    # 因此本计数 = 工具调用 + 奖励评测，**不能**当作工具调用次数。
    # 工具调用数见 CodeTool.execute / ReActAgentLoop._run_tests 里的专用计数。
    try:
        _c = os.environ.get('SANDBOX_EXEC_COUNTER',
                            '/root/autodl-tmp/runs/.sandbox_exec_count')
        with open(_c, 'a') as _f:
            _f.write('1\n')
    except Exception:
        pass

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="sbx-") as d:
        dd = Path(d)
        out = dd / "_out.txt"
        if mode == "script":
            (dd / "main.py").write_text(f"{solution}\n\n{test}{suffix}",
                                        encoding="utf-8")
            cmd = [sys.executable, "main.py"]
        else:
            (dd / "solution.py").write_text(solution, encoding="utf-8")
            body = (test if _IMPORTS_SOLUTION.search(test)
                    else f"{solution}\n\n{test}")
            (dd / "test_solution.py").write_text(body, encoding="utf-8")
            cmd = [sys.executable, "-m", "pytest", "test_solution.py",
                   "-q", "--tb=short", "--no-header",
                   "-p", "no:cacheprovider", "-p", "no:randomly"]

        env = dict(os.environ)
        env.update({
            "PYTHONPATH": d, "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg",
            # BLAS 后端默认按 CPU 核数开线程池。这台有 112 核，每线程的缓冲区
            # 加起来远超 RLIMIT_AS，numpy 一 import 就是
            # "OpenBLAS error: Memory allocation still failed"。
            # 沙箱里跑单元测试不需要多线程 BLAS，而并行 rollout 下多线程
            # 只会互相争抢——一律钉成 1。
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        })

        status = "ok"
        with open(out, "wb") as fo:
            try:
                p = subprocess.Popen(
                    cmd, cwd=d, stdout=fo, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=env, start_new_session=True,
                    preexec_fn=lambda: _apply_limits(mem_mb, int(timeout) + 1),
                )
            except OSError as e:
                return Result(status="crash", stderr=f"spawn failed: {e}",
                              duration=time.time() - t0)
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(p)
                status = "timeout"

        text = _read_capped(out)
        rc = p.returncode

    if mode == "script":
        # 断言脚本第一个失败就抛异常退出，天然没有 partial credit——
        # 这正是评测要的严格 0/1。
        if status == "timeout":
            return Result(failed=1, status="timeout",
                          stderr=text[-max_output:], duration=time.time() - t0)
        ok = rc == 0
        return Result(passed=int(ok), failed=int(not ok), status="ok",
                      stderr=text[-max_output:], duration=time.time() - t0)

    pa, fa, er = _parse(text)
    if status == "ok" and pa + fa + er == 0:
        status = "no_tests"          # 语法错误到连 collection 都没跑起来也落这里
    return Result(passed=pa, failed=fa, errors=er, status=status,
                  stderr=text[-max_output:], duration=time.time() - t0)


# ---------------------------------------------------------------------------
# 并行

def _cgroup_limit_bytes() -> int:
    """容器真实内存上限。`free` 报的是宿主机的值，不能用。"""
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v not in ("max", ""):
                return int(v)
        except (OSError, ValueError):
            continue
    return 8 << 30          # 拿不到就按 8 GiB 保守估


PARENT_RESERVE_MB = 1024    # 父进程常驻开销（HF 数据集等），不能拿来开 worker


def safe_workers(per_proc_mb: int = 350) -> int:
    """按内存而非核数定并行度。

    实测（2 GiB cgroup，40 条真实样本，tune_workers.py）：

        fork   1→2.0/s  2→1.9/s  4→1.8/s  8→1.7/s  12→1.7/s
        spawn  1→1.8/s  2→1.6/s  4→1.3/s  8→1.0/s  12→0.7/s

    **并行度越高越慢，串行最快**——2 GiB 下没有并行空间，112 核完全用不上，
    内存是唯一瓶颈；spawn 更差，因为每个 worker 都要重新 import。
    旧版按 150 MB/进程估会算出 8，实测反而把吞吐从 2.0 压到 1.7。

    开卡后内存几百 GB，同一个公式会自动放开到核数上限。
    """
    budget = int((_cgroup_limit_bytes() - (PARENT_RESERVE_MB << 20)) * 0.6)
    by_mem = max(1, budget // (per_proc_mb << 20))
    return max(1, min(os.cpu_count() or 1, by_mem))


def _one(args: tuple) -> Result:
    sol, tst, to, mem, mode = args
    return run_tests(sol, tst, timeout=to, mem_mb=mem, mode=mode)


def run_many(items: list[tuple[str, str]], workers: int | None = None,
             timeout: float = TIMEOUT_S, mem_mb: int = MEM_MB,
             mode: str = "pytest") -> list[Result]:
    """items 为 [(solution, test), ...]。mode 见 run_tests。"""
    w = workers or safe_workers()
    payload = [(s, t, timeout, mem_mb, mode) for s, t in items]
    with ProcessPoolExecutor(max_workers=w) as ex:
        return list(ex.map(_one, payload, chunksize=4))

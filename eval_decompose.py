"""分解评测：把 pass@1 的提升拆成"初稿质量"和"真·debug 能力"。

主图就是这里产出的三条曲线。两个**独立**的评测通道：

  通道 A（多轮）  542 道 EvalPlus，模型从零写 → 跑测试 → 修，最多 max_turns 轮。
                  turn-1 通过率 = 第一轮就全过的比例；总 pass@1 = 最后全过的比例。
  通道 B（修复）  542 份**固定的** buggy 初稿 + 真实报错，测模型能否修好。
                  分母是全部 542，与模型强弱无关，两个 checkpoint 拿到逐字相同
                  的题——这是 Step 0 推翻"自然失败集"后的替代方案。

按轮次批处理，不是逐题串行：每一轮把所有还没通过的题攒成一个 batch 交给
policy，所以总共只有 max_turns 次批量生成。542 题逐题串行会慢一个数量级。

policy 接口沿用 probes/policy.py：list[对话] -> list[生成文本]，贪心解码。
per-checkpoint 的指标应当是 checkpoint 的确定性函数。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import os

from code_tool_core import (extract_submitted_code, format_observation,
                            suspicious)
from sandbox import build_evalplus, run_many, safe_workers

# 与 prepare_code_data.py 的训练 SYSTEM 保持一致的措辞。评测和训练的 prompt
# 分布不一致会让 pass@1 无端偏低，而那个偏差看起来完全像是"模型不行"。
SYSTEM = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。请给出完整的 Python 代码，包含所需的 import。"""

MAX_TURNS = 4

# 失败后追问模型的措辞。做成可配置是为了做**干预实验**：
# 用强引导（"先分析根本原因再改"）对比默认措辞，看多轮救回率会不会动。
# 若显著上升 → 模型有能力只是没被引导出来，行为冷启动/SFT 值得做；
# 若纹丝不动 → 失败题确实超出能力边界，SFT 可以放心跳过。
# 一次评测约 ¥1，回答一个价值 ¥60 + 数天开发的问题。
DEFAULT_FOLLOWUP = "测试结果：\n{obs}\n\n请修正代码。"
FOLLOWUP = os.environ.get("CODE_EVAL_FOLLOWUP") or DEFAULT_FOLLOWUP


@dataclass
class Turn:
    turn: int
    passed: int
    total: int
    all_passed: bool
    status: str
    code: str = ""
    flags: list[str] = field(default_factory=list)
    # 模型这一轮实际看到的反馈。Step 3 事后要问"它是看到什么才改成这样的"，
    # 没存就永远查不了，而重跑一次是 12 小时。
    observation: str = ""


@dataclass
class Task:
    task_id: str
    source: str
    turns: list[Turn] = field(default_factory=list)
    mutation_kind: str = ""      # 仅通道 B

    @property
    def turn1_passed(self) -> bool:
        return bool(self.turns) and self.turns[0].all_passed

    @property
    def final_passed(self) -> bool:
        return bool(self.turns) and self.turns[-1].all_passed

    @property
    def n_turns(self) -> int:
        return len(self.turns)


# ---------------------------------------------------------------------------
# prompt 构造

def question_of(rec: dict) -> str:
    """MBPP+ 必须附上示例断言，否则模型不知道函数该叫什么名字。

    MBPP 的 prompt 只是一句自然语言（"Write a function to find the shared
    elements..."），而测试调用的是 `similar_elements(...)`。不给函数名的话
    pass@1 会异常低，而且低的原因与能力无关——这是 MBPP 的标准评测协议要求
    附上 test_list 的原因。HumanEval+ 的 prompt 自带签名，不需要这一步。
    """
    if rec.get("entry_point"):
        return rec["prompt"]
    tests = rec.get("test_list") or []
    body = "\n".join(str(t) for t in tests[:3])
    return f"{rec['prompt']}\n\n你的代码需要通过这些测试：\n{body}"


def repair_prompt(question: str, buggy: str, error: str) -> str:
    return (f"{question}\n\n下面这份实现没有通过测试：\n\n"
            f"```python\n{buggy}\n```\n\n测试报错：\n{error}\n\n请修正它。")


# ---------------------------------------------------------------------------
# 核心循环

def _run_batch(recs: list[dict], codes: list[str], workers: int) -> list:
    items = []
    for rec, code in zip(recs, codes):
        sol, test = build_evalplus(rec, solution=code)
        items.append((sol, test))
    return run_many(items, workers=workers, mode="script", timeout=30.0)


def evaluate(policy, recs: list[tuple[str, str, dict]], probes: dict | None = None,
             max_turns: int = MAX_TURNS, workers: int | None = None,
             max_tokens: int = 1024) -> list[Task]:
    """跑一个通道。probes 为 None 走通道 A，否则走通道 B。

    probes: {task_id: {"buggy_solution":…, "error":…, "mutation_kind":…}}
    """
    w = workers or safe_workers()
    tasks = {tid: Task(task_id=tid, source=src,
                       mutation_kind=(probes or {}).get(tid, {}).get("mutation_kind", ""))
             for src, tid, _ in recs}
    by_id = {tid: rec for _, tid, rec in recs}

    # 首轮的对话
    convs: dict[str, list[dict]] = {}
    for src, tid, rec in recs:
        q = question_of(rec)
        if probes is not None:
            p = probes.get(tid)
            if p is None:
                continue
            user = repair_prompt(q, p["buggy_solution"], p["error"])
        else:
            user = q
        convs[tid] = [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}]

    active = list(convs)
    for t in range(1, max_turns + 1):
        if not active:
            break
        outs = policy([convs[tid] for tid in active], max_tokens=max_tokens)
        codes = [extract_submitted_code(o) for o in outs]
        results = _run_batch([by_id[tid] for tid in active], codes, w)

        nxt = []
        for tid, out, code, res in zip(active, outs, codes, results):
            obs = format_observation(res.passed, res.total, res.status, res.stderr)
            tasks[tid].turns.append(Turn(
                turn=t, passed=res.passed, total=res.total,
                all_passed=res.all_passed, status=res.status,
                code=code, flags=suspicious(code), observation=obs))
            if res.all_passed or t == max_turns:
                continue
            convs[tid] = convs[tid] + [
                {"role": "assistant", "content": out},
                {"role": "user", "content": FOLLOWUP.format(obs=obs)},
            ]
            nxt.append(tid)
        active = nxt
    return list(tasks.values())


# ---------------------------------------------------------------------------
# 汇总

def summarize(multi: list[Task], repair: list[Task]) -> dict:
    """三个主指标 + 拆分。总 pass@1 与 turn-1 来自同一批轨迹，修复率来自另一批。"""
    n = len(multi) or 1
    out = {
        "n_multi": len(multi),
        "final_pass": sum(t.final_passed for t in multi) / n,
        "turn1_pass": sum(t.turn1_passed for t in multi) / n,
        "mean_turns": sum(t.n_turns for t in multi) / n,
        "turns_hist": dict(Counter(t.n_turns for t in multi)),
    }
    # 这一项才是"真·debug 能力"：分母固定，与模型强弱无关
    if repair:
        m = len(repair)
        out["n_repair"] = m
        out["repair_rate"] = sum(t.final_passed for t in repair) / m
        out["repair_rate_turn1"] = sum(t.turn1_passed for t in repair) / m
        by_kind = defaultdict(lambda: [0, 0])
        for t in repair:
            by_kind[t.mutation_kind][1] += 1
            by_kind[t.mutation_kind][0] += int(t.final_passed)
        out["repair_by_kind"] = {k: {"ok": v[0], "n": v[1], "rate": v[0] / v[1]}
                                 for k, v in sorted(by_kind.items())}
    # 作弊倾向：只记录不拦截，理由同 probes/common.py 的设计不变量
    flags = Counter(f for t in multi + repair for tr in t.turns for f in tr.flags)
    out["suspicious"] = dict(flags)
    return out


def dump(tasks: list[Task], path: Path, step: int, channel: str) -> None:
    """per-task per-turn 全量落盘——这是 repo 的差异化资产，也让配对检验成为可能。"""
    with open(path, "a", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps({"step": step, "channel": channel, **asdict(t)},
                               ensure_ascii=False) + "\n")


def scalars(summary: dict, prefix: str = "code") -> dict:
    """压成标量喂 verl 的 logger，训练时就能看见三条线分开走。"""
    out = {f"{prefix}/final_pass": summary["final_pass"],
           f"{prefix}/turn1_pass": summary["turn1_pass"],
           f"{prefix}/mean_turns": summary["mean_turns"]}
    if "repair_rate" in summary:
        out[f"{prefix}/repair_rate"] = summary["repair_rate"]
        # 主结论的那个差值：总提升里有多少不是靠初稿质量拿到的
        out[f"{prefix}/gap_final_minus_turn1"] = (
            summary["final_pass"] - summary["turn1_pass"])
    for k, v in summary.get("suspicious", {}).items():
        out[f"{prefix}/suspicious/{k}"] = v
    return out

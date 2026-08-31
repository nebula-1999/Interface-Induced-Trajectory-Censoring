"""训练过程中就地跑分解评测——权重一次都不落盘。

理由和 train/probe_hook.py 完全一样：1.5B 的完整训练态 checkpoint 约 21 GB
（bf16 权重 3 + AdamW 12 + fp32 主权重 6），数据盘只有 50 GB，存两个就满了。
就地评测把这一项从预算里删掉，也省掉每次写盘的停顿。

可行的原因同样是：verl 的 async rollout 本来就维护着一个与策略同步的 vLLM
服务并暴露 OpenAI 兼容端点，而评测只需要生成。

用法（verl 训练进程内，拿到 server_addresses 之后）::

    from code_eval_hook import CodeEvalHook
    hook = CodeEvalHook(server_addresses, model_path, out_dir)
    hook.run(step=global_step)      # 每个 eval 步调一次
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from chat_policy import ChatPolicy
from eval_decompose import dump, evaluate, scalars, summarize


def _base_url(addr: str) -> str:
    addr = addr.strip()
    if not addr.startswith(("http://", "https://")):
        addr = "http://" + addr
    return addr.rstrip("/") + "/v1"


class CodeEvalHook:
    def __init__(self, server_addresses: list[str], model_name: str,
                 out_dir: str = "/root/autodl-tmp/runs/code-eval",
                 probes_path: str = "probes_repair.jsonl",
                 max_turns: int = 4, workers: int = 32,
                 sandbox_workers: int | None = None, max_tokens: int = 1024,
                 limit: int | None = None):
        self.urls = [_base_url(a) for a in server_addresses]
        self.model_name = model_name
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self.workers = workers
        self.sandbox_workers = sandbox_workers
        self.max_tokens = max_tokens
        self.recs = self._load_evalplus()
        if limit:
            # 冒烟用：跑满 542 题要 1.5 小时（≈¥7.5），而冒烟只需要验证
            # 钩子端到端通不通。两个 benchmark 各取一半，别只取一个。
            half = limit // 2
            self.recs = self.recs[:half] + self.recs[-(limit - half):]
            print(f"[code-eval] limit={limit}，只评 {len(self.recs)} 题（冒烟模式）")
        self.probes = self._load_probes(Path(probes_path))

    @staticmethod
    def _load_evalplus() -> list[tuple[str, str, dict]]:
        from datasets import load_dataset
        out = []
        for r in load_dataset("evalplus/humanevalplus", split="test"):
            out.append(("HumanEval+", str(r["task_id"]), dict(r)))
        for r in load_dataset("evalplus/mbppplus", split="test"):
            out.append(("MBPP+", f"Mbpp/{r['task_id']}", dict(r)))
        return out

    @staticmethod
    def _load_probes(path: Path) -> dict:
        if not path.exists():
            print(f"⚠️  {path} 不存在，跳过修复通道——"
                  f"只剩总 pass@1 和 turn-1，主结论缺一条腿")
            return {}
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["task_id"]] = r
        return out

    def run(self, step: int) -> dict:
        t0 = time.time()
        policy = ChatPolicy(self.urls, self.model_name, temperature=0.0,
                            max_tokens=self.max_tokens, workers=self.workers)

        multi = evaluate(policy, self.recs, probes=None,
                         max_turns=self.max_turns, workers=self.sandbox_workers,
                         max_tokens=self.max_tokens)
        repair = []
        if self.probes:
            sub = [r for r in self.recs if r[1] in self.probes]
            repair = evaluate(policy, sub, probes=self.probes,
                              max_turns=self.max_turns,
                              workers=self.sandbox_workers,
                              max_tokens=self.max_tokens)

        # per-task per-turn 全量落盘：repo 的差异化资产，也让配对检验成为可能
        path = self.out / f"step_{step:05d}.jsonl"
        dump(multi, path, step, "multi")
        if repair:
            dump(repair, path, step, "repair")

        s = summarize(multi, repair)
        with open(self.out / "summary.log", "a", encoding="utf-8") as f:
            f.write(f"===== step {step} =====\n"
                    f"{json.dumps(s, ensure_ascii=False, indent=2)}\n"
                    f"{policy.failure_report()}\n\n")

        m = scalars(s)
        m["code/elapsed_s"] = round(time.time() - t0, 1)
        m["code/request_failures"] = sum(policy.failures.values())
        return m

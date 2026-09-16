"""In-place programmatic-feedback evaluation, not autonomous FC evaluation.

The serving endpoint's use of the current LoRA must be verified on the installed
stack. Sharing an address alone is NOT proof of weight synchronization. These
artifacts do not replace checkpoints or authorize a learning claim.

用法（verl 训练进程内，拿到 server_addresses 之后）::

    from code_eval_hook import CodeEvalHook
    hook = CodeEvalHook(server_addresses, model_path, out_dir)
    hook.run(step=global_step)      # 每个 eval 步调一次
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from chat_policy import ChatPolicy, NativeRolloutPolicy
from eval_decompose import evaluate, scalars, summarize, SYSTEM, FOLLOWUP
from eval_artifacts import StepArtifact, bind_manifest, unique_ids


def _base_url(addr: str) -> str:
    addr = addr.strip()
    if not addr.startswith(("http://", "https://")):
        addr = "http://" + addr
    addr = addr.rstrip("/")
    return addr if addr.endswith("/v1") else addr + "/v1"


class CodeEvalHook:
    def __init__(self, server_addresses: list[str], model_name: str,
                 out_dir: str = "/root/autodl-tmp/runs/code-eval",
                 probes_path: str = "probes_repair.jsonl",
                 max_turns: int = 4, workers: int = 32,
                 sandbox_workers: int | None = None, max_tokens: int = 1024,
                 limit: int | None = None, native_client=None, tokenizer=None,
                 expected_step: int | None = None):
        self.urls = [_base_url(a) for a in server_addresses]
        self.model_name = model_name
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self.workers = workers
        self.sandbox_workers = sandbox_workers
        self.max_tokens = max_tokens
        self.native_client = native_client
        self.tokenizer = tokenizer
        self.expected_step = expected_step
        if os.environ.get("CODE_EVAL_REQUIRE_NATIVE") == "1" and (
                native_client is None or tokenizer is None or expected_step is None):
            raise ValueError("P3 requires native rollout RPC with weight-step attestation")
        self.recs = self._load_evalplus()
        unique_ids([r[1] for r in self.recs], "EvalPlus")
        if limit:
            if limit < 0 or limit > len(self.recs):
                raise ValueError("Evaluation limit outside dataset bounds")
            # 冒烟用：跑满 542 题要 1.5 小时（≈¥7.5），而冒烟只需要验证
            # 钩子端到端通不通。两个 benchmark 各取一半，别只取一个。
            half = limit // 2
            self.recs = self.recs[:half] + self.recs[-(limit - half):]
            print(f"[code-eval] limit={limit}，只评 {len(self.recs)} 题（冒烟模式）")
        self.probes = self._load_probes(Path(probes_path))
        self.sub = [r for r in self.recs if r[1] in self.probes]
        if os.environ.get("CODE_EVAL_REQUIRE_REPAIR") == "1" and not self.sub:
            raise ValueError("Required repair channel has no tasks")
        # Pin actual bytes/content, not just a mutable dataset repository name.
        manifest = dict(records=self.recs, probes=self.probes, max_turns=max_turns,
                        max_tokens=max_tokens, temperature=0.0, system=SYSTEM,
                        followup=FOLLOWUP, protocol="programmatic_feedback",
                        generation_backend="native_rollout_rpc" if native_client else "openai_http")
        # Canonicalize tuples before comparison with JSON-loaded manifests.
        manifest = json.loads(json.dumps(manifest))
        manifest_path = os.environ.get("CODE_EVAL_MANIFEST", str(self.out / "manifest.json"))
        self.manifest_sha = bind_manifest(manifest_path, manifest)
        self.run_id = os.environ.get("CODE_EVAL_RUN_ID", str(self.out.resolve()))

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
                    if r["task_id"] in out:
                        raise ValueError(f"Duplicate repair probe: {r['task_id']}")
                    out[r["task_id"]] = r
        return out

    def run(self, step: int) -> dict:
        artifact = StepArtifact(self.out, step, self.run_id, self.manifest_sha)
        try:
            return self._run(step, artifact)
        except Exception as exc:
            artifact.fail(exc)
            raise

    def _run(self, step: int, artifact: StepArtifact) -> dict:
        t0 = time.time()
        if self.native_client is not None:
            policy = NativeRolloutPolicy(
                self.native_client, self.tokenizer, self.expected_step,
                max_tokens=self.max_tokens, workers=self.workers)
        else:
            policy = ChatPolicy(self.urls, self.model_name, temperature=0.0,
                                max_tokens=self.max_tokens, workers=self.workers,
                                raise_on_failure=True)

        multi = evaluate(policy, self.recs, probes=None,
                         max_turns=self.max_turns, workers=self.sandbox_workers,
                         max_tokens=self.max_tokens)
        repair = []
        if self.probes:
            repair = evaluate(policy, self.sub, probes=self.probes,
                              max_turns=self.max_turns,
                              workers=self.sandbox_workers,
                              max_tokens=self.max_tokens)

        s = summarize(multi, repair)
        m = scalars(s)
        m["code/elapsed_s"] = round(time.time() - t0, 1)
        m["code/request_failures"] = sum(policy.failures.values())
        if isinstance(policy, NativeRolloutPolicy):
            m["code/native_weight_step"] = self.expected_step
            m["code/native_requests"] = sum(policy.observed_steps.values())
        artifact.finish(multi, repair, [r[1] for r in self.recs],
                        [r[1] for r in self.sub], m)
        return m

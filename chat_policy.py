"""把 vLLM 的 OpenAI 兼容端点包成 batch policy：list[对话] -> list[生成文本]。

和 probes/policy.py 同构，但去掉了 `</tool_call>` 的 stop——那是 search agent
的需要，code agent 的输出是代码块，截断反而会毁掉结果。

默认贪心（temperature=0）：per-checkpoint 的指标应当是 checkpoint 的确定性
函数，采样方差另开一组测。

**请求失败必须显式可见**。静默失败会被记成"模型答错"，让整条曲线偏低而
无人察觉——基线阶段那 15% 的 400 就是这么白跑了一轮。
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from threading import Lock
from uuid import uuid4


class ChatPolicy:
    def __init__(self, urls: list[str], model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, workers: int = 32,
                 timeout: int = 180, retries: int = 2,
                 raise_on_failure: bool = False):
        self.urls = [u.rstrip("/") for u in urls] or ["http://localhost:8000/v1"]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self.failures: Counter = Counter()
        self.raise_on_failure = raise_on_failure
        self._failure_lock = Lock()
        self._rr = cycle(range(len(self.urls)))

    def _one(self, messages: list[dict], max_tokens: int) -> str:
        body = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": max_tokens,
        }).encode()
        last = "unknown"
        for attempt in range(self.retries + 1):
            url = self.urls[next(self._rr)] + "/chat/completions"
            try:
                req = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                return d["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
            except Exception as e:
                last = type(e).__name__
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        with self._failure_lock:
            self.failures[last] += 1
        if self.raise_on_failure:
            raise RuntimeError(f"Generation request exhausted retries: {last}")
        return ""          # 空串会被判成没通过；靠 failure_report 让它可见

    def __call__(self, convs: list[list[dict]],
                 max_tokens: int | None = None) -> list[str]:
        mt = max_tokens or self.max_tokens
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda c: self._one(c, mt), convs))

    def failure_report(self) -> str:
        if not self.failures:
            return "请求失败：0"
        tot = sum(self.failures.values())
        detail = "  ".join(f"{k}={v}" for k, v in self.failures.most_common())
        return f"⚠️  请求失败 {tot} 次（会被记成答错，务必核对）: {detail}"


class NativeRolloutPolicy:
    """Use verl's native rollout RPC, which injects the currently loaded LoRA.

    The OpenAI endpoint only knows its base-model registry: the adapter loaded by
    verl's checkpoint engine is passed by ``server.generate`` as a LoRARequest.
    Therefore a plain HTTP request can silently evaluate the base model instead.
    """
    def __init__(self, client, tokenizer, expected_step: int,
                 max_tokens: int = 1024, workers: int = 32):
        self.client = client
        self.tokenizer = tokenizer
        self.expected_step = expected_step
        self.max_tokens = max_tokens
        self.workers = workers
        self.failures: Counter = Counter()
        self.observed_steps: Counter = Counter()
        self._lock = Lock()

    async def _generate(self, messages: list[dict], max_tokens: int) -> str:
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        if isinstance(prompt_ids, Mapping):
            if "input_ids" not in prompt_ids:
                raise TypeError("Tokenizer chat template mapping has no input_ids")
            prompt_ids = prompt_ids["input_ids"]
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            if len(prompt_ids) != 1:
                raise ValueError("Expected one encoded conversation")
            prompt_ids = prompt_ids[0]
        output = await self.client.generate(
            request_id=f"code-eval-{self.expected_step}-{uuid4().hex}",
            prompt_ids=list(prompt_ids),
            sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                             "max_tokens": max_tokens},
        )
        extra = dict(getattr(output, "extra_fields", {}) or {})
        lo = extra.get("min_global_steps", extra.get("global_steps"))
        hi = extra.get("max_global_steps", extra.get("global_steps"))
        if lo != self.expected_step or hi != self.expected_step:
            raise RuntimeError(
                f"Rollout weight-step mismatch: expected={self.expected_step}, "
                f"observed min={lo}, max={hi}")
        with self._lock:
            self.observed_steps[int(lo)] += 1
        return self.tokenizer.decode(output.token_ids, skip_special_tokens=True)

    def _one(self, messages: list[dict], max_tokens: int) -> str:
        try:
            return asyncio.run(self._generate(messages, max_tokens))
        except Exception as exc:
            with self._lock:
                self.failures[type(exc).__name__] += 1
            raise

    def __call__(self, convs: list[list[dict]],
                 max_tokens: int | None = None) -> list[str]:
        mt = max_tokens or self.max_tokens
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda c: self._one(c, mt), convs))

    def failure_report(self) -> str:
        return f"native RPC failures={sum(self.failures.values())}"

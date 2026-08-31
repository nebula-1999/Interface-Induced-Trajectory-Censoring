"""把 vLLM 的 OpenAI 兼容端点包成 batch policy：list[对话] -> list[生成文本]。

和 probes/policy.py 同构，但去掉了 `</tool_call>` 的 stop——那是 search agent
的需要，code agent 的输出是代码块，截断反而会毁掉结果。

默认贪心（temperature=0）：per-checkpoint 的指标应当是 checkpoint 的确定性
函数，采样方差另开一组测。

**请求失败必须显式可见**。静默失败会被记成"模型答错"，让整条曲线偏低而
无人察觉——基线阶段那 15% 的 400 就是这么白跑了一轮。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle


class ChatPolicy:
    def __init__(self, urls: list[str], model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, workers: int = 32,
                 timeout: int = 180, retries: int = 2):
        self.urls = [u.rstrip("/") for u in urls] or ["http://localhost:8000/v1"]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self.failures: Counter = Counter()
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
        self.failures[last] += 1
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

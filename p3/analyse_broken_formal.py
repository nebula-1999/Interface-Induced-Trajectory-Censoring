#!/usr/bin/env python3
"""broken-FC 正式臂（150 步）的完整性校验与训练侧失败层分解。

**为什么单独写一个。** `summarise_p3.py` 给的是整轮汇总数；这里要的是两件它不答的事：

1. **完整性**：每条 extract 事件都带 ``text_sha256``，重算一遍确认全文没被截断也没损坏。
   这直接关系到论文 §7 item 11（「零执行 ≠ 零尝试」）能不能改写——历史 FC run 没存
   rollout 原文，所以那条限制一直挂着。有了可校验的全文，两者才分得开。
2. **训练侧的失败层分解**：论文的 Table 9 是**服务侧**（探针打 vLLM 端点）的分层。
   这里用同一个判据（`analysis/failure_layer.classify`，逐行重放 vLLM 0.27.1 的 hermes
   抽取器）跑**训练循环内部**的每一次生成，看层次分布是否一致。

用法::

    python3 p3/analyse_broken_formal.py [事件目录]
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.failure_layer import classify  # noqa: E402

TIER_ORDER = ("server_parsed", "no_envelope", "bad_payload", "strict_only", "parser_loss")


def load(d: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(d, "events.*.jsonl"))):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("kind") == "extract":
                    out.append(e)
    return out


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else "p3/results/broken_formal"
    ev = load(d)
    if not ev:
        sys.exit(f"没有在 {d} 找到 extract 事件")

    # ---- 完整性 ----
    bad = [e for e in ev
           if e.get("text_sha256") != hashlib.sha256(e["text"].encode("utf-8")).hexdigest()]
    lens = sorted(e["text_chars"] for e in ev)
    print(f"extract 事件 {len(ev)} 条")
    print(f"  sha256 校验不通过: {len(bad)}")
    print(f"  文本长度 最大 {max(lens)} / 中位 {lens[len(lens) // 2]}"
          f" / 超 4000 字符 {sum(1 for c in lens if c > 4000)} 条")

    # ---- 失败层 ----
    tiers = collections.Counter(classify(e["text"], bool(e.get("accepted"))) for e in ev)
    print("\n训练侧失败层：")
    for t in TIER_ORDER:
        print(f"  {t:<14} {tiers[t]:>6}  {100 * tiers[t] / len(ev):>5.2f}%")

    # ---- 漏斗（与论文同口径）----
    print("\n漏斗：")
    print(f"  extract 检查        {len(ev)}")
    print(f"  tight（语义完整）    {sum(1 for e in ev if e.get('tight'))}")
    print(f"  带 <tool_call> 信封  {sum(1 for e in ev if e.get('has_envelope'))}")
    print(f"  被 hermes 接受       {sum(1 for e in ev if e.get('accepted'))}")


if __name__ == "__main__":
    main()

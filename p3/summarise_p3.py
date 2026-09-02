#!/usr/bin/env python3
"""收尾：判定本轮探针是否成立，并给出 emitted / accepted / executed 三个数。

**先判钩子，再判结论。** 三个计数天然都可能是 0，「钩子没装上」与「模型真的
没发出调用」在产物上无法区分。所以任一钩子零触发 → 落 P3_INVALID，不给结论。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

OUT = Path(os.environ.get("P3_OUT", "/root/autodl-tmp/runs/p3_rollout_probe"))
rows = [json.loads(l) for l in (OUT / "events.jsonl").open(encoding="utf-8")
        if l.strip()] if (OUT / "events.jsonl").exists() else []

ext = [r for r in rows if r["kind"] == "extract"]
res = {
    "extract_calls": len(ext),
    "call_tool_events": sum(1 for r in rows if r["kind"] == "call_tool"),
    "code_tool_executes": sum(1 for r in rows if r["kind"] == "execute"),
    "with_envelope": sum(1 for r in ext if r["has_envelope"]),
    "names_run_tests": sum(1 for r in ext if r["names_run_tests"]),
    "emitted_vllm_acceptable": sum(1 for r in ext if r["vllm_would_accept"]),
    "emitted_verl_acceptable": sum(1 for r in ext if r["verl_would_accept"]),
    "emitted_tight": sum(1 for r in ext if r["tight"]),
    "accepted_total": sum(r["accepted"] for r in ext),
    "accepted_name_hist": dict(Counter(
        n for r in ext for n in (r.get("accepted_names") or []) if n)),
    "parsers_seen": sorted({r["parser"] for r in ext}),
}

dead = []
if not ext:
    dead.append("extract_tool_calls")          # 这个必须触发，否则整轮无意义
res["valid"] = not dead
res["dead_hooks"] = dead

if not dead:
    e, a, x = res["emitted_tight"], res["accepted_total"], res["code_tool_executes"]
    if e > 0 and a == 0 and x == 0:
        # 措辞要精确：不是「parser 错误地拒绝了合规调用」。实测机制是模型发出了
        # 语义正确、载荷完整的裸 JSON 调用但**缺少 <tool_call> 包装层**，因此
        # 栈里每一个 parser 看不见它都是正确行为。两种说法的可操作含义完全不同：
        # 前者要修 parser，后者要修模型—模板—parser 这份三方契约。
        res["verdict"] = ("闭合：训练栈内发出了语义正确的工具调用，但缺包装层，"
                          "栈内无一 parser 接受，零执行零 observation")
    elif e == 0:
        res["verdict"] = "未闭合：训练栈内没有发出合规调用——与 1.5B 同类，是策略问题"
    else:
        res["verdict"] = f"混合：emitted={e} accepted={a} executed={x}，需逐条看"
else:
    res["verdict"] = "★ 作废：钩子未触发，三个 0 没有信息"
    (OUT / "P3_INVALID").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

(OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
sys.exit(0 if res["valid"] else 2)

#!/usr/bin/env python3
"""收尾：判定本轮探针是否成立，并给出 emitted / accepted / executed 三个数。

**先判钩子，再判结论。** 三个运行时计数天然都可能是 0，尤其 broken 臂本来就
应当零执行；因此不能用事件数证明安装成功。三个 patch 点分别写 installation
record，任一缺失或失败才落 P3_INVALID；parser 还必须至少产生一条运行时记录。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

OUT = Path(os.environ.get("P3_OUT", "/root/autodl-tmp/runs/p3_rollout_probe"))
event_files = sorted(OUT.glob("events.*.jsonl"))
rows = []
for path in event_files:
    with path.open(encoding="utf-8") as fh:
        rows.extend(json.loads(l) for l in fh if l.strip())

ext = [r for r in rows if r["kind"] == "extract"]
launch_rc_path = OUT / "launch_rc"
launch_rc = int(launch_rc_path.read_text().strip()) if launch_rc_path.exists() else None
res = {
    "launch_rc": launch_rc,
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
    "event_files": [p.name for p in event_files],
    "parser_hook_installed": any(r["kind"] == "install_parser" and r.get("ok")
                                 for r in rows),
    "agentloop_hook_installed": any(r["kind"] == "install_agentloop" and r.get("ok")
                                    for r in rows),
    "trajectory_identity_installed": any(r["kind"] == "install_agentloop"
                                          and r.get("identity_ok") for r in rows),
    "code_tool_hook_installed": any(r["kind"] == "install_code_tool" and r.get("ok")
                                    for r in rows),
    "custom_parser_registered": any(r["kind"] == "register_custom_parser" and r.get("ok")
                                    for r in rows),
}

dead = []
if launch_rc != 0:
    dead.append(f"launch_ppo rc={launch_rc!r}")
for key, label in (("parser_hook_installed", "extract_tool_calls install"),
                   ("agentloop_hook_installed", "_call_tool install"),
                   ("trajectory_identity_installed", "trajectory identity install"),
                   ("code_tool_hook_installed", "CodeTool.execute install"),
                   ("custom_parser_registered", "qwen2_5_coder registration")):
    if not res[key]:
        dead.append(label)
if not ext:
    dead.append("extract_tool_calls runtime")
res["valid"] = not dead
res["valid_scope"] = "launch-and-basic-instrumentation-only; NOT learning acceptance"
res["dead_hooks"] = dead

if not dead:
    e, a, x = res["emitted_tight"], res["accepted_total"], res["code_tool_executes"]
    if e > 0 and a == 0 and x == 0:
        # Heuristic matches do not establish JSON/schema validity or causality.
        res["verdict"] = ("存在 tight 启发式候选，accepted=0、execute事件=0；"
                          "需独立校验 JSON/schema 及 observation，不能单凭 regex 宣告机制闭合")
    elif e == 0:
        res["verdict"] = "未检出 tight 启发式候选；不能据此断言模型从未尝试调用"
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

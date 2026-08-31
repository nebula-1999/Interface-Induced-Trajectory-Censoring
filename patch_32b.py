#!/usr/bin/env python3
"""只补跑 32B 那几条因上下文溢出而出错的题，再合并回原 JSONL。

原因：32B 用 max_model_len=4096，ReAct 多轮轨迹（含 Observation 报错）
超出上下文导致请求失败。补跑时提到 8192，按 clean_index 精确定位。
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-Coder-32B-Instruct")
ap.add_argument("--dir", default="/root/autodl-tmp/code-agent")
a = ap.parse_args()
D = Path(a.dir)

TARGETS = {"react_optional": [26, 35, 53], "react_mandatory": [9, 38]}

for cond, ids in TARGETS.items():
    proto, strength = cond.split("_")
    jl = D / f"traj_32B_{cond}.jsonl"
    recs = [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]
    tmp = D / f"patch_32B_{cond}.jsonl"
    cmd = [sys.executable, str(D / "probe_react_full.py"),
           "--model", a.model, "--port", str(a.port), "--n", "0",
           "--protocol", proto, "--strength", strength,
           "--out", str(tmp), "--only-ids", ",".join(map(str, ids))]
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not tmp.exists():
        print(f"FAIL {cond} rc={rc}")
        continue
    patched = {json.loads(l)["clean_index"]: json.loads(l)
               for l in open(tmp, encoding="utf-8") if l.strip()}
    merged, n_rep = [], 0
    for r in recs:
        if r["clean_index"] in patched:
            merged.append(patched[r["clean_index"]]); n_rep += 1
        else:
            merged.append(r)
    with open(jl, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK {cond}: 替换 {n_rep}/{len(ids)} 条，总 {len(merged)} 条")

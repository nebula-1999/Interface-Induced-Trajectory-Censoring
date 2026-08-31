#!/usr/bin/env python3
"""用沙箱跑 KodCode 官方 solution 对自己的 test。

两个目的：
  1. 数据质量——官方解跑不过自己测试的题必须剔掉。留着的话奖励函数
     永远给不到满分，模型在那些题上学到的全是噪声。
  2. 标定 rollout 开销——单题耗时决定训练时每个 step 要花多久。

用法：python verify_kodcode.py [n]    n 省略则跑全部干净样本
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

from datasets import load_dataset

from sandbox import run_many, safe_workers

n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None

kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
idxs = clean[:n_arg] if n_arg else clean

w = safe_workers()
print(f"待验 {len(idxs)} 条（干净集共 {len(clean)}），并行度 {w}"
      f"（按 cgroup 内存算，不是按 112 核）", flush=True)

items = [(kc[i]["solution"], kc[i]["test"]) for i in idxs]
t0 = time.time()
results = run_many(items, workers=w)
el = time.time() - t0

ok = [i for i, r in zip(idxs, results) if r.all_passed]
bad = [(i, r) for i, r in zip(idxs, results) if not r.all_passed]

print(f"\n耗时 {el:.1f}s  单题均值 {el / len(idxs):.2f}s  "
      f"吞吐 {len(idxs) / el:.1f}/s")
print(f"官方解通过: {len(ok)}/{len(idxs)} = {len(ok) / len(idxs):.1%}")
print(f"状态分布: {dict(Counter(r.status for r in results))}")

durs = sorted(r.duration for r in results)
print(f"单题耗时 中位={durs[len(durs) // 2]:.2f}s "
      f"p90={durs[int(len(durs) * .9)]:.2f}s max={durs[-1]:.2f}s")

if bad:
    print(f"\n跑不过的前 5 条：")
    for i, r in bad[:5]:
        print(f"  #{i} status={r.status} passed={r.passed} failed={r.failed} "
              f"errors={r.errors} | {r.stderr.strip()[-160:]}")

out = "verified_ids.json" if n_arg is None else f"verified_ids_{n_arg}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({"n_checked": len(idxs), "verified_index": ok}, f)
print(f"\n已写出 {out}（{len(ok)} 条可用）")

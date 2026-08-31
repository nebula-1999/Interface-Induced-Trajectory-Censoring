#!/usr/bin/env python3
"""去污染是否有偏——剔掉的 2331 条如果集中在某个难度/来源，训练集分布会歪。"""

from __future__ import annotations

import json
from collections import Counter

from datasets import load_dataset

kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
keep = set(json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"])
print(f"全量 {len(kc)} → 保留 {len(keep)}（剔除 {len(kc) - len(keep)}）\n")

for field in ("gpt_difficulty", "subset"):
    vals = kc[field]
    all_c = Counter(vals)
    keep_c = Counter(v for i, v in enumerate(vals) if i in keep)
    print(f"=== {field} ===")
    print(f"{'':16s} {'全量':>7} {'保留':>7} {'保留率':>8}")
    for k, n in all_c.most_common():
        r = keep_c[k] / n
        flag = "  <-- 剔除偏高" if r < 0.65 else ""
        print(f"{str(k):16s} {n:>7d} {keep_c[k]:>7d} {r:>7.1%}{flag}")
    print()

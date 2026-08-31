#!/usr/bin/env python3
"""看 KodCode-Light-RL-10K 的实际格式与构成。

关心三件事：
  1. question 是不是函数补全式（和 HumanEval+/MBPP+ 同源），还是竞赛式
  2. test 是不是可直接执行的 pytest
  3. subset / 难度分布——决定采样策略
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
print(f"n = {len(ds)}\n")

print("subset 分布:", Counter(ds["subset"]).most_common())
print("style 分布:", Counter(ds["style"]).most_common())
print("难度分布:", Counter(ds["gpt_difficulty"]).most_common())

for i in (0, 1):
    r = ds[i]
    print(f"\n{'=' * 60}\n#{i}  subset={r['subset']}  difficulty={r['gpt_difficulty']}")
    print(f"--- question ---\n{r['question'][:600]}")
    print(f"--- solution ---\n{r['solution'][:400]}")
    print(f"--- test ---\n{r['test'][:500]}")

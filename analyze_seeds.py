#!/usr/bin/env python3
"""两个独立训练 seed 的一致性检验。

seed 同时作用于数据采样与 rollout 采样，所以这是**含训练方差**的误差棒
——正是 repro-variance 那条线批评各家论文缺失的那种。

主结论若在两个 seed 间一致，"增益全在初稿"就不再是一次观察，而是稳定现象。
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = {"seed 42": Path("runs/step3"), "seed 1": Path("runs/seed1")}
STEPS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]


def load(d: Path, step: int):
    multi, rep = {}, {}
    f = d / f"step_{step:05d}.jsonl"
    if not f.exists():
        return None, None
    for line in open(f, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            (multi if r["channel"] == "multi" else rep)[r["task_id"]] = r
    return multi, rep


def fin(r): return bool(r["turns"]) and r["turns"][-1]["all_passed"]
def t1(r):  return bool(r["turns"]) and r["turns"][0]["all_passed"]


print("=" * 72)
print("多轮救回的绝对题数（542 题中，靠 turn2+ 救回来的）")
print(f"{'step':>6}" + "".join(f"{k:>12}" for k in RUNS))
for s in STEPS:
    row = f"{s:>6}"
    for k, d in RUNS.items():
        m, _ = load(d, s)
        row += f"{sum(1 for v in m.values() if fin(v) and not t1(v)):>12}" if m else f"{'-':>12}"
    print(row)

print("\n" + "=" * 72)
print("step 0 → 150 的变化")
print(f"{'':10}{'final':>16}{'turn-1':>16}{'repair':>16}")
summary = {}
for k, d in RUNS.items():
    m0, r0 = load(d, 0)
    m1, r1 = load(d, 150)
    if not m0 or not m1:
        continue
    n, mm = len(m0), len(r0)
    f0, f1 = sum(map(fin, m0.values())) / n, sum(map(fin, m1.values())) / n
    a0, a1 = sum(map(t1, m0.values())) / n, sum(map(t1, m1.values())) / n
    p0, p1 = sum(map(fin, r0.values())) / mm, sum(map(fin, r1.values())) / mm
    summary[k] = (f1 - f0, a1 - a0, p1 - p0)
    print(f"{k:10}{f0:6.1%}→{f1:6.1%}  {a0:6.1%}→{a1:6.1%}  {p0:6.1%}→{p1:6.1%}")
print(f"\n{'':10}{'Δfinal':>16}{'Δturn-1':>16}{'Δrepair':>16}")
for k, (df, da, dp) in summary.items():
    print(f"{k:10}{df:>+15.1%}{da:>+15.1%}{dp:>+15.1%}")
if len(summary) == 2:
    a, b = summary.values()
    print(f"{'seed 间差':10}{abs(a[0]-b[0]):>15.1%}{abs(a[1]-b[1]):>15.1%}{abs(a[2]-b[2]):>15.1%}")

print("\n" + "=" * 72)
print("核心分解：新通过的题里，第一轮就过的占比")
for k, d in RUNS.items():
    m0, _ = load(d, 0)
    m1, _ = load(d, 150)
    if not m0 or not m1:
        continue
    keys = sorted(set(m0) & set(m1))
    gained = [x for x in keys if not fin(m0[x]) and fin(m1[x])]
    g1 = sum(1 for x in gained if t1(m1[x]))
    print(f"  {k}: 新通过 {len(gained):2d} 题，其中第一轮就过 {g1:2d} 题 "
          f"= {g1/max(len(gained),1):.1%}（靠多轮救回 {len(gained)-g1} 题）")

#!/usr/bin/env python3
"""Step 3 主分析：RL 的增益到底落在哪一轮。

用 per-task 配对数据做 McNemar 检验。**这个检验外部做不了**——公开论文只发
聚合百分数，而配对检验需要每道题的成败。这也是本 repo 的差异化资产。

三个检验：
  1. final_pass  step0 vs step150   —— RL 到底有没有用
  2. turn1_pass  step0 vs step150   —— 初稿质量变了多少
  3. repair_rate step0 vs step150   —— 修正已知错误的能力变了多少

外加一个分解：step150 新通过的题里，有多少是**第一轮就过**的。
这个比例直接回答"增益落在 turn-1 还是 turn-N"。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

D = Path("runs/step3")


def load(step: int) -> tuple[dict, dict]:
    """返回 (multi, repair)，均为 {task_id: 记录}。"""
    multi, rep = {}, {}
    for line in open(D / f"step_{step:05d}.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        (multi if r["channel"] == "multi" else rep)[r["task_id"]] = r
    return multi, rep


def passed_final(r) -> bool:
    return bool(r["turns"]) and r["turns"][-1]["all_passed"]


def passed_turn1(r) -> bool:
    return bool(r["turns"]) and r["turns"][0]["all_passed"]


def mcnemar(a: dict, b: dict, fn) -> tuple[int, int, float, float]:
    """b = 前对后错，c = 前错后对；返回 (b, c, z, 双侧 p)。"""
    keys = sorted(set(a) & set(b))
    nb = sum(1 for k in keys if fn(a[k]) and not fn(b[k]))
    nc = sum(1 for k in keys if not fn(a[k]) and fn(b[k]))
    n = nb + nc
    if n == 0:
        return nb, nc, 0.0, 1.0
    # 连续性校正的 McNemar
    z = (abs(nc - nb) - 1) / math.sqrt(n) if n > 0 else 0.0
    z = z if nc >= nb else -z
    p = math.erfc(abs(z) / math.sqrt(2))
    return nb, nc, z, p


m0, r0 = load(0)
m150, r150 = load(150)
N, M = len(m0), len(r0)
print(f"配对样本：多轮通道 {N} 题，修复通道 {M} 题\n")

print("=" * 68)
print("McNemar 配对检验（step 0 → step 150）")
print(f"{'指标':16s} {'step0':>7s} {'step150':>8s} {'Δ':>7s} "
      f"{'退步':>5s} {'进步':>5s} {'z':>6s} {'p':>9s}")
print("-" * 68)

for label, a, b, fn, n in (
    ("final_pass", m0, m150, passed_final, N),
    ("turn1_pass", m0, m150, passed_turn1, N),
    ("repair_rate", r0, r150, passed_final, M),
):
    p0 = sum(fn(v) for v in a.values()) / n
    p1 = sum(fn(v) for v in b.values()) / n
    nb, nc, z, p = mcnemar(a, b, fn)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{label:16s} {p0:6.1%} {p1:7.1%} {p1 - p0:+6.1%} "
          f"{nb:5d} {nc:5d} {z:6.2f} {p:8.4f}{star}")

print("\n" + "=" * 68)
print("核心分解：step 150 新通过的题，是第一轮就过的吗")
keys = sorted(set(m0) & set(m150))
gained = [k for k in keys if not passed_final(m0[k]) and passed_final(m150[k])]
lost = [k for k in keys if passed_final(m0[k]) and not passed_final(m150[k])]
g_t1 = sum(1 for k in gained if passed_turn1(m150[k]))
print(f"  新通过 {len(gained)} 题，其中 **第一轮就过** {g_t1} 题 "
      f"= {g_t1 / max(len(gained), 1):.1%}")
print(f"  靠多轮才救回来的只有 {len(gained) - g_t1} 题")
print(f"  （同时退步 {len(lost)} 题）")

print("\n" + "=" * 68)
print("多轮贡献的绝对量（靠 turn2+ 救回来的题数）")
for step in (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150):
    try:
        m, _ = load(step)
    except FileNotFoundError:
        continue
    rescued = sum(1 for v in m.values() if passed_final(v) and not passed_turn1(v))
    print(f"  step {step:3d}: {rescued:3d} 题 / {len(m)} = {rescued / len(m):5.2%}")

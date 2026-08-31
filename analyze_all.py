#!/usr/bin/env python3
"""报告里全部四张表的唯一来源。

    python analyze_all.py

固化在这里而不是散在临时脚本里，是因为报告的可信度依赖"所有数字出自同一套
代码、同一套口径"。尤其是 runs/baseline 那份数据——两次 baseline 都写 step 0
并**追加**进同一个文件，必须按写入顺序切分，用 dict 加载会被后写的覆盖
（我第一次就这么错过，把通用 1.5B 的数字标成了 Coder 版）。

输出四张表：
  1. 主检验   GRPO / RLOO 的 step 0 → 150，McNemar 配对
  2. seed     两个独立训练 seed 的一致性
  3. 算法     GRPO vs RLOO（同超参，只换 advantage 估计器）
  4. 规模     1.5B 通用 / 1.5B Coder / 7B Coder 的多轮价值
"""

from __future__ import annotations

import json
import math
from pathlib import Path

N_MULTI, N_REPAIR = 542, 454

# EvalPlus 自身已确认的两条数据缺陷，官方参考解都过不了：
#   HumanEval/32 —— test 断言写错（_poly 参数顺序反且把 float 用 * 展开）
#   Mbpp/599     —— 生成测试含 n≈5e11，参考解自身就超时
# 它们对任何模型都必然失败，因此：
#   - 对 step0 vs step150 的**比较**无影响（两边都算错，McNemar 的 b=c=0）
#   - 但会把**绝对 pass@1 低估 0.37%**，且与公开 EvalPlus 口径不可比
# 主表一律排除，分母 542 → 540。探针通道早已剔除（456 → 454）。
KNOWN_DEFECTS = {"HumanEval/32", "Mbpp/599"}
RUNS = {"GRPO seed42": "runs/step3", "GRPO seed1": "runs/seed1", "RLOO seed42": "runs/rloo"}
STEPS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]


# --- 载入 -------------------------------------------------------------------

def load_step(d: str, step: int):
    """按 channel 分开。同一 run 内 task_id 唯一，可以安全用 dict。"""
    m, r = {}, {}
    f = Path(d) / f"step_{step:05d}.jsonl"
    if not f.exists():
        return None, None
    for line in open(f, encoding="utf-8"):
        if line.strip():
            x = json.loads(line)
            if x["task_id"] in KNOWN_DEFECTS:
                continue
            (m if x["channel"] == "multi" else r)[x["task_id"]] = x
    return m, r


def load_appended(path: str, k: int):
    """一个文件里追加了多份 run 时，按写入顺序取第 k 份（0-based）。

    **不能用 dict 载入**：两份 run 的 task_id 完全相同，后写的会覆盖先写的，
    结果是静默拿到最后一份而毫无报错。
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    blk = N_MULTI + N_REPAIR
    seg = rows[k * blk:(k + 1) * blk]
    assert len(seg) == blk, f"{path} 第 {k} 份长度 {len(seg)}，期望 {blk}"
    multi, rep = seg[:N_MULTI], seg[N_MULTI:]
    assert all(x["channel"] == "multi" for x in multi)
    assert all(x["channel"] == "repair" for x in rep)
    # 与 load_step 保持同一口径
    multi = [x for x in multi if x["task_id"] not in KNOWN_DEFECTS]
    rep = [x for x in rep if x["task_id"] not in KNOWN_DEFECTS]
    return multi, rep


def fin(x): return bool(x["turns"]) and x["turns"][-1]["all_passed"]
def t1(x):  return bool(x["turns"]) and x["turns"][0]["all_passed"]


def mcnemar(a: dict, b: dict, fn):
    """连续性校正的 McNemar。返回 (前对后错, 前错后对, z, 双侧 p)。"""
    keys = sorted(set(a) & set(b))
    nb = sum(1 for k in keys if fn(a[k]) and not fn(b[k]))
    nc = sum(1 for k in keys if not fn(a[k]) and fn(b[k]))
    n = nb + nc
    if n == 0:
        return nb, nc, 0.0, 1.0
    z = (abs(nc - nb) - 1) / math.sqrt(n)
    z = z if nc >= nb else -z
    return nb, nc, z, math.erfc(abs(z) / math.sqrt(2))


def star(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


# --- 表 1：主检验 -----------------------------------------------------------

print("=" * 78)
print("表 1  主检验：step 0 → 150，McNemar 配对")
print(f"{'run':14}{'指标':13}{'step0':>8}{'step150':>9}{'Δ':>8}{'退':>4}{'进':>4}{'z':>7}{'p':>9}")
print("-" * 78)
for run, d in RUNS.items():
    m0, r0 = load_step(d, 0)
    m1, r1 = load_step(d, 150)
    if m0 is None or m1 is None:
        print(f"{run:14}(数据缺失)")
        continue
    for label, a, b, fn in (("final_pass", m0, m1, fin),
                            ("turn1_pass", m0, m1, t1),
                            ("repair_rate", r0, r1, fin)):
        n = len(a)
        p0, p1 = sum(map(fn, a.values())) / n, sum(map(fn, b.values())) / n
        nb, nc, z, p = mcnemar(a, b, fn)
        print(f"{run:14}{label:13}{p0:7.1%}{p1:8.1%}{p1-p0:+8.1%}"
              f"{nb:4d}{nc:4d}{z:7.2f}{p:8.3f}{star(p)}")
    print()

# --- 表 2：核心分解 + seed 一致性 -------------------------------------------

print("=" * 78)
print("表 2  核心分解与 seed 一致性")
print(f"{'run':14}{'新通过':>8}{'其中首轮就过':>14}{'占比':>8}{'多轮救回':>10}")
print("-" * 78)
for run, d in RUNS.items():
    m0, _ = load_step(d, 0)
    m1, _ = load_step(d, 150)
    if m0 is None or m1 is None:
        continue
    keys = sorted(set(m0) & set(m1))
    gained = [k for k in keys if not fin(m0[k]) and fin(m1[k])]
    g1 = sum(1 for k in gained if t1(m1[k]))
    print(f"{run:14}{len(gained):8d}{g1:14d}{g1/max(len(gained),1):8.1%}"
          f"{len(gained)-g1:10d}")

print(f"\n多轮救回的绝对题数（540 题中，靠 turn2+ 救回；已排除 2 条 EvalPlus 缺陷）")
print(f"{'step':>6}" + "".join(f"{k:>14}" for k in RUNS))
for s in STEPS:
    row = f"{s:>6}"
    for d in RUNS.values():
        m, _ = load_step(d, s)
        row += f"{sum(1 for v in m.values() if fin(v) and not t1(v)):>14}" if m else f"{'-':>14}"
    print(row)

# --- 表 3：算法对比 ---------------------------------------------------------

print("\n" + "=" * 78)
print("表 3  算法对比：同 seed 42、同全部超参，只换 advantage 估计器")
print(f"{'':14}{'Δfinal':>10}{'Δturn-1':>10}{'救回 0→150':>14}{'首轮占比':>10}")
for run in ("GRPO seed42", "RLOO seed42"):
    d = RUNS[run]
    m0, _ = load_step(d, 0); m1, _ = load_step(d, 150)
    n = len(m0)
    df = sum(map(fin, m1.values()))/n - sum(map(fin, m0.values()))/n
    da = sum(map(t1, m1.values()))/n - sum(map(t1, m0.values()))/n
    g0 = sum(1 for v in m0.values() if fin(v) and not t1(v))
    g1 = sum(1 for v in m1.values() if fin(v) and not t1(v))
    keys = sorted(set(m0) & set(m1))
    gained = [k for k in keys if not fin(m0[k]) and fin(m1[k])]
    gt1 = sum(1 for k in gained if t1(m1[k]))
    print(f"{run:14}{df:>+10.1%}{da:>+10.1%}{f'{g0}→{g1}':>14}"
          f"{gt1/max(len(gained),1):>10.1%}")

# --- 表 4：模型规模 ---------------------------------------------------------

print("\n" + "=" * 78)
print("表 4  模型规模/类型：多轮的价值（均为未训练 base model，同一评测协议）")
print(f"{'模型':24}{'turn-1':>9}{'final':>9}{'gap':>9}{'救回':>8}{'repair':>9}")
print("-" * 78)
BASE = "runs/baseline/step_00000.jsonl"
entries = [("Qwen2.5-Coder-1.5B", load_appended(BASE, 0)),
           ("Qwen2.5-1.5B（通用）", load_appended(BASE, 1))]
r7 = [json.loads(l) for l in open("runs/eval7b/step_00000.jsonl", encoding="utf-8") if l.strip()]
entries.append(("Qwen2.5-Coder-7B",
                ([x for x in r7 if x["channel"] == "multi"],
                 [x for x in r7 if x["channel"] == "repair"])))
for name, (mm, rr) in sorted(entries, key=lambda e: sum(map(t1, e[1][0]))):
    n = len(mm)
    a = sum(map(t1, mm))/n; f = sum(map(fin, mm))/n
    p = sum(map(fin, rr))/len(rr) if rr else float("nan")
    resc = sum(1 for v in mm if fin(v) and not t1(v))
    print(f"{name:24}{a:8.1%}{f:8.1%}{f-a:8.1%}{resc:8d}{p:9.1%}")
print("\n注：gap 不是能力的单调函数——通用 1.5B 的 gap 高于同规模 Coder 版，")
print("    因为它更愿意重试（Coder 版 23.5% 的失败题一个字都不改，通用版仅 3.3%）。")
print("    单调的是 repair_rate：47.1% → 53.5% → 82.4%。")

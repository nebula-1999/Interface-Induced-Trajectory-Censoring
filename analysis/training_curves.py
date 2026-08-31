#!/usr/bin/env python3
"""训练过程分析与制图。

数据源两类：
  1. runs/{step3,seed1,rloo}/step_*.jsonl —— 三个 150 步 run 的逐题评测产物（本地已有）
  2. runs/final/full*.log —— 原始训练日志（含 loss/entropy/kl/grad_norm，
     **目前仅在服务器**，见 BOOT_CHECKLIST.md §0；缺失时自动跳过第 3 张图）

产出 figures/fig4_training_curves.png 与一份文字分析。
"""
import json, os, re, sys, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
KNOWN_DEFECTS = {"HumanEval/32", "Mbpp/599"}
RUNS = {"GRPO seed42": "runs/step3", "GRPO seed1": "runs/seed1", "RLOO seed42": "runs/rloo"}
STEPS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

def load_step(d, step):
    f = os.path.join(ROOT, d, f"step_{step:05d}.jsonl")
    if not os.path.exists(f): return None, None
    m, r = {}, {}
    for line in open(f, encoding="utf-8"):
        if not line.strip(): continue
        x = json.loads(line)
        if x["task_id"] in KNOWN_DEFECTS: continue
        (m if x["channel"] == "multi" else r)[x["task_id"]] = x
    return m, r

def fin(x): return bool(x["turns"]) and x["turns"][-1]["all_passed"]
def t1(x):  return bool(x["turns"]) and x["turns"][0]["all_passed"]

series = {}
for name, d in RUNS.items():
    rows = []
    for s in STEPS:
        m, rp = load_step(d, s)
        if m is None: continue
        n = len(m)
        rows.append(dict(step=s, n=n,
                         turn1=sum(map(t1, m.values())) / n,
                         final=sum(map(fin, m.values())) / n,
                         repair=sum(map(fin, rp.values())) / len(rp) if rp else float("nan"),
                         rescue=sum(1 for v in m.values() if fin(v) and not t1(v))))
    if rows: series[name] = rows

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
colors = {"GRPO seed42": "#c0392b", "GRPO seed1": "#e67e22", "RLOO seed42": "#2980b9"}

ax = axes[0]
for name, rows in series.items():
    ax.plot([r["step"] for r in rows], [r["final"]*100 for r in rows], "o-",
            color=colors[name], lw=1.8, ms=4, label=f"{name} final")
    ax.plot([r["step"] for r in rows], [r["turn1"]*100 for r in rows], "--",
            color=colors[name], lw=1.2, alpha=0.55)
ax.set_xlabel("training step"); ax.set_ylabel("pass@1 (%)")
ax.set_title("EvalPlus multi channel (n=540)\nsolid = final, dashed = turn-1", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")

ax = axes[1]
for name, rows in series.items():
    ax.plot([r["step"] for r in rows], [r["rescue"] for r in rows], "o-",
            color=colors[name], lw=1.8, ms=4, label=name)
ax.set_xlabel("training step"); ax.set_ylabel("items rescued by turn ≥2")
ax.set_ylim(0, 20)
ax.set_title("Multi-turn rescues: flat across 150 steps\n(this is the puzzle the paper explains)", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5)

ax = axes[2]
for name, rows in series.items():
    ax.plot([r["step"] for r in rows], [r["repair"]*100 for r in rows], "o-",
            color=colors[name], lw=1.8, ms=4, label=name)
ax.set_xlabel("training step"); ax.set_ylabel("repair-channel pass@1 (%)")
ax.set_title("Seeded-bug repair channel (n=454)", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("RL improves first drafts; multi-turn repair never moves", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(ROOT, "figures", "fig4_training_curves.png")); plt.close(fig)

print("=== 逐 run 的起止与变化 ===")
print(f"{'run':<14}{'turn-1':>16}{'final':>16}{'rescue':>14}{'repair':>16}")
for name, rows in series.items():
    a, b = rows[0], rows[-1]
    print(f"{name:<14}{a['turn1']:>7.1%}→{b['turn1']:<7.1%}{a['final']:>8.1%}→{b['final']:<7.1%}"
          f"{a['rescue']:>6d}→{b['rescue']:<6d}{a['repair']:>8.1%}→{b['repair']:<7.1%}")

print("\n=== 救回数的全程波动（关键：有没有趋势）===")
for name, rows in series.items():
    v = [r["rescue"] for r in rows]
    print(f"  {name:<14} {v}   min={min(v)} max={max(v)} 末-首={v[-1]-v[0]:+d}")

print("\n=== 增益归因：新通过的题里有多少是首轮就过的 ===")
for name, d in RUNS.items():
    m0, _ = load_step(d, 0); m1, _ = load_step(d, 150)
    if m0 is None or m1 is None: continue
    K = sorted(set(m0) & set(m1))
    gained = [k for k in K if not fin(m0[k]) and fin(m1[k])]
    g1 = sum(1 for k in gained if t1(m1[k]))
    print(f"  {name:<14} 新通过 {len(gained):>3}   其中首轮就过 {g1:>3} = {g1/max(len(gained),1):.0%}"
          f"   靠多轮救回 {len(gained)-g1:>2}")

logs = glob.glob(os.path.join(ROOT, "runs", "final", "full*.log"))
print(f"\n=== 原始训练日志 ===")
if logs:
    print(f"  找到 {len(logs)} 份，可分析 loss / entropy / kl / grad_norm")
else:
    print("  ❌ full*.log 不在本地 —— 见 BOOT_CHECKLIST.md §0，开机后 rsync 拉回")
    print("     缺失影响：无法回答'训练是否稳定、有无 KL 崩溃'；pass@1 数字不受影响")
print("\n图已写入 figures/fig4_training_curves.png")

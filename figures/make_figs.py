#!/usr/bin/env python3
"""Three main figures. All data from runs/final trajectories; definitions match the writeup."""
import json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "runs", "final")
OUT = os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

def load(f):
    p = os.path.join(D, f)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from intent import classify   # 唯一判据，正文与图共用

# ---------- Fig 1: intent-vs-parse gap across scale ----------
scales = ["1.5B", "3B", "7B", "14B", "32B"]; xs = [1.5, 3, 7, 14, 32]
parsed, tight = [], []
for s in scales:
    c = classify(os.path.join(D, f"traj_v5_Qwen{s}_fc_intent.jsonl"))
    parsed.append(c["parsed"]); tight.append(c["tight"])

fig, ax = plt.subplots(figsize=(6.4, 4.1))
ax.plot(xs, tight, "o-", color="#c0392b", lw=2.2, ms=7, label="Well-formed calls the model actually emits")
ax.plot(xs, parsed, "s--", color="#2c3e50", lw=2.2, ms=7, label="Calls the server (hermes) parses")
ax.fill_between(xs, parsed, tight, color="#c0392b", alpha=0.12)
for x, y in zip(xs, tight):
    ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(scales)
ax.set_xlabel("Qwen2.5-Coder parameters"); ax.set_ylabel("items out of 100")
ax.set_ylim(-4, 94)
ax.set_title("Capability silently discarded by interface mismatch grows with scale\n"
             "tool_choice=auto, hermes parser, n=100 per scale", fontsize=10.5)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.text(2.4, 62, "silently discarded", color="#c0392b", fontsize=10, style="italic")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_intent_parse_gap.png")); plt.close(fig)

# ---------- Fig 2: four-family failure taxonomy ----------
# Mistral 不设 "best config" 柱：四种配置解析出 2/5/1/2、错误 2/42/3/39，
# 解析最多的那个恰是崩掉 42% 请求的那个。用阴影柱+注记如实表示"无可用配置"。
fams = ["DeepSeek-Coder\ntemplate omits tools", "Qwen2.5-Coder\nparser mismatch",
        "Llama-3.1-8B\nschema hallucination", "Mistral-7B\ntoken-level repeat"]
before = [0, 0, 74, 2]     # default config, parsed calls per 100
after  = [0, 84, 98, 5]    # best-known config (Mistral: highest-parse config = official template)
err_after = [0, 0, 0, 42]  # request errors in that config
fix = ["no remedy", "adapter\n0 -> 84", "strict:true\n23 -> 0", "no usable config"]
fig, ax = plt.subplots(figsize=(8.0, 4.5))
x = range(4); w = 0.36
ax.bar([i - w/2 for i in x], before, w, label="default configuration", color="#95a5a6")
bars = ax.bar([i + w/2 for i in x], after, w, label="best-known configuration", color="#27ae60")
bars[3].set_color("#e67e22"); bars[3].set_hatch("///")   # Mistral: 该配置有 42/100 请求失败
for i, (b, a, f) in enumerate(zip(before, after, fix)):
    ax.text(i + w/2, a + 2, str(a), ha="center", fontsize=9)
    ax.text(i - w/2, b + 2, str(b), ha="center", fontsize=9, color="#555")
    ax.text(i, -17, f, ha="center", fontsize=8.5, color="#c0392b")
ax.annotate("all 4 configs error\n(2-42 failed requests / 100)",
            xy=(3 + w/2, 5), xytext=(2.55, 46), fontsize=8.5, color="#e67e22",
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2))
ax.set_xticks(list(x)); ax.set_xticklabels(fams, fontsize=8.5)
ax.set_ylabel("tool calls parsed by server / 100 items"); ax.set_ylim(-26, 116)
ax.set_title("Four families, four failure layers, different remedies", fontsize=11)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_family_taxonomy.png")); plt.close(fig)

# ---------- Fig 3: training-side causal link ----------
fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4))
labels = ["FC\n(hermes)", "ReAct"]; colors = ["#95a5a6", "#27ae60"]
for ax, (vals, title, ylab) in zip(axes, [
        ([0.0, 2.052], "Tool calls per rollout", "calls"),
        ([2.000, 5.883], "Dialogue turns (num_turns/mean)", "turns"),
        ([0.0, 12.21], "Tool-call time\n(verl agent_loop timer, mean)", "s")]):
    ax.bar(labels, vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.04, f"{v:g}", ha="center", fontsize=10)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylab)
    ax.set_ylim(0, max(vals) * 1.28)
fig.suptitle("RL training rollouts: under FC the tool is never successfully called\n"
             "Qwen2.5-Coder-1.5B, verl GRPO, mandatory tool-use prompt on both arms", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(OUT, "fig3_training_causal.png")); plt.close(fig)

print("parsed:", parsed)
print("tight :", tight)
print("figures written")

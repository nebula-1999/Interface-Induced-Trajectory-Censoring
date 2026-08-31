#!/usr/bin/env python3
"""五层漏斗：意图 → 服务端解析 → 工具执行 → Observation 回流 → 多轮救回。

两处必须协议感知，否则会造出误导性的 0：
  · ①「模型产出合格调用」只能在**服务端解析失败**时测量 —— 解析成功时
    调用被移入 tool_calls 而 content 为空，对 content 判 tight 恒为 0。
    这类臂标 N/A，不标 0。
  · ②「服务端解析出调用」的 parse_mode 因协议而异：FC 是 fc_tool_call，
    ReAct 是 react_action。用同一个判据会让 ReAct 臂假显示为 0。
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from intent import is_tight_call

D = os.path.join(os.path.dirname(__file__), "..", "runs", "final")
CALL_MODE = {"fc": "fc_tool_call", "react": "react_action"}

def funnel(fname, proto):
    R = [json.loads(l) for l in open(os.path.join(D, fname), encoding="utf-8") if l.strip()]
    first = [(r["turns"] or [{}])[0] for r in R]
    mode = CALL_MODE[proto]
    parsed = sum(1 for t in first if t.get("parse_mode") == mode)
    tight = None if parsed > 0 else sum(1 for t in first if is_tight_call(t.get("raw_output") or ""))
    return dict(
        n=len(R),
        tight=tight,                                                     # None = 不可测
        parsed=parsed,
        executed=sum(1 for t in first if t.get("parse_mode") == mode and t.get("total")),
        obs=sum(1 for r in R if len(r["turns"]) >= 2),
        rescue=sum(1 for r in R if r["final_ok"] and not r["first_ok"]),
        final=sum(r["final_ok"] for r in R),
        fallback=sum(1 for t in first
                     if t.get("parse_mode") in ("fc_direct_fenced_code", "direct_fenced_code")
                     and t.get("total")))

ARMS = [
    ("traj_v5_Qwen32B_fc_intent.jsonl", "fc",    "Qwen-32B · FC + hermes\n(documented default)"),
    ("traj_v5_Qwen7B_fc_intent.jsonl",  "fc",    "Qwen-7B · FC + hermes\n(documented default)"),
    ("traj_v6b_Qwen7B_fc_plugin.jsonl", "fc",    "Qwen-7B · FC + dedicated adapter\n(repaired)"),
    ("traj_v3_Qwen7B_react.jsonl",      "react", "Qwen-7B · ReAct\n(reference)"),
]

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    data = {lab: funnel(f, p) for f, p, lab in ARMS}
    json.dump({k: v for k, v in data.items()},
              open(os.path.join(os.path.dirname(__file__), "funnel_data.json"), "w"),
              indent=1, ensure_ascii=False)

    layers = ["emitted", "parsed", "executed", "Observation", "rescued"]
    keys = ["tight", "parsed", "executed", "obs", "rescue"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.4), sharey=True)
    for ax, (f, p, lab) in zip(axes, ARMS):
        d = data[lab]
        vals = [d[k] for k in keys]
        col = ["#c0392b" if lab.startswith("Qwen-32B") or "hermes" in lab else "#27ae60"] * 5
        shown = [0 if v is None else v for v in vals]
        bars = ax.bar(range(5), shown, color=col, width=0.62)
        if vals[0] is None:
            bars[0].set_color("#bdc3c7"); bars[0].set_hatch("//")
        for i, v in enumerate(vals):
            ax.text(i, (0 if v is None else v) + 2.5,
                    "N/A†" if v is None else str(v), ha="center", fontsize=8.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels(layers, fontsize=8, rotation=32, ha="right")
        ax.set_ylim(0, 118); ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.text(4.35, 112, f"final pass {d['final']}", fontsize=8, color="#555", ha="right")
    axes[0].set_ylabel("items out of 100")
    fig.suptitle("Where the agent trajectory is censored: emitted → parsed → executed → observed → rescued"
                 "   (n=100 per arm)", fontsize=11)
    fig.text(0.005, 0.015,
             "† Emitted-call count is measurable only when the server parses nothing: a successful parse "
             "moves the call into tool_calls and leaves content empty.",
             fontsize=7, color="#555")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "fig0_funnel.png"))
    for lab, d in data.items():
        print(f"{lab.splitlines()[0]:<34} " + "  ".join(
            f"{k}={'N/A' if d[k] is None else d[k]}" for k in keys) + f"  final={d['final']}")
    print("\n已写 figures/fig0_funnel.png")

#!/usr/bin/env python3
"""TRAIN-01B 最终分析：ReAct 75 步 vs 三条历史 FC 曲线。

三件事：
  1. 救回数曲线并列（本次 ReAct 与历史 GRPO×2 / RLOO）
  2. 工具调用按时间戳拆分为「训练 rollout」与「评测」
  3. 逐题翻转分析 —— 总量可能不动而成分在变（§5.9 的第三处证据）
"""
import json, glob, os, re, sys
KD = {"HumanEval/32", "Mbpp/599"}
ROOT = os.path.join(os.path.dirname(__file__), "..")

def curve(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "step_*.jsonl"))):
        R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        m = [x for x in R if x["channel"] == "multi" and x["task_id"] not in KD]
        r = [x for x in R if x["channel"] == "repair" and x["task_id"] not in KD]
        if not m: continue
        t1 = sum(1 for x in m if x["turns"] and x["turns"][0]["all_passed"])
        fin = sum(1 for x in m if x["turns"] and x["turns"][-1]["all_passed"])
        out.append(dict(step=int(os.path.basename(f)[5:10]), n=len(m), t1=t1, fin=fin,
                        rescue=fin - t1,
                        repair=sum(1 for x in r if x["turns"] and x["turns"][-1]["all_passed"]),
                        nr=len(r),
                        solved=frozenset(x["task_id"] for x in m
                                         if x["turns"] and x["turns"][-1]["all_passed"])))
    return out

def split_tool_calls(counter, log):
    """按时间戳把工具调用拆成训练 rollout 与评测两部分。"""
    if not os.path.exists(counter): return None
    ts = []
    for line in open(counter):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            try: ts.append(float(p[1]))
            except ValueError: pass
    if not ts: return dict(total=sum(1 for _ in open(counter)), note="计数无时间戳，无法拆分")
    return dict(total=len(ts), first=min(ts), last=max(ts))

if __name__ == "__main__":
    print("=" * 78)
    print("TRAIN-01B  ReAct 75 步  vs  历史 FC 150 步")
    print("=" * 78)
    react = curve("/root/autodl-tmp/runs/code-eval-grpo-seed42")
    hist = {"GRPO seed1": "/root/autodl-tmp/runs/code-eval-grpo-seed1",
            "RLOO seed42": "/root/autodl-tmp/runs/code-eval-rloo-seed42",
            "GRPO (first)": "/root/autodl-tmp/runs/code-eval"}
    print(f"\n【本次 ReAct】{'step':>6}{'n':>6}{'turn1':>7}{'final':>7}{'rescue':>8}{'repair':>8}")
    for r in react:
        print(f"{'':>12}{r['step']:>6}{r['n']:>6}{r['t1']:>7}{r['fin']:>7}{r['rescue']:>8}{r['repair']:>8}")
    if react:
        v = [r["rescue"] for r in react if r["n"] == 540]
        if v:
            print(f"\n  救回数（仅 540 口径）: {v}   min={min(v)} max={max(v)} 末-首={v[-1]-v[0]:+d}")
    print("\n【历史 FC run 的救回数】")
    for lab, d in hist.items():
        c = curve(d)
        v = [x["rescue"] for x in c if x["n"] == 540]
        if v: print(f"  {lab:<14} {v}   末-首={v[-1]-v[0]:+d}")
    print("\n【成分变动：相邻 checkpoint 的通过集合】")
    for a, b in zip(react, react[1:]):
        if a["n"] != b["n"]: continue
        inter = len(a["solved"] & b["solved"]); union = len(a["solved"] | b["solved"])
        gained = len(b["solved"] - a["solved"]); lost = len(a["solved"] - b["solved"])
        print(f"  step {a['step']:>3}→{b['step']:<3}  Jaccard {inter/union:.3f}   "
              f"新增 {gained:>3}  丢失 {lost:>3}  净 {gained-lost:+d}")
    print("\n【工具调用拆分】")
    s = split_tool_calls("/root/autodl-tmp/runs/.tool_call_count",
                         "/root/autodl-tmp/code-agent/train01b.log")
    print(f"  {s}")

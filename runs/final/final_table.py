#!/usr/bin/env python3
"""主表与 A9 方差表。含请求错误的臂一律 N/A，不计算通过率差与 p 值。

规则（此前版本只打印警告仍照算，属于分析纪律缺陷）：
  n_err > 0  →  该臂存在缺失数据 → 通过率不可比 → 拒绝计算，标 N/A
                （其错误率普查仍然有效，另表报告）
"""
import json, os, statistics as st
from math import comb

def load(f):
    if not os.path.exists(f): return None
    return {r["clean_index"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}

def n_err(D):
    return sum(1 for r in D.values() for t in r["turns"]
               if (t.get("raw_output") or "").startswith("__ERROR__"))

def mcnemar(A, B):
    K = sorted(set(A) & set(B))
    b = sum(1 for k in K if A[k]["final_ok"] and not B[k]["final_ok"])
    c = sum(1 for k in K if B[k]["final_ok"] and not A[k]["final_ok"])
    n = b + c
    p = min(sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n * 2, 1.0) if n else 1.0
    return b, c, p, len(K)

PAIRS = [
    ("Llama-3.1-8B",     "traj_v14_Llama8B_react.jsonl",   "traj_v6_Llama8B_fc_strict.jsonl",   "官方模板+strict"),
    ("Qwen2.5-Coder-7B", "traj_v14_Qwen7B_react.jsonl",    "traj_v6b_Qwen7B_fc_plugin.jsonl",   "专用适配器"),
    ("Mistral-7B-v0.3",  "traj_v14_Mistral7B_react.jsonl", "traj_v9_Mistral7B_fc_strict.jsonl", "parser+strict"),
]

print("=" * 100)
print("表 1  主配对（全部真实 max_tokens=2048、同一 100 题、temperature=0）")
print(f"{'模型':<20}{'ReAct首轮':>10}{'ReAct最终':>10}{'FC首轮':>9}{'FC最终':>9}{'b/c':>9}{'p':>10}   FC 臂")
print("-" * 100)
for name, rf, ff, flab in PAIRS:
    R, F = load(rf), load(ff)
    if not R or not F:
        print(f"{name:<20}  缺文件"); continue
    K = sorted(set(R) & set(F))
    ne = n_err(F)
    r1 = sum(R[k]["first_ok"] for k in K); rr = sum(R[k]["final_ok"] for k in K)
    if ne > 0:
        print(f"{name:<20}{r1:>10}{rr:>10}{'N/A':>9}{'N/A':>9}{'N/A':>9}{'N/A':>10}   "
              f"{flab}（该臂 n_err={ne}，缺失数据，通过率不可比）")
        continue
    b, c, p, n = mcnemar(R, F)
    print(f"{name:<20}{r1:>10}{rr:>10}"
          f"{sum(F[k]['first_ok'] for k in K):>9}{sum(F[k]['final_ok'] for k in K):>9}"
          f"{f'{b}/{c}':>9}{p:>10.4f}   {flab}")
print("-" * 100)
print("Mistral 的 FC 臂全部含请求错误（v3 n_err=2、v5 官方 42、v9 strict 3、v9 官方+strict 39），")
print("故 Mistral 无可用于通过率比较的 FC 臂；其错误率本身见表 3。")

print("\n" + "=" * 100)
print("表 2  A9 方差（Llama-3.1-8B，temperature=0.6，n=100×3 seed）")
print(f"{'臂':<10}{'seed':>5}{'首轮':>7}{'最终':>7}{'n_err':>7}  有效性")
print("-" * 100)
clean = {"ReAct": [], "FC": []}
for lab, pat in [("ReAct", "traj_v13_Llama8B_react_t06_s%d.jsonl"),
                 ("FC", "traj_v13_Llama8B_fcstrict_t06_s%d.jsonl")]:
    for sd in (1, 2, 3):
        D = load(pat % sd)
        if not D: continue
        ne = n_err(D)
        ok = sum(r["final_ok"] for r in D.values())
        f1 = sum(r["first_ok"] for r in D.values())
        valid = ne == 0
        if valid: clean[lab].append(ok)
        print(f"{lab:<10}{sd:>5}{f1:>7}{ok:>7}{ne:>7}  {'有效' if valid else '**无效**（并行工具调用被 vLLM 拒收）'}")
print("-" * 100)
for lab in ("ReAct", "FC"):
    v = clean[lab]
    # 少于 3 个有效臂不报标准差：2 个样本的 sd 无意义，且与正文口径冲突
    s = f"标准差={st.stdev(v):.2f}" if len(v) >= 3 else "（有效臂 <3，不报标准差）"
    print(f"  {lab:<8} 有效臂 {len(v)}/3  取值 {v}  均值={st.mean(v):.1f}  {s}")
if clean["ReAct"] and clean["FC"]:
    print(f"\n  仅用有效臂的最坏情况：ReAct 最低 {min(clean['ReAct'])} vs FC 最高 {max(clean['FC'])} "
          f"= {min(clean['ReAct']) - max(clean['FC']):+d} 个点")
    print("  ※ FC 仅 2 个有效臂，不足以给出可信的标准差；此处只报区间与最坏情况。")

print("\n" + "=" * 100)
print("表 3  接口层错误率（这些臂的错误率有效，通过率无效）")
print(f"{'臂':<46}{'n':>5}{'n_err':>7}")
print("-" * 100)
for f in ["traj_v3_Mistral7B_fc.jsonl", "traj_v5_Mistral7B_fc_official.jsonl",
          "traj_v9_Mistral7B_fc_strict.jsonl", "traj_v9_Mistral7B_fc_official_strict.jsonl",
          "traj_v13_Llama8B_fcstrict_t06_s3.jsonl"]:
    D = load(f)
    if D: print(f"{f:<46}{len(D):>5}{n_err(D):>7}")

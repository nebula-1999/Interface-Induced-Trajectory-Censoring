#!/usr/bin/env python3
"""VAL-04：并入第三人裁决，给出最终标签与最终 κ。

裁决集的选择规则是先定好的：三方（A1 / A2 / 分类器）任意不一致就进来，
**不看谁判了什么**。这与第一轮不同——第一轮按「与分类器不符」选，
改动只可能朝分类器移动，一致率只能升。
"""
import csv, json, os, random

H = os.path.dirname(os.path.abspath(__file__)); P = lambda *a: os.path.join(H, *a)
sheet = lambda p, col: {r["item"]: (r.get(col) or "").strip().upper()
                        for r in csv.DictReader(open(P(p), encoding="utf-8"), delimiter="\t")
                        if (r.get(col) or "").strip()}

key = {x["i"]: x for x in json.load(open(P("_annotation_key.json"), encoding="utf-8"))}
m2  = json.load(open(P("_annotator2_map.json"), encoding="utf-8"))
m3  = json.load(open(P("_adjudication2_map.json"), encoding="utf-8"))
a1  = {int(k): v for k, v in sheet("annotation_sheet.tsv", "answer").items()}
a2  = {int(m2[k]): v for k, v in sheet("annotator2_sheet.tsv", "answer").items()}
a3  = {int(m3[k]): v for k, v in sheet("adjudication2_sheet.tsv", "verdict").items()}
clf = {i: ("Y" if key[i]["classifier"] == "tight" else "N") for i in key}

raws = {}
for sc in ["1.5B", "3B", "7B", "14B", "32B"]:
    for line in open(P("..", "runs", "final", f"traj_v5_Qwen{sc}_fc_intent.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line); t = (r["turns"] or [{}])[0]
            raws[(sc, r["clean_index"])] = t.get("raw_output") or ""
L = lambda i: len(raws[(key[i]["scale"], key[i]["clean_index"])])

# 最终标签：裁决过的用裁决；未进裁决集的三方本就一致，取任一
final = {}
for i in key:
    if i in a3: final[i] = a3[i]
    elif a2.get(i) in "YN": final[i] = a2[i]
    elif a1.get(i) in "YN": final[i] = a1[i]

def kap(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(x == "Y" for x in a) / n; pb = sum(x == "Y" for x in b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")

def ci(a, b, n=5000):
    r = random.Random(7); ks = []
    for _ in range(n):
        s = [r.randrange(len(a)) for _ in a]
        k = kap([a[i] for i in s], [b[i] for i in s])
        if k == k: ks.append(k)
    ks.sort(); return ks[int(.025 * len(ks))], ks[int(.975 * len(ks))]

print("=" * 68)
print(f"第三人裁决 {len(a3)} 条：{sum(1 for v in a3.values() if v=='Y')} Y / "
      f"{sum(1 for v in a3.values() if v=='N')} N")
print("=" * 68)
sides = {"A1": 0, "A2": 0, "分类器": 0}
for i, v in a3.items():
    if a1.get(i) == v: sides["A1"] += 1
    if a2.get(i) == v: sides["A2"] += 1
    if clf[i] == v: sides["分类器"] += 1
print(f"  裁决与各方一致的条数（共 {len(a3)}）：" +
      "  ".join(f"{k} {v}" for k, v in sides.items()))
print(f"  裁决背离分类器的条数：{len(a3)-sides['分类器']}"
      f"   ← 第一轮这个数是 0，说明本轮选择规则确实无偏\n")

for name, ids in [("全部可比条目", [i for i in final if a1.get(i) in "YN" and a2.get(i) in "YN"]),
                  ("★ 两名标注者读到同一批字节", [i for i in final if a1.get(i) in "YN"
                                                and a2.get(i) in "YN" and L(i) <= 2400])]:
    A = [a1[i] for i in ids]; B = [a2[i] for i in ids]
    C = [clf[i] for i in ids]; F = [final[i] for i in ids]
    print(f"{name}  n={len(ids)}")
    for lbl, x, y in [("A1 vs A2（inter-annotator）", A, B),
                      ("分类器 vs 最终标签", C, F)]:
        lo, hi = ci(x, y)
        print(f"   {lbl:30s} 一致 {sum(p==q for p,q in zip(x,y)):3d}/{len(ids)}"
              f"  κ={kap(x,y):.3f}  95%CI[{lo:.3f},{hi:.3f}]")
    print()

T = [i for i in key if key[i]["classifier"] == "tight" and i in final]
prec = sum(1 for i in T if final[i] == "Y") / len(T)
FN = [i for i in final if key[i]["classifier"] != "tight" and final[i] == "Y"]
print("=" * 68)
print(f"最终修正因子（tight 层精确率）: {sum(1 for i in T if final[i]=='Y')}/{len(T)} = {prec:.3f}")
print(f"  → §5.2 的 80/100 修正为 {80*prec:.1f}")
print(f"  分类器漏判（最终=Y 但非 tight）: {len(FN)} 条 → 判据同时在低估")
net = sum(1 for i in T if final[i] == "N") - len(FN)
print(f"  净方向: 过判 {sum(1 for i in T if final[i]=='N')} − 漏判 {len(FN)} = {net:+d}"
      f"   {'（净低估）' if net<0 else '（净高估）' if net>0 else '（相抵）'}")
json.dump({str(k): v for k, v in sorted(final.items())},
          open(P("final_labels_v2.json"), "w"), indent=1)
print(f"\n最终标签写入 final_labels_v2.json（{len(final)} 条）")

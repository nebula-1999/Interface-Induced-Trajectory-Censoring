#!/usr/bin/env python3
"""VAL-03 评分：两名标注者之间的 inter-annotator reliability。

与 score_annotation.py 的区别：那个算的是「人 vs 分类器」，
本脚本算的是「人 vs 人」——审稿意见指出前者不能当作 reliability 报告。

同时对第一轮裁决程序做了一个审计（见 audit_adjudication）：
裁决集是按「与分类器不一致」选出来的，因此裁决只可能把一致率往上推，
κ 从 0.713 升到 0.936 是选择方式决定的，不是独立的可靠性提升。

用法: python validation/score_annotator2.py
"""
import csv, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)


def kappa(a, b):
    n = len(a)
    if not n: return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(x == "Y" for x in a) / n
    pb = sum(x == "Y" for x in b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def boot_ci(a, b, n=5000, seed=7):
    rng = random.Random(seed)
    idx = range(len(a))
    ks = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        k = kappa([a[i] for i in s], [b[i] for i in s])
        if k == k: ks.append(k)
    ks.sort()
    return ks[int(.025 * len(ks))], ks[int(.975 * len(ks))]


def load_sheet(path, mapping=None):
    out = {}
    if not os.path.exists(path): return out
    for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        v = (r.get("answer") or "").strip().upper()
        if v:
            k = r["item"]
            out[mapping[k] if mapping else int(k)] = v
    return out


def audit_adjudication():
    """第一轮裁决程序的审计——这段结论要写进论文，与 A2 无关。"""
    key = {x["i"]: x for x in json.load(open(P("_annotation_key.json"), encoding="utf-8"))}
    r1 = load_sheet(P("annotation_sheet.tsv"))
    fin = {int(k): v for k, v in json.load(open(P("final_labels.json"), encoding="utf-8")).items()}
    clf = {i: ("Y" if key[i]["classifier"] == "tight" else "N") for i in key}
    adj = [r for r in csv.DictReader(open(P("adjudication_sheet.tsv"), encoding="utf-8"), delimiter="\t")]
    aset = {int(r["item"]) for r in adj}
    dis = {i for i in r1 if r1[i] != clf[i]}
    flips_to = sum(1 for r in adj if r["round1"] != r["final"] and r["final"] == clf[int(r["item"])])
    flips_away = sum(1 for r in adj if r["round1"] != r["final"] and r["final"] != clf[int(r["item"])])
    print("=" * 66)
    print("第一轮裁决程序审计")
    print("=" * 66)
    print(f"  第一轮与分类器不一致        {len(dis)} 条")
    print(f"  送去裁决                    {len(aset)} 条")
    print(f"  两者重合                    {len(aset & dis)} / {len(aset)}"
          f"   {'← 裁决集完全等于分歧集' if aset == dis else ''}")
    print(f"  裁决后朝分类器移动          {flips_to}")
    print(f"  裁决后背离分类器            {flips_away}")
    print()
    print(f"  → 只复核分歧、且分歧按「与分类器不符」定义时，一致率只能升不能降。")
    print(f"    κ 0.713 → 0.936 由此产生，**不是独立的可靠性提升**。")
    print(f"    可报告的无偏系数是第一轮的 κ = {kappa([r1[i] for i in sorted(r1)], [clf[i] for i in sorted(r1)]):.3f}。")
    print()


def main():
    audit_adjudication()
    mp = P("_annotator2_map.json")
    if not os.path.exists(mp):
        print("缺 _annotator2_map.json，先运行 make_annotator2_pack.py"); return 1
    mapping = json.load(open(mp, encoding="utf-8"))
    a2 = load_sheet(P("annotator2_sheet.tsv"), mapping)
    if not a2:
        print("=" * 66)
        print("annotator2_sheet.tsv 的 answer 列还是空的——第二名标注者尚未标注。")
        print("完成后重新运行本脚本，即可得到 inter-annotator κ。")
        print("=" * 66)
        return 0

    key = {x["i"]: x for x in json.load(open(P("_annotation_key.json"), encoding="utf-8"))}
    a1 = load_sheet(P("annotation_sheet.tsv"))
    clf = {i: ("Y" if key[i]["classifier"] == "tight" else "N") for i in key}

    ids = sorted(i for i in a2 if a2[i] in "YN" and a1.get(i) in ("Y", "N"))
    A1 = [a1[i] for i in ids]; A2 = [a2[i] for i in ids]; C = [clf[i] for i in ids]
    unsure = [i for i in a2 if a2[i] == "?"]

    print("=" * 66)
    print(f"inter-annotator reliability   n={len(ids)}   （A2 拿不准 {len(unsure)} 条，不计入）")
    print("=" * 66)
    for nm, x, y in [("A1(第一轮) vs A2   ← 这是要报告的那个", A1, A2),
                     ("A2 vs 分类器", A2, C),
                     ("A1(第一轮) vs 分类器", A1, C)]:
        lo, hi = boot_ci(x, y)
        print(f"  {nm:34s} 一致 {sum(p==q for p,q in zip(x,y)):3d}/{len(x)}"
              f"  κ={kappa(x,y):.3f}  95%CI[{lo:.3f},{hi:.3f}]")

    tri = [i for i in ids if len({a1[i], a2[i], clf[i]}) > 1]
    print(f"\n  三方不完全一致: {len(tri)} 条 → {tri}")
    print(f"  其中两名标注者不一致: {[i for i in ids if a1[i]!=a2[i]]}")
    print()
    print("  裁决规则（先定后做，避免重蹈第一轮覆辙）：")
    print("    复核集 = 两名标注者不一致的全部条目，**不看分类器判断**选出；")
    print("    由第三人裁决；裁决后重算 κ 时必须同时报告第一轮的无偏 κ。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

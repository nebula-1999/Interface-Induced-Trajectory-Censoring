#!/usr/bin/env python3
"""VAL-01 评分：把人工标注与 analysis/intent.py 的 tight 判据对照。

输出：一致率、Cohen's κ、按类的精确率/召回率、以及**错分方向**
（分类器多判了什么 / 漏判了什么），后者比总体一致率更有诊断价值。

用法: python validation/score_annotation.py
"""
import csv, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

def load():
    key = {x["i"]: x for x in json.load(open(os.path.join(HERE, "_annotation_key.json"), encoding="utf-8"))}
    ann = {}
    with open(os.path.join(HERE, "annotation_sheet.tsv"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            v = (row.get("answer") or "").strip().upper()
            if v: ann[int(row["item"])] = v
    return key, ann

def kappa(a, b):
    """Cohen's κ，二分类。"""
    n = len(a)
    if n == 0: return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")

def main():
    key, ann = load()
    if not ann:
        print("annotation_sheet.tsv 的 answer 列还是空的。"); return 1
    unsure = [i for i, v in ann.items() if v == "?"]
    pairs = [(i, v) for i, v in ann.items() if v in ("Y", "N")]
    print(f"已标注 {len(ann)}/{len(key)}   其中拿不准 {len(unsure)} 条（不计入一致率）\n")
    if not pairs: print("没有可用的 Y/N 标注。"); return 1

    human = [1 if v == "Y" else 0 for _, v in pairs]
    clf   = [1 if key[i]["classifier"] == "tight" else 0 for i, _ in pairs]

    tp = sum(1 for h, c in zip(human, clf) if h and c)
    fp = sum(1 for h, c in zip(human, clf) if not h and c)
    fn = sum(1 for h, c in zip(human, clf) if h and not c)
    tn = sum(1 for h, c in zip(human, clf) if not h and not c)
    n = len(pairs)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec  = tp / (tp + fn) if tp + fn else float("nan")

    print("混淆矩阵（行=分类器 tight，列=人工 Y）")
    print(f"              人工 Y   人工 N")
    print(f"  分类器 tight   {tp:>5}   {fp:>5}")
    print(f"  分类器 其他    {fn:>5}   {tn:>5}")
    print(f"\n  一致率      {(tp+tn)/n:.1%}   ({tp+tn}/{n})")
    print(f"  Cohen's κ   {kappa(human, clf):.3f}")
    print(f"  精确率      {prec:.1%}   （分类器判 tight 中人工也认可的比例）")
    print(f"  召回率      {rec:.1%}   （人工认可中分类器抓到的比例）")

    print("\n错分方向（比总体一致率更有诊断价值）")
    if fp: print(f"  假阳性 {fp} 条：分类器判 tight、人工判 N  →  报告的 tight 计数偏高")
    if fn: print(f"  假阴性 {fn} 条：人工判 Y、分类器未判 tight  →  报告的 tight 计数偏低")
    if not fp and not fn: print("  无错分")

    by = Counter()
    for i, v in pairs:
        by[(key[i]["classifier"], v)] += 1
    print("\n按分类器分层的分歧位置")
    for k in ("tight", "strong_not_tight", "neither"):
        y, nn = by[(k, "Y")], by[(k, "N")]
        if y + nn: print(f"  分类器={k:<18} 人工 Y {y:>3}  人工 N {nn:>3}")

    print("\n对论文的含义")
    if not (fp or fn):
        print("  tight 计数可按原值报告。")
    else:
        net = fn - fp
        print(f"  净偏差 {net:+d} / {n} 条抽样 → 报告 tight 数时应附")
        print(f"  '人工验证 n={n}，精确率 {prec:.0%}、召回率 {rec:.0%}' 而非裸数字。")
    if unsure:
        print(f"  另有 {len(unsure)} 条人工拿不准（题号 {unsure[:8]}{'…' if len(unsure)>8 else ''}），")
        print("  这些是判据本身的灰区，值得在 Limitations 里点名。")
    return 0

if __name__ == "__main__":
    sys.exit(main())

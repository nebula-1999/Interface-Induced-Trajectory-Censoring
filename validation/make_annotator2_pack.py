#!/usr/bin/env python3
"""VAL-03：生成第二名标注者的盲标包。

目的与第一包不同：第一包测的是「人 vs 分类器」，本包测的是
**inter-annotator reliability**——两个互不通气的人对同一批 98 条的一致程度。
这是审稿意见里唯一零算力就能补上的控制。

盲标设计：
  · 同一批 98 条（必须同批，否则算不出 inter-annotator κ）
  · **重新打乱**，编号与第一包无关——A2 看到的 #37 不是 A1 的 #37，
    两人即便交谈也无法对齐条目
  · 不含分类器判断，不含 A1 的任何标注
  · 判定细则与第一轮**逐字相同**（协议不同就不能比一致率）
  · A2 只看原始输出，对应 A1 的**第一轮**协议，而不是加了上下文的裁决轮

映射写在 _annotator2_map.json，标注完成前不要打开。
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260901
MAXCHARS = 4000          # 第一包截在 2400，导致判据落在 2585 的条目被截掉；这里放宽

key = json.load(open(os.path.join(HERE, "_annotation_key.json"), encoding="utf-8"))

# 从第一包的 markdown 里取回原文，保证两名标注者看到的字节完全一致
pack = open(os.path.join(HERE, "annotation_pack.md"), encoding="utf-8").read()
blocks = dict(re.findall(r'\n## (\d+)\n\n```\n(.*?)\n```\n', pack, re.S))
missing = [k["i"] for k in key if str(k["i"]) not in blocks]
if missing:
    sys.exit(f"第一包里取不到这些条目的原文: {missing[:10]}")

items = [{"a1_item": k["i"], "raw": blocks[str(k["i"])]} for k in key]
rng = random.Random(SEED)
rng.shuffle(items)

RUBRIC = open(os.path.join(HERE, "annotation_pack.md"), encoding="utf-8").read()
RUBRIC = RUBRIC[RUBRIC.index("## 你要回答的唯一问题"):RUBRIC.index("\n## 1\n")]

with open(os.path.join(HERE, "annotator2_pack.md"), "w", encoding="utf-8") as f:
    f.write(f"""# 标注包 · 第二名标注者（{len(items)} 条）

**在 `annotator2_sheet.tsv` 的 `answer` 列填 `Y` / `N` / `?`，不要改这个文件。**

> 请**不要**在标注完成前查看 `annotation_pack.md`、`annotation_sheet.tsv`、
> `final_labels.json`、`_annotation_key.json` 或 `_annotator2_map.json`。
> 本包的价值完全来自你与第一名标注者互不知情——看过任何一个都会让结果作废。
>
> 判定细则与第一名标注者所用的**逐字相同**，请严格按细则判，
> 不要凭「这条看起来像陷阱」调整尺度。拿不准就填 `?`，那比猜一个更有用。

---

{RUBRIC}
---

""")
    for i, s in enumerate(items, 1):
        raw = s["raw"]
        trunc = f"\n\n…（超出 {MAXCHARS} 字符已截断）" if len(raw) > MAXCHARS else ""
        f.write(f"## {i}\n\n```\n{raw[:MAXCHARS]}{trunc}\n```\n\n---\n\n")

with open(os.path.join(HERE, "annotator2_sheet.tsv"), "w", encoding="utf-8") as f:
    f.write("item\tanswer\tnote\n")
    for i in range(len(items)):
        f.write(f"{i+1}\t\t\n")

with open(os.path.join(HERE, "_annotator2_map.json"), "w", encoding="utf-8") as f:
    json.dump({str(i): s["a1_item"] for i, s in enumerate(items, 1)}, f, indent=1)

print(f"第二标注包已生成（种子 {SEED}）：")
print(f"  annotator2_pack.md    {len(items)} 条，重新打乱，不含任何既有标注")
print(f"  annotator2_sheet.tsv  填这个")
print(f"  _annotator2_map.json  编号映射（标注完成前不要看）")
print(f"\n与第一包的差异：")
print(f"  · 顺序完全不同——两个包的 #{1} 分别是原第 {key[0]['i']} 条与第 {items[0]['a1_item']} 条")
print(f"  · 截断放宽到 {MAXCHARS} 字符（第一包 2400，曾截掉位于 2585 的判据）")

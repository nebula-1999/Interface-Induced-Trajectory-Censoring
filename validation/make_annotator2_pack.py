#!/usr/bin/env python3
"""VAL-03：生成第二名标注者的盲标包。

目的与第一包不同：第一包测的是「人 vs 分类器」，本包测的是
**inter-annotator reliability**——两个互不通气的人对同一批 98 条的一致程度。

原文一律取自**原始轨迹文件**，不从第一包的 markdown 里回捞。
第一版这么做过，踩了两个坑：
  1. 第一包已在 2400 字符处截断，"放宽上限"是空操作；
  2. 模型输出本身就含 ```python 围栏，用 ``` 包裹再用正则回捞，
     会在第一个内部围栏处停下——98 条里 71 条被截在任意位置。
因此本包：原始 JSONL 取全文、用 ~~~~ 围栏（模型输出里不会出现）。

盲标设计：
  · 同一批 98 条（必须同批，否则算不出 inter-annotator κ）
  · **重新打乱**，编号与第一包无关
  · 不含分类器判断，不含 A1 的任何标注
  · 判定细则与第一轮**逐字相同**
  · A2 只看原始输出，对应 A1 的第一轮协议
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SEED = 20260901
SCALES = ["1.5B", "3B", "7B", "14B", "32B"]
FENCE = "~" * 8          # 模型输出里出现连续 8 个 ~ 的概率可忽略；仍会校验

key = json.load(open(os.path.join(HERE, "_annotation_key.json"), encoding="utf-8"))

raws = {}
for sc in SCALES:
    f = os.path.join(ROOT, "runs", "final", f"traj_v5_Qwen{sc}_fc_intent.jsonl")
    if not os.path.exists(f):
        sys.exit(f"缺原始轨迹 {f}")
    for line in open(f, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        t = (r["turns"] or [{}])[0]
        raws[(sc, r["clean_index"])] = t.get("raw_output") or ""

missing = [k["i"] for k in key if (k["scale"], k["clean_index"]) not in raws]
if missing: sys.exit(f"原始轨迹里找不到条目 {missing[:10]}")

items = [{"a1": k["i"], "raw": raws[(k["scale"], k["clean_index"])]} for k in key]
bad = [s["a1"] for s in items if FENCE in s["raw"]]
if bad: sys.exit(f"围栏与正文冲突，条目 {bad}——请加长 FENCE")

random.Random(SEED).shuffle(items)

pack1 = open(os.path.join(HERE, "annotation_pack.md"), encoding="utf-8").read()
RUBRIC = pack1[pack1.index("## 你要回答的唯一问题"):pack1.index("\n## 1\n")]

with open(os.path.join(HERE, "annotator2_pack.md"), "w", encoding="utf-8") as f:
    f.write(f"""# 标注包 · 第二名标注者（{len(items)} 条）

**在 `annotator2_sheet.tsv` 的 `answer` 列填 `Y` / `N` / `?`，不要改这个文件。**

> 请**不要**在标注完成前查看 `annotation_pack.md`、`annotation_sheet.tsv`、
> `final_labels.json`、`_annotation_key.json` 或 `_annotator2_map.json`。
> 本包的价值完全来自你与第一名标注者互不知情——看过任何一个都会让结果作废。
>
> 判定细则与第一名标注者所用的**逐字相同**，请严格按细则判，
> 不要凭「这条看起来像陷阱」调整尺度。拿不准就填 `?`，那比猜一个更有用。
>
> **每条都是完整原文，没有任何截断。** 模型输出本身常含 ```python 代码块，
> 因此本文件用 `{FENCE}` 作为外层围栏——里面出现的 ``` 是原文的一部分。
> 判据有时出现在代码块**之后**，请读完整条再判。

---

{RUBRIC}
---

""")
    for i, s in enumerate(items, 1):
        f.write(f"## {i}\n\n{FENCE}\n{s['raw']}\n{FENCE}\n\n---\n\n")

with open(os.path.join(HERE, "annotator2_sheet.tsv"), "w", encoding="utf-8") as f:
    f.write("item\tanswer\tnote\n")
    for i in range(len(items)): f.write(f"{i+1}\t\t\n")

with open(os.path.join(HERE, "_annotator2_map.json"), "w", encoding="utf-8") as f:
    json.dump({str(i): s["a1"] for i, s in enumerate(items, 1)}, f, indent=1)

L = sorted(len(s["raw"]) for s in items)
print(f"第二标注包已生成（种子 {SEED}）：")
print(f"  annotator2_pack.md    {len(items)} 条全文，无截断")
print(f"  annotator2_sheet.tsv  填这个")
print(f"  _annotator2_map.json  编号映射（标注完成前不要看）")
print(f"\n原文长度：中位 {L[len(L)//2]}，最长 {L[-1]}，超 2400 的 {sum(1 for x in L if x>2400)} 条")
print(f"注：raw_output 在探针存盘时上限 4000 字符（{sum(1 for x in L if x==4000)} 条触顶），")
print(f"    分类器读到的也是同样这些字节，故一致率的输入两边完全相同。")

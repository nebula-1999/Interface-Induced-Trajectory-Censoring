#!/usr/bin/env python3
"""VAL-01：生成人工标注包，用于检验 analysis/intent.py 的 tight 判据。

设计要点：
  · **盲标**——标注包不含分类器判断，只有原文
  · **按分类器判断分层抽样**——40 条判为 tight、30 条 strong 但非 tight、
    30 条两者皆非。这样能同时估计假阳性与假阴性，而不是只测容易的样本。
    （代价：本包不能用来估计总体比例，只用于估计各类的精确率/召回率）
  · **固定随机种子**，顺序打乱，标注者无法从位置推断
"""
import json, os, random, sys, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from intent import is_tight_call, STRONG_TAGS

D = os.path.join(os.path.dirname(__file__), "..", "runs", "final")
SCALES = ["1.5B", "3B", "7B", "14B", "32B"]
SEED = 20260831
MAXCHARS = 2400

pool = {"tight": [], "strong_not_tight": [], "neither": []}
for sc in SCALES:
    f = os.path.join(D, f"traj_v5_Qwen{sc}_fc_intent.jsonl")
    if not os.path.exists(f): continue
    for line in open(f, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        t = (r["turns"] or [{}])[0]
        raw = t.get("raw_output") or ""
        if not raw.strip(): continue
        tight = is_tight_call(raw)
        strong = t.get("_fc_intent") in STRONG_TAGS
        key = "tight" if tight else ("strong_not_tight" if strong else "neither")
        pool[key].append({"scale": sc, "clean_index": r["clean_index"], "raw": raw, "_gold_hint": key})

rng = random.Random(SEED)
want = {"tight": 40, "strong_not_tight": 30, "neither": 30}
sample = []
for k, n in want.items():
    have = pool[k]
    rng.shuffle(have)
    take = have[:n]
    if len(take) < n:
        print(f"  ⚠️ {k} 仅有 {len(have)} 条，取全部")
    sample += take
rng.shuffle(sample)

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
key_path = os.path.join(os.path.dirname(__file__), "_annotation_key.json")
pack_path = os.path.join(os.path.dirname(__file__), "annotation_pack.md")
sheet_path = os.path.join(os.path.dirname(__file__), "annotation_sheet.tsv")

with open(key_path, "w", encoding="utf-8") as f:
    json.dump([{"i": i + 1, "scale": s["scale"], "clean_index": s["clean_index"],
                "classifier": s["_gold_hint"]} for i, s in enumerate(sample)], f,
              ensure_ascii=False, indent=1)

with open(sheet_path, "w", encoding="utf-8") as f:
    f.write("item\tanswer\tnote\n")
    for i in range(len(sample)):
        f.write(f"{i+1}\t\t\n")

with open(pack_path, "w", encoding="utf-8") as f:
    header = """# 标注包：工具调用判据人工验证（NITEMS 条）

**在 `annotation_sheet.tsv` 的 `answer` 列填 `Y` / `N` / `?`，不要改这个文件。**

---

## 你要回答的唯一问题

> 这段模型输出里，**是否存在一个格式完好的 `run_tests` 调用**——
> 即一个 JSON 对象，`name` 为 `run_tests`，其 `arguments` 里的 `code`
> 是一个**字符串字面量**且内容是**真实的 Python 代码**？

- `Y` — 是
- `N` — 否
- `?` — 拿不准（会单独统计，不计入一致率）

## 三条判定细则

**1. 必须是"调用"，不是"复述工具定义"**

模型有时会把我们注入的 schema 原样抄一遍。那是复述，不是调用：

```
不算：{"name": "run_tests", "description": "...", "parameters": {"type": "object", ...}}
      有 parameters/description、无 arguments  →  判 N

算：  {"name": "run_tests", "arguments": {"code": "def add(a,b):\\n    return a+b"}}
      →  判 Y
```

**2. `code` 必须是字符串字面量，不能是变量名或占位符**

```
不算：{"name": "run_tests", "arguments": {"code": test_code}}
      test_code 是标识符不是字符串  →  判 N

不算：{"name": "run_tests", "arguments": {"code": "<your code here>"}}
      占位符不是真实代码  →  判 N
```

**3. 不看标签，不看代码对不对**

- **不要求**包在 `<tool_call>` 或 `<tools>` 标签里 —— 有没有标签都不影响判定
- **不要求**代码正确或能通过测试。只要是真实 Python（有 `def` / `class` /
  `import` / 完整语句）即可。**写错的算法照样判 `Y`**

## 一条重要提醒

同一段输出里可能**既有** ```python 代码块**又有** JSON 调用。
只要 JSON 调用满足上面三条就判 `Y`，代码块的存在不影响判定。

---

"""
    f.write(header.replace("NITEMS", str(len(sample))))
    for i, s in enumerate(sample, 1):
        raw = s["raw"]
        trunc = "\n\n…（超出 {} 字符已截断）".format(MAXCHARS) if len(raw) > MAXCHARS else ""
        f.write(f"## {i}\n\n```\n{raw[:MAXCHARS]}{trunc}\n```\n\n---\n\n")

print(f"标注包已生成：")
print(f"  {pack_path}          {len(sample)} 条待标（盲标，不含分类器判断）")
print(f"  {sheet_path}       填这个文件的 answer 列")
print(f"  {key_path}   分类器判断（**标注完成前不要看**）")
print(f"\n抽样构成（按分类器判断分层，固定种子 {SEED}）：")
for k, n in want.items():
    print(f"  {k:<18} 目标 {n:>2}  实际 {sum(1 for s in sample if s['_gold_hint']==k):>2}  池中 {len(pool[k]):>3}")

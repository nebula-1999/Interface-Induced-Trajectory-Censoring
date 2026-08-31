#!/usr/bin/env python3
"""去污染结果的自审：哪两条评测题漏掉了，以及高分命中是真污染还是误报。"""

from __future__ import annotations

import json

from datasets import load_dataset

from decontaminate import N_CODE, N_TEXT, ngrams, norm_code, norm_text

he = load_dataset("evalplus/humanevalplus", split="test")
mb = load_dataset("evalplus/mbppplus", split="test")

print("=== 未进入索引的评测题（token 数不足一个 n-gram）===")
for r in he:
    t, c = norm_text(r["prompt"]), norm_code(r["prompt"] + "\n" + r["canonical_solution"])
    if len(ngrams(t, N_TEXT)) == 0 or len(ngrams(c, N_CODE)) == 0:
        print(f"  {r['task_id']}: text_tok={len(t)} code_tok={len(c)}")
for r in mb:
    t, c = norm_text(r["prompt"]), norm_code(r["code"])
    if len(ngrams(t, N_TEXT)) == 0 or len(ngrams(c, N_CODE)) == 0:
        print(f"  Mbpp/{r['task_id']}: text_tok={len(t)} code_tok={len(c)}")

print("\n=== 评测侧 n-gram 规模分布（分母太小会让 containment 虚高）===")
sizes = sorted(len(ngrams(norm_text(r["prompt"]), N_TEXT)) for r in list(he) + list(mb))
print(f"  文本 n-gram 数: min={sizes[0]} p10={sizes[len(sizes)//10]} "
      f"中位={sizes[len(sizes)//2]} max={sizes[-1]}")
print(f"  少于 10 个 n-gram 的评测题: {sum(1 for s in sizes if s < 10)} 条")

rep = json.load(open("contamination_report.json", encoding="utf-8"))
kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
mb_by_id = {f"Mbpp/{r['task_id']}": r for r in mb}
he_by_id = {r["task_id"]: r for r in he}

print("\n=== 抽检：解法通道最高的 3 条（代码相似才是硬证据）===")
for r in sorted(rep["rows"], key=lambda r: -r["code_c"])[:3]:
    hit = r["code_hit"]
    ev = mb_by_id.get(hit) or he_by_id.get(hit)
    print(f"\n--- KodCode #{r['i']} (subset={r['subset']}) "
          f"vs {hit}  code_c={r['code_c']:.3f} ---")
    print(f"[训练题] {kc[r['i']]['question'][:200]}")
    print(f"[训练解] {kc[r['i']]['solution'][:220]}")
    print(f"[评测解] {(ev.get('code') or ev.get('canonical_solution'))[:220]}")

print("\n=== 抽检：问题通道最高的 2 条（怀疑是短文本虚高）===")
for r in sorted(rep["rows"], key=lambda r: -r["text_c"])[:2]:
    hit = r["text_hit"]
    ev = mb_by_id.get(hit) or he_by_id.get(hit)
    q = kc[r["i"]]["question"]
    print(f"\n--- KodCode #{r['i']} vs {hit}  text_c={r['text_c']:.3f} "
          f"训练题 n-gram 数={len(ngrams(norm_text(q), N_TEXT))} ---")
    print(f"[训练题] {q[:200]}")
    print(f"[评测题] {ev['prompt'][:200]}")

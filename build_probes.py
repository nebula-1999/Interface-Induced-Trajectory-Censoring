#!/usr/bin/env python3
"""生成 scripted 修复探针：542 题各一份 buggy 初稿 + 真实的测试报错。

流程：注入 → 过沙箱 → **只保留确实让测试失败的那个变体**。改了但测试照过的
不算数（比如改到了不影响结果的分支）——那种题留着会把修复率算虚。

产出 probes_repair.jsonl，每行一条：
    task_id / source / mutation_kind / buggy_solution / error / n_tried

给模型看的是 buggy_solution + error，两个模型拿到的**逐字相同**，
这正是 Step 0 要求的可配对、无难度混淆的分母。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

from datasets import load_dataset

from mutate import candidates
from sandbox import build_evalplus, run_tests

n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None

recs: list[tuple[str, str, dict]] = []
for r in load_dataset("evalplus/humanevalplus", split="test"):
    recs.append(("HumanEval+", str(r["task_id"]), r))
for r in load_dataset("evalplus/mbppplus", split="test"):
    recs.append(("MBPP+", f"Mbpp/{r['task_id']}", r))
if n_arg:
    recs = recs[:n_arg // 2] + recs[-(n_arg - n_arg // 2):]

print(f"待生成 {len(recs)} 条", flush=True)
t0 = time.time()
rows, failed, kinds = [], [], Counter()

for n, (src, tid, rec) in enumerate(recs):
    sol, test = build_evalplus(rec)
    # 种子绑定 task_id：同一道题永远得到同一个 bug，换台机器也一样
    cands = candidates(sol, seed=abs(hash(tid)) % (2 ** 31))
    hit = None
    for i, (kind, buggy) in enumerate(cands):
        res = run_tests(buggy, test, mode="script", timeout=30.0)
        if not res.all_passed:
            hit = (kind, buggy, res.stderr, i + 1)
            break
    if hit:
        kind, buggy, err, tried = hit
        kinds[kind] += 1
        rows.append({"task_id": tid, "source": src, "mutation_kind": kind,
                     "buggy_solution": buggy, "error": err, "n_tried": tried})
    else:
        failed.append((src, tid, len(cands)))
    if (n + 1) % 100 == 0:
        print(f"  ...{n + 1}/{len(recs)}  成功 {len(rows)}", flush=True)

el = time.time() - t0
print(f"\n耗时 {el:.1f}s  单题 {el / len(recs):.2f}s")
print(f"成功注入 {len(rows)}/{len(recs)} = {len(rows) / len(recs):.1%}")
print(f"\nbug 类型分布（分析时按这个拆开修复率）:")
for k, c in kinds.most_common():
    print(f"  {k:20s} {c:4d}  {c / max(len(rows), 1):5.1%}")

if failed:
    print(f"\n注入失败 {len(failed)} 条（候选数 0 或改了测试照过）:")
    for s, t, nc in failed[:8]:
        print(f"  {s} {t}  候选变体数={nc}")

out = "probes_repair.jsonl" if not n_arg else f"probes_repair_{n_arg}.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n已写出 {out}")

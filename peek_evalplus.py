#!/usr/bin/env python3
"""看 EvalPlus 的 test 是什么格式——不能假设它和 KodCode 一样。

HumanEval+ 的 canonical_solution 是**函数体**（不含签名），要和 prompt 拼起来
才是完整函数；MBPP+ 的 code 是完整函数。这个差异会直接影响 bug 注入。
"""

from __future__ import annotations

from datasets import load_dataset

from sandbox import run_tests

he = load_dataset("evalplus/humanevalplus", split="test")
mb = load_dataset("evalplus/mbppplus", split="test")

r = he[0]
print("=" * 66, "\nHumanEval+ #0", r["task_id"], " entry_point =", r["entry_point"])
print(f"[prompt]\n{r['prompt']}")
print(f"[canonical_solution]\n{r['canonical_solution'][:300]}")
print(f"[test 前 500]\n{r['test'][:500]}")

r2 = mb[0]
print("=" * 66, "\nMBPP+ #0 task_id =", r2["task_id"])
print(f"[prompt]\n{r2['prompt'][:300]}")
print(f"[code]\n{r2['code'][:300]}")
print(f"[test 前 400]\n{r2['test'][:400]}")
print(f"[test_list] {str(r2.get('test_list'))[:200]}")

print("=" * 66, "\n用沙箱跑官方解（应当全通过，否则说明拼法不对）")
sol_he = r["prompt"] + r["canonical_solution"]
res = run_tests(sol_he, r["test"])
print(f"  HumanEval+ : status={res.status} p={res.passed} f={res.failed} "
      f"e={res.errors} all={res.all_passed}")
if not res.all_passed:
    print(f"    {res.stderr[-300:]}")
res2 = run_tests(r2["code"], r2["test"])
print(f"  MBPP+      : status={res2.status} p={res2.passed} f={res2.failed} "
      f"e={res2.errors} all={res2.all_passed}")
if not res2.all_passed:
    print(f"    {res2.stderr[-300:]}")

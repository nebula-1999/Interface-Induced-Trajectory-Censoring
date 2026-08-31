#!/usr/bin/env python3
"""诊断两件事：官方解为何 NameError，以及单次 pytest 为何要 4.6 秒。"""

from __future__ import annotations

import json
import time

from datasets import load_dataset

from sandbox import run_tests

kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

print("=" * 70)
print("A. 失败样本的原文")
for i in (17, 26, 27):
    r = kc[i]
    print(f"\n--- #{i}  subset={r['subset']} ---")
    print(f"[solution 全文]\n{r['solution']}")
    print(f"[test 前 700 字]\n{r['test'][:700]}")
    print(f"[test_info] {str(r.get('test_info'))[:300]}")
    res = run_tests(r["solution"], r["test"])
    print(f"[结果] status={res.status} p={res.passed} f={res.failed} "
          f"e={res.errors}\n{res.stderr[-500:]}")

print("\n" + "=" * 70)
print("B. 计时分解（串行，无并行干扰）")
TRIV_T = "from solution import f\n\ndef test_a():\n    assert f(1) == 1\n"
for label, sol, tst in [
    ("最简单例", "def f(x): return x", TRIV_T),
    ("真实样本", kc[0]["solution"], kc[0]["test"]),
]:
    ds = []
    for _ in range(3):
        t0 = time.time()
        run_tests(sol, tst)
        ds.append(time.time() - t0)
    print(f"  {label}: {[f'{d:.2f}s' for d in ds]}")

import subprocess, sys
t0 = time.time()
subprocess.run([sys.executable, "-c", "pass"], capture_output=True)
print(f"  裸解释器启动: {time.time() - t0:.2f}s")
t0 = time.time()
subprocess.run([sys.executable, "-c", "import pytest"], capture_output=True)
print(f"  import pytest: {time.time() - t0:.2f}s")

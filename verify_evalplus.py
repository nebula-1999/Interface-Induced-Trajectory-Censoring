#!/usr/bin/env python3
"""用官方解跑一遍 EvalPlus，验证拼接方式正确。

**这一步必须 100% 通过**。官方解都跑不过，说明 build_evalplus 的拼法错了，
后面所有 pass@1 数字都是错的——比数据质量问题严重得多。

用法：python verify_evalplus.py [n]   n 省略则跑全部 542 条
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

from datasets import load_dataset

from sandbox import build_evalplus, run_tests

n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None

recs = []
for r in load_dataset("evalplus/humanevalplus", split="test"):
    recs.append(("HumanEval+", str(r["task_id"]), r))
for r in load_dataset("evalplus/mbppplus", split="test"):
    recs.append(("MBPP+", f"Mbpp/{r['task_id']}", r))
if n_arg:
    recs = recs[:n_arg // 2] + recs[-(n_arg - n_arg // 2):]   # 两个都取到

print(f"待验 {len(recs)} 条", flush=True)
t0 = time.time()
bad, by_src = [], Counter()
for n, (src, tid, r) in enumerate(recs):
    sol, test = build_evalplus(r)
    res = run_tests(sol, test, mode="script", timeout=30.0)
    if res.all_passed:
        by_src[src] += 1
    else:
        bad.append((src, tid, res))
    if (n + 1) % 50 == 0:
        print(f"  ...{n + 1}/{len(recs)}  通过 {sum(by_src.values())}", flush=True)
el = time.time() - t0

n_by_src = Counter(s for s, _, _ in recs)
print(f"\n耗时 {el:.1f}s  单题 {el / len(recs):.2f}s")
for s in n_by_src:
    print(f"  {s:12s} {by_src[s]}/{n_by_src[s]} = {by_src[s] / n_by_src[s]:.1%}")
print(f"  合计 {sum(by_src.values())}/{len(recs)}")


# --------------------------------------------------------------------------
# EvalPlus 数据集本身已确认的数据缺陷（不是 build_evalplus 拼错、也不是模型问题）。
# 这两条官方解都过不了，属于 benchmark 的坏题；如实分类，不静默改口径：
# --------------------------------------------------------------------------
KNOWN_DEFECTS = {
    "HumanEval/32": (
        "test 断言本身写错",
        "末尾 `_poly(*candidate(*inp), inp) <= 0.0001` 把一个 float 用 `*` 展开"
        "（TypeError），且参数顺序也反了；正确应为 `_poly(inp, candidate(*inp))`。"
        "官方解 find_zero 数学上正确：对测试给出的预期根，abs(poly(xs, root))<=1e-4 通过。",
    ),
    "Mbpp/599": (
        "测试输入对参考解不可行",
        "参考解是 `sum(range(1, n+1))`，但生成测试含 n≈5e11 的输入，参考解自身也要"
        "几十亿次迭代，任何资源上限下都超时（实测 4 GB / 30 s 仍 timeout）。",
    ),
}


if bad:
    print(f"\n跑不过的前 5 条（拼法可能有问题）：")
    for src, tid, res in bad[:5]:
        print(f"  {src} {tid} status={res.status}\n    {res.stderr.strip()[-260:]}")

    print("\n分类：")
    bad_def = []
    for src, tid, _ in bad:
        if tid in KNOWN_DEFECTS:
            why, note = KNOWN_DEFECTS[tid]
            bad_def.append(tid)
            print(f"  [基准缺陷] {src} {tid} — {why}\n      {note}")
        else:
            print(f"  [需排查]   {src} {tid}（build_evalplus 可能拼错，必须修）")

    n_ver = len(recs) - len(bad)
    print(f"\n可验证且通过：{n_ver}/{len(recs)}；其中有 {len(bad_def)} 条为已确认的 "
          f"EvalPlus 数据缺陷：{', '.join(bad_def)}。")
    print("⚠️ 不要宣称 542/542。应把这 2 条缺陷题从评测集剔除并披露"
          "（或对 HumanEval/32 仅用修正后的断言证明官方解正确），"
          "其余 540 条用严格全通过口径，可在 80% power 下做配对检验。")

    _note = f"(含 {len(bad_def)} 条已知 EvalPlus 缺陷)" if len(bad) == len(bad_def) else ""
    print(f"  建议把评测集定为 {n_ver} 条 {_note}。")
else:
    print("\n✅ 官方解全部通过，build_evalplus 的拼法正确")

out = "evalplus_bad.json" if n_arg is None else None
if out:
    json.dump({"bad": [[s, t] for s, t, _ in bad],
               "known_defects": {t: KNOWN_DEFECTS[t][0] for _, t, _ in bad
                                 if t in KNOWN_DEFECTS}},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"已写出 {out}")

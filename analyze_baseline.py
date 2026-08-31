#!/usr/bin/env python3
"""解释一个矛盾：给模型 buggy 代码它能修好 47%，它自己写错的题却几乎救不回来。

搞清楚这个再花 ¥60 跑 Step 3——答案直接决定怎么解读那次训练的结果：
  - 若失败题是"改不动"（模型在改但改不对）→ 多轮能力有真实的提升余地
  - 若是"不会改"（反复提交同一份代码）→ 那 RL 该学的是"重试"这个行为本身
  - 若纯粹是"题更难"→ 两个通道不可比，repair_rate 不能当作 debug 能力的代理

数据：runs/baseline/step_00000.jsonl。两次 baseline 都写 step=0 追加进同一文件，
所以按写入顺序切分：[542 multi][454 repair] × 2 个模型。
"""

from __future__ import annotations

import json
from collections import Counter

rows = [json.loads(l) for l in open("runs/baseline/step_00000.jsonl", encoding="utf-8") if l.strip()]

# 按写入顺序切分两个模型，并用 task_id 序列自校验
n_multi, n_rep = 542, 454
blocks = [rows[0:n_multi], rows[n_multi:n_multi + n_rep],
          rows[n_multi + n_rep:2 * n_multi + n_rep],
          rows[2 * n_multi + n_rep:]]
assert [len(b) for b in blocks] == [n_multi, n_rep, n_multi, n_rep], [len(b) for b in blocks]
assert all(r["channel"] == "multi" for r in blocks[0] + blocks[2])
assert all(r["channel"] == "repair" for r in blocks[1] + blocks[3])
MODELS = [("Coder-1.5B", blocks[0], blocks[1]), ("普通 1.5B", blocks[2], blocks[3])]
print("切分自校验通过：每个模型 542 multi + 454 repair\n")


def code_changed(turns) -> int:
    """相邻两轮代码不同的次数——模型到底有没有在改。"""
    return sum(1 for a, b in zip(turns, turns[1:])
               if (a.get("code") or "").strip() != (b.get("code") or "").strip())


for name, multi, rep in MODELS:
    print("=" * 62)
    print(name)

    exhausted = [r for r in multi if len(r["turns"]) >= 4 and not r["turns"][-1]["all_passed"]]
    print(f"\n【多轮通道】跑满 4 轮仍失败 {len(exhausted)}/{len(multi)}")

    # 1) 这些题里，模型到底改没改代码
    never = sum(1 for r in exhausted if code_changed(r["turns"]) == 0)
    print(f"  从头到尾**一个字都没改**（反复提交同一份代码）: {never} "
          f"({never / max(len(exhausted), 1):.1%})")

    # 2) 改了的题，通过率有没有往上走
    moved = [r for r in exhausted if code_changed(r["turns"]) > 0]
    up = sum(1 for r in moved
             if r["turns"][-1]["passed"] > r["turns"][0]["passed"])
    down = sum(1 for r in moved
               if r["turns"][-1]["passed"] < r["turns"][0]["passed"])
    print(f"  改了代码的 {len(moved)} 条里：通过数变多 {up}、变少 {down}、"
          f"不变 {len(moved) - up - down}")

    # 3) 起点差异：失败题第一轮就一个测试都没过吗
    zero_start = sum(1 for r in exhausted if r["turns"][0]["passed"] == 0)
    print(f"  第一轮就 0 个测试通过（不是差一点，是完全不会）: {zero_start} "
          f"({zero_start / max(len(exhausted), 1):.1%})")

    # 4) 修复通道：按注入的 bug 类型看修复率
    print(f"\n【修复通道】整体 "
          f"{sum(r['turns'][-1]['all_passed'] for r in rep)}/{len(rep)}")
    by_kind: dict[str, list[int]] = {}
    for r in rep:
        k = r.get("mutation_kind") or "?"
        by_kind.setdefault(k, [0, 0])
        by_kind[k][1] += 1
        by_kind[k][0] += int(r["turns"][-1]["all_passed"])
    for k, (ok, n) in sorted(by_kind.items(), key=lambda x: -x[1][0] / x[1][1]):
        print(f"  {k:20s} {ok:3d}/{n:3d} = {ok / n:5.1%}")

    # 5) 关键对比：修复通道里，模型第一轮就修好的比例
    r1 = sum(1 for r in rep if r["turns"][0]["all_passed"])
    print(f"  其中第一轮就修好: {r1}/{len(rep)} = {r1 / len(rep):.1%}")
    print()

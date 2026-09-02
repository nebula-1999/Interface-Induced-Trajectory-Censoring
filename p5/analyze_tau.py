#!/usr/bin/env python3
"""τ-bench 结果的 per-task 分析，口径与全文其余部分同源。

**为什么要 per-task 而不是 per-request。** 两条臂的 HTTP 请求数天然不同——
repaired 臂真的调用工具，对话更长、请求更多。直接比 `解析数/请求数` 会把
「修好之后对话变长」也算进分子分母，不可比。任务是配对的（同一批 task_id、
同一用户模拟器、同一种子），所以 per-task 才是正确的配对口径。

判据复用 analysis/failure_layer.py：assistant 消息里 tool_calls 非空即
「服务端已解析」；tool_calls 为空但 content 里有调用形状的载荷即
「发出了但未被解析」。全文所有意图判据出自同一处，此处不另起炉灶。

**归因来源。** 逐题数据全部取自 τ-bench 自己写出的结果 JSON——每条记录自带
`task_id` 与完整 `traj`。**本脚本不读记录代理的日志**，因此不存在「按串行顺序把
请求对上任务」这种假设。代理日志只用于**聚合层面的有效性检查**（每个请求是否都
带了 tools、tool_choice 是否为 auto、HTTP 错误数），那些量不需要逐题归因。
要做逐题的 HTTP 级归因，唯一严谨的办法是给 τ-bench 的 agent 注入 task-id 请求头，
但那会改动 benchmark 本身，而「无需任何适配即可复现」正是本实验相对 BFCL 的优势，
且逐题科学结论已完整存在于 traj 中——因此不做。

用法：
    python3 p5/analyze_tau.py <documented.json> <repaired.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.failure_layer import HERMES_TOOL_CALL_REGEX  # noqa: E402

# 调用形状：τ-bench 的工具名不是 run_tests，所以判据放宽到「命名 + 参数」的
# JSON 对象，与 analysis/intent.py 的 strong 档同构（不含 run_tests 硬编码）。
NAMED_CALL = re.compile(r'\{\s*"(?:name|function)"\s*:\s*"[A-Za-z_][A-Za-z0-9_]*"'
                        r'.{0,400}?"(?:arguments|parameters)"\s*:', re.S)


def emitted_unparsed(content: str) -> bool:
    """content 里是否有调用形状的载荷（此时 tool_calls 为空）。"""
    if not content:
        return False
    return bool(NAMED_CALL.search(content) or HERMES_TOOL_CALL_REGEX.search(content))


def per_task(path: str) -> dict[int, dict]:
    out = {}
    for rec in json.load(open(path, encoding="utf-8")):
        asst = [m for m in rec["traj"] if m.get("role") == "assistant"]
        out[rec["task_id"]] = {
            "reward": float(rec.get("reward") or 0.0),
            "assistant_turns": len(asst),
            "parsed": sum(1 for m in asst if m.get("tool_calls")),
            "emitted_unparsed": sum(1 for m in asst
                                    if not m.get("tool_calls")
                                    and emitted_unparsed(m.get("content") or "")),
            "observations": sum(1 for m in rec["traj"] if m.get("role") == "tool"),
        }
    return out


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    A, B = per_task(sys.argv[1]), per_task(sys.argv[2])
    ks = sorted(set(A) & set(B))
    if not ks:
        sys.exit("两个文件没有共同的 task_id，无法配对")

    # 任一臂缺失的任务必须显式报出来。出错的任务可能根本不进结果文件，
    # 静默丢弃会让分母随臂变化——这正是最容易产生偏向性结论的地方。
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        print(f"★ 未配对任务：仅 documented {len(only_a)} 个 {only_a[:8]}，"
              f"仅 repaired {len(only_b)} 个 {only_b[:8]}")
        print("  它们已被排除。若数量不小或集中在一条臂，结论需重新审视——"
              "任务在某条臂里出错而被丢弃，本身可能就是接口造成的。\n")

    def agg(D, key):
        return sum(D[k][key] for k in ks)

    print(f"配对 task {len(ks)} 个（仅统计两臂都有的，未配对的一律丢弃）\n")
    print(f"  {'':<26}{'documented':>12}{'repaired':>11}")
    print("  " + "-" * 49)
    for lab, key in [("任务成功数 (reward=1)", None),
                     ("assistant 轮次", "assistant_turns"),
                     ("服务端解析出调用", "parsed"),
                     ("发出但未被解析", "emitted_unparsed"),
                     ("工具执行/observation", "observations")]:
        if key is None:
            a = sum(1 for k in ks if A[k]["reward"] >= 1.0)
            b = sum(1 for k in ks if B[k]["reward"] >= 1.0)
        else:
            a, b = agg(A, key), agg(B, key)
        print(f"  {lab:<26}{a:>12}{b:>11}")
    print(f"  {'平均 reward':<26}{agg(A,'reward')/len(ks):>12.3f}{agg(B,'reward')/len(ks):>11.3f}")

    # 配对的逐题差：多少题在 repaired 下才出现工具执行
    only_b = [k for k in ks if A[k]["observations"] == 0 and B[k]["observations"] > 0]
    only_a = [k for k in ks if B[k]["observations"] == 0 and A[k]["observations"] > 0]
    print(f"\n  仅 repaired 有工具执行的题: {len(only_b)}   仅 documented 有的: {len(only_a)}")
    print("  ★ 这是配对口径下最直接的量：接口决定了多少题能进入工具循环")


if __name__ == "__main__":
    main()

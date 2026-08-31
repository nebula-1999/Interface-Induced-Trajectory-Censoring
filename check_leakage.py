#!/usr/bin/env python3
"""核对 EvalPlus 实际条数 + 训练/评测集泄漏。

Step 0 的 CI 表按标准口径填的是 HumanEval+ = 164、MBPP+ = 378。
EvalPlus 改过版本，必须按实际下载到的数据核对——N 变了，MDD 就变了。

更要紧的是第二件事：MBPP+ 源自 MBPP sanitized，而训练集大概率也要从
MBPP 来。必须逐条比对 task_id，把重叠部分从训练集里剔掉。
这一条不查会毁掉整篇报告。

先探查 schema 再比对，不对字段名做任何假设。
"""

from __future__ import annotations

import re

from datasets import load_dataset


def probe(path: str, config: str | None = None) -> dict:
    """加载数据集并打印 split / 条数 / 字段 / 样例 task_id。

    config 是数据集的子配置名（mbpp 有 full / sanitized 两个）。
    不要把这个参数叫 name——会和 load_dataset 的位置参数撞上。
    """
    label = path if config is None else f"{path} [{config}]"
    print(f"\n{'=' * 60}\n{label}")
    try:
        ds = load_dataset(path, config) if config else load_dataset(path)
    except Exception as e:  # 数据集改名或需要 config 时给出可读信息
        print(f"  加载失败: {type(e).__name__}: {e}")
        return {}
    out = {}
    for split, d in ds.items():
        ids = d["task_id"] if "task_id" in d.column_names else []
        print(f"  split={split:12s} n={len(d):5d}  字段={d.column_names}")
        if ids:
            print(f"    task_id 类型={type(ids[0]).__name__}  样例={ids[:3]}")
        out[split] = ids
    return out


def to_int(tid) -> int | None:
    """把 'Mbpp/11' / 11 / '11' 统一成整数，取不出就返回 None。"""
    if isinstance(tid, int):
        return tid
    m = re.search(r"(\d+)", str(tid))
    return int(m.group(1)) if m else None


def main() -> None:
    he = probe("evalplus/humanevalplus")
    mbpp_plus = probe("evalplus/mbppplus")
    mbpp = probe("google-research-datasets/mbpp", "full")
    mbpp_san = probe("google-research-datasets/mbpp", "sanitized")

    print(f"\n{'=' * 60}\n泄漏比对：MBPP+ 的 task_id vs MBPP 各 split")

    plus_ids = set()
    for split, ids in mbpp_plus.items():
        plus_ids |= {to_int(t) for t in ids}
    plus_ids.discard(None)
    print(f"  MBPP+ 去重后 task_id 数 = {len(plus_ids)}")

    for label, sets in (("full", mbpp), ("sanitized", mbpp_san)):
        for split, ids in sets.items():
            s = {to_int(t) for t in ids}
            s.discard(None)
            ov = plus_ids & s
            flag = "  <-- 训练集必须剔掉" if ov and split == "train" else ""
            print(f"  mbpp[{label}]/{split:12s} n={len(s):5d}  "
                  f"与 MBPP+ 重叠 = {len(ov):5d}{flag}")

    print(f"\n{'=' * 60}\n对 Step 0 CI 表的影响")
    n_he = max((len(v) for v in he.values()), default=0)
    n_mb = max((len(v) for v in mbpp_plus.values()), default=0)
    print(f"  实测 HumanEval+ = {n_he}（表里填的 164）")
    print(f"  实测 MBPP+      = {n_mb}（表里填的 378）")
    print(f"  实测合并        = {n_he + n_mb}（表里填的 542）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""探查训练集候选——只拉 metadata，不下载全量。

MBPP 因泄漏被排除后，训练集必须另找。硬要求两条：
  1. 有**可执行的单元测试**（奖励函数要靠它）
  2. 格式最好是函数补全，和 HumanEval+/MBPP+ 的评测格式一致；
     竞赛式 stdin/stdout 题格式不同，会额外引入分布迁移风险。
"""

from __future__ import annotations

from datasets import load_dataset_builder

CANDIDATES = [
    ("KodCode/KodCode-Light-RL-10K", None),
    ("KodCode/KodCode-V1", None),
    ("codeparrot/apps", None),
    ("BAAI/TACO", None),
    ("deepmind/code_contests", None),
]

for path, cfg in CANDIDATES:
    print(f"\n{'=' * 60}\n{path}")
    try:
        b = load_dataset_builder(path, cfg) if cfg else load_dataset_builder(path)
        info = b.info
        for name, sp in (info.splits or {}).items():
            print(f"  split={name:12s} n={sp.num_examples}")
        feats = list((info.features or {}).keys())
        print(f"  字段={feats}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {str(e)[:200]}")

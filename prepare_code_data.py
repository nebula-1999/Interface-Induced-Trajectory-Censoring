#!/usr/bin/env python3
"""把 KodCode 的干净样本转成 verl 要的 parquet。

    python prepare_code_data.py --out-dir /root/autodl-tmp/train/code-data

数据来源是 verified_ids.json —— 已经过两道筛：
  1. decontaminate.py 按 n-gram containment 剔掉与 HumanEval+/MBPP+ 重合的题
  2. verify_kodcode.py 剔掉官方解跑不过自己测试的题（留着会让奖励永远给不满）

test 要放**两个**地方，用途不同，缺一不可：
  - extra_info.tools_kwargs.run_tests.create_kwargs.test  → CodeTool.create 拿它跑沙箱
  - reward_model.ground_truth                             → reward_code.compute_score 拿它算最终奖励
（`calc_reward` 在 verl 0.9.0 是死接口，奖励只能走后者。）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ReAct 文本协议。实测 Qwen2.5-Coder 全系列在 function calling 的 auto 模式下
# 从不主动调用（0/1000，跨 1.5B–32B），换 ReAct 后 95–100% 主动发起，
# 因此训练必须走 ReAct，否则多轮信号无效（tool_calls/mean 恒为 0）。
SYSTEM = """你是一个 Python 编程助手，按以下格式逐步工作：

Thought: 你的思考
Action: run_tests
Action Input: ```python
<完整代码>
```
Observation: （由系统填写测试结果）

... 可以重复以上循环 ...

Thought: 我已确认代码正确
Final Answer: ```python
<最终代码>
```"""


def build(rec: dict, idx: int) -> dict | None:
    q = (rec.get("question") or "").strip()
    test = (rec.get("test") or "").strip()
    if not q or not test:
        return None
    return {
        "data_source": "kodcode",
        # **必须在顶层**：verl 从样本顶层读 agent_name，放进 extra_info 会被无视，
        # 静默退回 single_turn_agent —— 训练看似正常实则单轮。
        "agent_name": "react_agent",
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": q}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": test},
        "extra_info": {
            "index": idx,
            "question_id": str(rec.get("question_id", "")),
            "subset": rec.get("subset", ""),
            "difficulty": rec.get("gpt_difficulty", ""),
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "run_tests": {
                    "create_kwargs": {"test": test},
                    # Parquet 写不了空 struct，补占位字段；
                    # CodeTool.execute 有 **kwargs，会忽略它
                    "execute_kwargs": {"_": 0},
                    "tool_config": {"timeout": 10.0, "mem_mb": 512},
                }
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default="verified_ids.json")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/train/code-data")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    idxs = json.load(open(a.verified, encoding="utf-8"))["verified_index"]
    print(f"可用样本 {len(idxs)} 条")

    rng = random.Random(a.seed)
    rng.shuffle(idxs)
    # val 先切走，确保与 train 无交集
    val_idx = idxs[:a.n_val]
    train_idx = idxs[a.n_val:a.n_val + a.n_train]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, sel in (("train", train_idx), ("val", val_idx)):
        rows, n = [], 0
        for i in sel:
            r = build(kc[i], n)
            if r is not None:
                rows.append(r)
                n += 1
        pd.DataFrame(rows).to_parquet(out / f"{name}.parquet", index=False)
        print(f"→ {out / name}.parquet  {len(rows)} 条")

    # 自检：训练需要 150 步 × batch 16 = 2400 个样本，不重复才不会被背下来
    need = 2400
    n_tr = len(train_idx)
    print(f"\n训练集 {n_tr} 条，150 步 × batch 16 需要 {need} 个样本："
          f"{'✅ 不必重复' if n_tr >= need else '❌ 不足，会重复采样'}")


if __name__ == "__main__":
    main()

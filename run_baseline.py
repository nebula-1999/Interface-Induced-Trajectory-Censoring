#!/usr/bin/env python3
"""独立跑一次分解评测（baseline，不训练）。

CodeEvalHook 平时挂在 verl 的 _validate 上、复用训练自带的 rollout 服务；
baseline 阶段没有训练进程，所以需要这个入口：连一个独立起的 vLLM serve。

    python run_baseline.py --model /root/autodl-tmp/models/Qwen2.5-Coder-1.5B-Instruct

选 base model 的标准**不是谁强，是谁留下足够的提升空间**——太强会撞天花板，
6 个点的止损线就涨不出来。所以要同时看 turn1_pass（初稿质量）和
repair_rate（真 debug 能力），两个都接近满分的那个不要选。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

from code_eval_hook import CodeEvalHook


def wait_ready(port: int, timeout: int = 600) -> None:
    """等 vLLM 起来。冷启动要加载权重 + 编译，1.5B 通常 1-3 分钟。"""
    t0 = time.time()
    url = f"http://localhost:{port}/v1/models"
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"vLLM 就绪（等了 {time.time() - t0:.0f}s）", flush=True)
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"vLLM 在 {timeout}s 内没起来，检查 vllm.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", default="/root/autodl-tmp/runs/baseline")
    ap.add_argument("--probes", default="/root/autodl-tmp/code-agent/probes_repair.jsonl")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    wait_ready(a.port)
    hook = CodeEvalHook([f"localhost:{a.port}"], model_name=a.model,
                        out_dir=a.out_dir, probes_path=a.probes,
                        max_turns=a.max_turns)
    t0 = time.time()
    m = hook.run(step=0)
    print(f"\n=== baseline {a.tag or a.model} （{time.time() - t0:.0f}s）===")
    print(json.dumps(m, ensure_ascii=False, indent=2))

    # 选型判据直接打出来，免得看一堆数字还要自己算
    t1, fin = m.get("code/turn1_pass", 0), m.get("code/final_pass", 0)
    rep = m.get("code/repair_rate", 0)
    print(f"\nturn-1 {t1:.1%} → final {fin:.1%}（多轮增量 {fin - t1:+.1%}）"
          f"，scripted 修复率 {rep:.1%}")
    if fin > 0.85:
        print("⚠️  final 已超过 85%，天花板太近，6 个点的提升空间不够 —— 换更弱的 base")
    elif fin < 0.15:
        print("⚠️  final 低于 15%，多半是格式/prompt 问题而非能力问题，先查轨迹再下结论")
    else:
        print("✅ 提升空间合适")


if __name__ == "__main__":
    main()

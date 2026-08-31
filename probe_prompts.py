#!/usr/bin/env python3
"""对比几种 system prompt，找出能让模型在 tool_choice=auto 下主动调用工具的措辞。

背景：训练时 tool_choice=auto，而模型 0/6 主动调用（强制 required 时 6/6 成功，
说明能力没问题）。原措辞"你**可以**调用 run_tests"太弱，模型直接给答案了事，
于是 150 步训练全程没有真实的多轮反馈。
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from datasets import load_dataset

VARIANTS = {
    "A 原版（可以）": """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。请给出完整的 Python 代码，包含所需的 import。""",

    "B 强制（必须先验证）": """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

**你必须先调用 run_tests 工具来验证你的代码**，不要直接给出答案。
工具会返回测试通过情况与报错；如果有测试失败，请根据报错修改代码并再次调用
run_tests，直到全部通过为止。""",

    "C 强制+流程说明": """你是一个 Python 编程助手。

工作流程（必须严格遵守）：
1. 阅读题目，写出完整的 Python 实现（含所需 import）
2. **调用 run_tests 工具**提交这份代码，等待测试结果
3. 若有测试失败，分析报错、修改代码，再次调用 run_tests
4. 全部测试通过后，才给出最终答案

不要跳过第 2 步。任何未经 run_tests 验证的代码都不算完成。""",
}

TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string",
                                           "description": "完整的 Python 代码（函数定义及其依赖的 import）"}},
                   "required": ["code"]}}}]


def ask(port, model, system, question):
    body = {"model": model, "temperature": 0.0, "max_tokens": 1024,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": question}],
            "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
    qs = [kc[i]["question"] for i in clean[:a.n]]

    print(f"每个变体测 {len(qs)} 道真实训练题，tool_choice=auto\n")
    print(f"{'变体':24}{'主动调用':>10}{'调用率':>10}")
    print("-" * 46)
    best = None
    for name, sysmsg in VARIANTS.items():
        n_tc = 0
        for q in qs:
            d = ask(a.port, a.model, sysmsg, q)
            if "_error" in d:
                continue
            if (d["choices"][0].get("message") or {}).get("tool_calls"):
                n_tc += 1
        rate = n_tc / len(qs)
        print(f"{name:24}{n_tc:>6}/{len(qs):<4}{rate:>9.0%}")
        if best is None or rate > best[1]:
            best = (name, rate)
    print(f"\n最佳：{best[0]}（{best[1]:.0%}）")
    print("训练要求：调用率需接近 100%，否则仍有大量轨迹拿不到真实的多轮反馈。")


if __name__ == "__main__":
    main()

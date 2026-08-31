#!/usr/bin/env python3
"""协议消融：0/1000 到底是模型的性质，还是这套技术栈的性质？

已排除：parser、template 注入、prompt 措辞、few-shot、任务需求、题目难度、规模。
本脚本补测四个仍未排除、且足以翻转结论的条件：

  T1 采样解码      —— 全程 temperature=0 是贪心。若"直接给代码"概率 0.6、
                      "调用工具"0.3，贪心永远选前者，采样却会有 30% 调用。
  T2 不传 tool_choice —— OpenAI 规范里不传即默认 auto，但显式传字符串 "auto"
                      与不传，vLLM 未必走同一条路径。
  T3 去掉压制句    —— system prompt 里"请给出完整的 Python 代码"本身就是在
                      要求直接给代码，可能压制工具调用。
  T4 ReAct 格式    —— 只测了 OpenAI function calling。大量 agent 框架用的是
                      Thought/Action/Observation 文本协议，小模型表现可能完全不同。
                      **这条最可能翻转结论。**

任何一条 > 0，0/1000 都必须重新表述。
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request

from datasets import load_dataset

TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string", "description": "完整的 Python 代码"}},
                   "required": ["code"]}}}]

SYS_ORIG = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。请给出完整的 Python 代码，包含所需的 import。"""

# T3：删掉最后那句"请给出完整的 Python 代码"——它本身就在要求直接作答
SYS_NOPUSH = """你是一个 Python 编程助手。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。"""

# T4：ReAct 文本协议，完全不用 OpenAI function calling
SYS_REACT = """你是一个 Python 编程助手，按以下格式逐步工作：

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
```

**必须先用 Action 提交代码验证，再给 Final Answer。**"""

_ACTION = re.compile(r"^\s*Action\s*:\s*run_tests", re.M | re.I)


def post(port, body):
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["choices"][0]
    except Exception as e:
        return {"_error": str(e)}


def called(ch) -> bool:
    return bool((ch.get("message") or {}).get("tool_calls"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
    qs = [kc[i]["question"] for i in clean[:a.n]]

    print(f"模型: {a.model}   每条件 n={len(qs)}\n")
    print(f"{'条件':30}{'触发调用':>12}{'比率':>9}")
    print("-" * 53)
    res = {}

    def report(name, hits, tot, errs=0):
        res[name] = hits / tot if tot else 0.0
        note = f"  ({errs} 次错误)" if errs else ""
        print(f"{name:30}{hits:>7}/{tot:<4}{hits/tot:>8.0%}{note}")

    # 基线复现
    h = e = 0
    for q in qs:
        ch = post(a.port, {"model": a.model, "temperature": 0.0, "max_tokens": 1024,
                           "messages": [{"role": "system", "content": SYS_ORIG},
                                        {"role": "user", "content": q}],
                           "tools": TOOLS, "tool_choice": "auto"})
        if "_error" in ch: e += 1; continue
        h += called(ch)
    report("基线（temp=0, 显式 auto）", h, len(qs), e)

    # T1 采样解码
    for temp in (0.7, 1.0):
        h = e = 0
        for q in qs:
            ch = post(a.port, {"model": a.model, "temperature": temp, "top_p": 0.95,
                               "max_tokens": 1024,
                               "messages": [{"role": "system", "content": SYS_ORIG},
                                            {"role": "user", "content": q}],
                               "tools": TOOLS, "tool_choice": "auto"})
            if "_error" in ch: e += 1; continue
            h += called(ch)
        report(f"T1 采样 temperature={temp}", h, len(qs), e)

    # T2 不传 tool_choice
    h = e = 0
    for q in qs:
        ch = post(a.port, {"model": a.model, "temperature": 0.0, "max_tokens": 1024,
                           "messages": [{"role": "system", "content": SYS_ORIG},
                                        {"role": "user", "content": q}],
                           "tools": TOOLS})       # 不传 tool_choice
        if "_error" in ch: e += 1; continue
        h += called(ch)
    report("T2 不传 tool_choice", h, len(qs), e)

    # T3 去掉压制句
    h = e = 0
    for q in qs:
        ch = post(a.port, {"model": a.model, "temperature": 0.0, "max_tokens": 1024,
                           "messages": [{"role": "system", "content": SYS_NOPUSH},
                                        {"role": "user", "content": q}],
                           "tools": TOOLS, "tool_choice": "auto"})
        if "_error" in ch: e += 1; continue
        h += called(ch)
    report("T3 去掉『请给出完整代码』", h, len(qs), e)

    # T4 ReAct 文本协议（不传 tools，改看文本里有没有 Action:）
    h = e = 0
    for q in qs:
        ch = post(a.port, {"model": a.model, "temperature": 0.0, "max_tokens": 1024,
                           "messages": [{"role": "system", "content": SYS_REACT},
                                        {"role": "user", "content": q}]})
        if "_error" in ch: e += 1; continue
        if _ACTION.search((ch.get("message") or {}).get("content") or ""):
            h += 1
    report("T4 ReAct 文本协议", h, len(qs), e)

    print("\n" + "=" * 53)
    flipped = [k for k, v in res.items() if not k.startswith("基线") and v > 0]
    if flipped:
        print("⚠️  以下条件下模型**会**发起工具调用：")
        for k in flipped:
            print(f"      {k}  ({res[k]:.0%})")
        print("    → 0/1000 必须重新表述为『在该特定协议下不主动调用』。")
    else:
        print("✅ 四个条件全部为 0 —— 不是解码、不是传参、不是压制句、也不是协议格式。")


if __name__ == "__main__":
    main()

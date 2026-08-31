#!/usr/bin/env python3
"""阳性对照：验证"主动调用"这个行为到底测不测得出来。

**为什么必须做**：五个规模 200 题全是 0/200。有两种解释——
  (a) 这些模型确实不主动调用工具
  (b) **我们的测试协议本身测不出这个行为**，换谁来都是 0
没有阳性对照，(b) 无法排除，0/1000 这个数字就不可信。

三个条件，逐个放松限制：
  A 基线      = 现有协议（"你可以调用 run_tests"）
  B few-shot  = prompt 里给一段完整的"调用工具"示例对话
  C 工具必要  = 换成**不执行就答不出**的题（现有题目全是"写个函数"，
                模型凭知识直接写对，工具没有信息增益——这可能才是根因）

判读：
  A=0 B>0        → 协议偏严，模型需要示例；0/1000 要改写为"默认 prompt 下"
  A=0 B=0 C>0    → 是任务设计问题：我们的题不需要工具
  三者皆 0       → 模型确实不主动调用，0/1000 成立
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from datasets import load_dataset

TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string", "description": "完整的 Python 代码"}},
                   "required": ["code"]}}}]

SYS = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。请给出完整的 Python 代码，包含所需的 import。"""

# B：few-shot —— 用一轮完整的工具调用示例，把"该怎么做"演示给模型
FEWSHOT = [
    {"role": "user", "content": "写一个函数 double(x)，返回 x 的两倍。"},
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_demo", "type": "function",
        "function": {"name": "run_tests",
                     "arguments": json.dumps({"code": "def double(x):\n    return x * 2"})}}]},
    {"role": "tool", "tool_call_id": "call_demo", "name": "run_tests",
     "content": "3/3 个测试通过。"},
    {"role": "assistant", "content": "测试全部通过，实现如下：\n```python\ndef double(x):\n    return x * 2\n```"},
]

# C：工具必要型任务 —— 不执行就无法回答，凭知识写不出来
TOOL_NEEDED = [
    "下面的函数在输入 n=999983（一个质数）时返回什么？请给出确切的返回值。\n"
    "```python\ndef f(n):\n    s = 0\n    for i in range(1, n):\n        if n % i == 0:\n            s += i\n    return s\n```",
    "这段代码在 Python 3.12 下会抛出什么异常？请给出确切的异常类型和消息。\n"
    "```python\nd = {'a': 1}\nprint(d['b'])\n```",
    "计算 sum(i*i for i in range(1, 100001)) 的确切数值。",
    "下面这段代码的输出是什么？逐行给出。\n"
    "```python\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x, y, x is y)\n```",
    "字符串 'hello world' 经过 ''.join(sorted(set('hello world'))) 后的确切结果是什么？",
]


def ask(port, model, messages):
    body = {"model": model, "temperature": 0.0, "max_tokens": 1024,
            "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["choices"][0]
    except Exception as e:
        return {"_error": str(e)}


def count(port, model, cases, prefix=()):
    n = 0
    errs = 0
    for q in cases:
        msgs = [{"role": "system", "content": SYS}] + list(prefix) + \
               [{"role": "user", "content": q}]
        ch = ask(port, model, msgs)
        if "_error" in ch:
            errs += 1
            continue
        if ((ch.get("message") or {}).get("tool_calls")):
            n += 1
    return n, len(cases), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
    normal = [kc[i]["question"] for i in clean[:a.n]]
    hard = [kc[i]["question"] for i in clean
            if kc[i].get("gpt_difficulty") == "hard"][:a.n]

    print(f"模型: {a.model}\n")
    rows = [
        ("A 基线（现有协议）", normal, ()),
        ("B few-shot 示例", normal, FEWSHOT),
        ("C 工具必要型任务", TOOL_NEEDED, ()),
        ("D 难题子集", hard, ()),
    ]
    res = {}
    print(f"{'条件':22}{'主动调用':>12}{'调用率':>9}")
    print("-" * 45)
    for name, cases, prefix in rows:
        if not cases:
            print(f"{name:22}{'(无样本)':>12}"); continue
        n, tot, errs = count(a.port, a.model, cases, prefix)
        res[name] = n / tot
        note = f"  ({errs} 次请求错误)" if errs else ""
        print(f"{name:22}{n:>7}/{tot:<4}{n/tot:>8.0%}{note}")

    print("\n" + "=" * 45)
    if any(v > 0 for k, v in res.items() if not k.startswith("A")):
        print("⚠️  某个放松条件下模型开始主动调用 —— **协议偏严**，")
        print("    0/1000 必须改写为『默认 prompt / 该类任务下不主动调用』。")
    else:
        print("✅ 全部条件下均为 0 —— 模型确实不主动调用，0/1000 成立。")
        print("   （注意：这仍不能排除所有协议问题，但已排除最可能的三种。）")


if __name__ == "__main__":
    main()

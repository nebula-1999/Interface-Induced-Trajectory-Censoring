#!/usr/bin/env python3
"""验证强制第一轮调用后，模型会不会自己继续多轮 debug。

这决定了 patch verl 值不值得：
  - 若拿到失败反馈后模型继续调工具 → 强制首轮即可，patch 有意义
  - 若只调一次就收手 → 需要全程 required，任务形态会变，要重新权衡
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from datasets import load_dataset

from sandbox import run_tests

SYSTEM = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。请给出完整的 Python 代码，包含所需的 import。"""

TOOLS = [{"type": "function", "function": {
    "name": "run_tests", "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                   "required": ["code"]}}}]


def chat(port, model, messages, tool_choice):
    body = {"model": model, "temperature": 0.0, "max_tokens": 1024,
            "messages": messages, "tools": TOOLS, "tool_choice": tool_choice}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
    items = [(kc[i]["question"], kc[i]["test"]) for i in clean[:a.n]]

    n_first, n_failed, n_second = 0, 0, 0
    for q, test in items:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        # 第 1 轮：强制调用
        ch = chat(a.port, a.model, msgs, "required")
        tcs = (ch.get("message") or {}).get("tool_calls") or []
        if not tcs:
            continue
        n_first += 1
        args = tcs[0]["function"]["arguments"]
        code = json.loads(args).get("code", "") if isinstance(args, str) else args.get("code", "")
        res = run_tests(code, test)
        if res.all_passed:
            continue                     # 一次就过，没有 debug 的机会
        n_failed += 1
        # 把工具结果按 OpenAI 规范喂回去，第 2 轮放开成 auto
        msgs += [ch["message"],
                 {"role": "tool", "tool_call_id": tcs[0]["id"], "name": "run_tests",
                  "content": f"{res.passed}/{res.total} 个测试通过。\n{res.stderr[-600:]}"}]
        ch2 = chat(a.port, a.model, msgs, "auto")
        if ((ch2.get("message") or {}).get("tool_calls")):
            n_second += 1

    print(f"\n强制首轮成功调用      : {n_first}/{len(items)}")
    print(f"其中首轮未通过（有的可修）: {n_failed}")
    print(f"**收到失败反馈后主动再调**: {n_second}/{n_failed if n_failed else 1}")
    print("\n判读：第二轮主动调用率高 → 只需强制首轮，patch 代价小；")
    print("      接近 0 → 必须全程强制，多轮就成了外部驱动而非模型自主。")


if __name__ == "__main__":
    main()

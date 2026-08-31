#!/usr/bin/env python3
"""用**真实训练数据**重测模型会不会调用工具。

第一版探测的缺陷：测试用例是"写一个 add(a,b)"，这题太简单，模型直接给答案
是合理行为，测不出真实训练场景下的表现。这一版改用：
  - prepare_code_data.py 里训练时实际使用的 SYSTEM
  - KodCode 训练集里的真实题目（含难题）
  - 对比 tool_choice: auto / required 两种设置
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from datasets import load_dataset

SYSTEM = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。请给出完整的 Python 代码，包含所需的 import。"""

TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string",
                                           "description": "完整的 Python 代码（函数定义及其依赖的 import）"}},
                   "required": ["code"]}}}]


def ask(port, model, question, tool_choice):
    body = {"model": model, "temperature": 0.0, "max_tokens": 1024,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question}],
            "tools": TOOLS}
    if tool_choice:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
    qs = [kc[i]["question"] for i in clean[:a.n]]

    for choice_name, choice in (("auto（训练时的设置）", "auto"),
                                ("required（强制调用）", "required")):
        n_tc = n_tag = 0
        errs = []
        for q in qs:
            d = ask(a.port, a.model, q, choice)
            if "_error" in d:
                errs.append(d["_error"]); continue
            msg = d["choices"][0].get("message") or {}
            if msg.get("tool_calls"):
                n_tc += 1
            if "<tool_call>" in (msg.get("content") or ""):
                n_tag += 1
        print(f"\ntool_choice = {choice_name}")
        print(f"  解析出 tool_calls : {n_tc}/{len(qs)}")
        print(f"  content 含标签    : {n_tag}/{len(qs)}")
        if errs:
            print(f"  请求错误 {len(errs)} 次: {errs[0][:120]}")

    print("\n" + "=" * 62)
    print("判读：auto 下为 0 而 required 下 > 0 → 模型有能力、只是不主动用，")
    print("      训练时可强制或改 prompt；两者皆 0 → 模型确实不会用工具。")


if __name__ == "__main__":
    main()

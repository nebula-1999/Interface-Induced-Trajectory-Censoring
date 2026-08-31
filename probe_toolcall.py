#!/usr/bin/env python3
"""诊断：模型到底有没有发出工具调用，以及 vLLM 能不能解析出来。

**为什么需要这个**：150 步训练里 `timing_s/agent_loop/tool_calls/mean` 全程
为 0.0，而工具 schema 明明被 AgentLoopWorker 正确加载了。已知问题是
Qwen2.5-Coder-1.5B-Instruct 与 vLLM 默认 tool parser 不兼容，模型永远返回
finish_reason="stop" 而非 "tool_calls"，于是 agent loop 直接收工、工具从不执行。

修法取决于故障在哪一环，这个脚本就是用来区分的：

  A. content 里**没有** <tool_call> → 模型根本没发 → 要改 prompt / chat template
  B. content 里**有** <tool_call> 但 tool_calls 字段为空 → 解析失败 → 要写 parser
  C. tool_calls 字段有内容 → 解析正常 → 故障在 verl 侧的路由

用法（vLLM 需已在 --port 上运行）：
    python probe_toolcall.py --model <路径> [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "完整的 Python 代码"},
            },
            "required": ["code"],
        },
    },
}]

SYSTEM = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。请给出完整的 Python 代码，包含所需的 import。"""

USER = "写一个函数 add(a, b)，返回两数之和。"


def ask(port: int, model: str, with_tools: bool) -> dict:
    body = {"model": model, "temperature": 0.0, "max_tokens": 512,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER}]}
    if with_tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def report(tag: str, d: dict) -> None:
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    print(f"\n{'=' * 68}\n{tag}")
    print(f"  finish_reason : {ch.get('finish_reason')!r}")
    print(f"  tool_calls 字段: {len(tcs)} 个" + (f" -> {tcs[0]['function']['name']}" if tcs else ""))
    print(f"  content 含 <tool_call> 标签: {'<tool_call>' in content}")
    print(f"  --- content 前 400 字 ---\n{content[:400]}")
    if tcs:
        print(f"  --- 解析出的 arguments 前 200 字 ---\n"
              f"{str(tcs[0]['function'].get('arguments'))[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    d_no = ask(a.port, a.model, with_tools=False)
    report("【对照】不传 tools（评测端就是这个形态）", d_no)
    d_yes = ask(a.port, a.model, with_tools=True)
    report("【实验】传 tools（训练端 agent loop 的形态）", d_yes)

    ch = d_yes["choices"][0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    print(f"\n{'=' * 68}\n诊断结论")
    if tcs:
        print("  → C：解析正常，tool_calls 字段有内容。")
        print("     故障在 verl 侧的路由/配置，不是 parser。")
    elif "<tool_call>" in content:
        print("  → B：模型发了 <tool_call>，但 vLLM 没解析成 tool_calls 字段。")
        print("     **这就是已知的 Qwen2.5-Coder + 默认 parser 不兼容**。")
        print("     修法：vLLM 启动时加 --enable-auto-tool-choice")
        print("           --tool-call-parser hermes（或自定义 parser）。")
    else:
        print("  → A：模型根本没有生成 <tool_call>。")
        print("     修法在 prompt / chat template 一侧，写 parser 没用。")


if __name__ == "__main__":
    main()

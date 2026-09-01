#!/usr/bin/env python3
"""把「模型在尝试调用但服务端没解析出来」拆成失败发生在哪一层。

`analysis/intent.py` 回答的是**有没有意图**，这个模块回答的是**卡在哪一层**，
两者正交，谁也不替代谁。之所以需要它：探针原来把所有 strong-intent 一律标成
「格式完好但 parser 不认 → 静默低估」，而那句话从未校验过 JSON 合法性。实测
Qwen2.5-7B-Instruct 的 34 条里，**格式完好的是 0 条**。

判据不是自己重写一套，而是**逐行重放 vLLM 0.27.1 的 hermes parser**
（`vllm/tool_parsers/hermes_tool_parser.py` 的非流式分支），这样
「parser 本应解析成功却没有」才是可证伪的，而不是一句断言。

四档：

``server_parsed``  服务端解析成功，不属于损失。
``no_envelope``    输出里没有 ``<tool_call>`` 标签。hermes 见不到调用是**正确行为**，
                   不是 bug。Qwen2.5-Coder 全尺寸都落在这一档：载荷是合法 JSON，
                   缺的是包装层。
``bad_payload``    有标签，但载荷本身就不是合法 JSON，宽松解码也救不回来。
                   典型成因是把 Python 代码塞进 JSON 字符串时没转义：
                   docstring 的 ``\"\"\"``、裸换行、坏转义。
``strict_only``    载荷本身合法，只是 ``json.loads`` 不容忍捕获组尾部的残留文本。
                   换成 ``raw_decode`` 就能救回——**这一档才是 parser 严格度造成的**。
``parser_loss``    照 hermes 原样重放本应解析成功，服务端却没解析出来。
                   这一档才配叫「静默低估」。

用法::

    from analysis.failure_layer import classify, TIERS
    tier = classify(raw_output, server_parsed=turn["parse_mode"] == "fc_tool_call")
"""
from __future__ import annotations

import json
import re

# vLLM 0.27.1 hermes_tool_parser.py 第 38-40 行，逐字照抄。
# 改这里等于换了判据，任何修改都必须同时更新引用的 vLLM 版本号。
HERMES_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)
HERMES_START_TOKEN = "<tool_call>"
_DECODER = json.JSONDecoder()

TIERS = ("server_parsed", "no_envelope", "parser_loss", "strict_only", "bad_payload")

TIER_ZH = {
    "server_parsed": "服务端已解析",
    "no_envelope": "无 <tool_call> 包装层（parser 看不见，属正确行为）",
    "parser_loss": "★ 真·parser 损失（重放本应成功）",
    "strict_only": "载荷合法，仅因尾部残留被 json.loads 拒绝",
    "bad_payload": "载荷本身非法（模型自己写坏）",
}


def _hermes_strict(model_output: str) -> bool:
    """hermes 现在的做法：整段捕获交给 json.loads，任一条失败则整体失败。"""
    try:
        captures = HERMES_TOOL_CALL_REGEX.findall(model_output)
        calls = [json.loads(m[0] if m[0] else m[1]) for m in captures]
        for call in calls:
            _ = call["name"], call["arguments"]
        return bool(calls)
    except Exception:
        return False


def _lenient_recoverable(model_output: str) -> bool:
    """只解码第一个完整 JSON 对象，容忍尾部残留。"""
    for match in HERMES_TOOL_CALL_REGEX.findall(model_output):
        capture = (match[0] if match[0] else match[1]).lstrip()
        try:
            obj, _end = _DECODER.raw_decode(capture)
        except Exception:
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return True
    return False


def classify(raw_output: str | None, server_parsed: bool) -> str:
    if server_parsed:
        return "server_parsed"
    text = raw_output or ""
    if HERMES_START_TOKEN not in text:
        return "no_envelope"
    if _hermes_strict(text):
        return "parser_loss"
    if _lenient_recoverable(text):
        return "strict_only"
    return "bad_payload"

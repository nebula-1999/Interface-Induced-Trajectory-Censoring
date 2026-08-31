"""verl 的 Qwen2.5-Coder 工具调用解析器。

**为什么需要它**：`multi_turn.format=hermes` 选的是 **verl 自己的** parser
（`verl/experimental/agent_loop/tool_parser.py`，注册名 hermes，
从 vLLM v0.9.1 的实现移植），**不是** vLLM 服务端的 parser。
因此给 vLLM 挂插件对训练链路无效——TRAIN-01 必须在 verl 这一侧注册。

Qwen2.5-Coder 未在 tool-call token 上训练，不产出 `<tool_call>`；
它产出 ```json 代码块或裸 JSON，few-shot 诱导下产出 `<tools>`。
本解析器同时接受这三种，是 §5.5 适配器覆盖面的超集——
目的是最大化检出，若仍学不会，该负结果才有意义。
"""
from __future__ import annotations

import json
import re

# <tools>{...}</tools>（hanXen few-shot 诱导的格式）
_TOOLS_TAG = re.compile(r"<tools>\s*(\{.*?\})\s*</tools>", re.S)
# ```json 代码块里的调用
_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
# 裸 JSON：含 "name" 与 "arguments" 的对象（Qwen 无诱导时的默认产物）
_BARE = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', re.S)


def _norm(obj):
    """规整成 (name, arguments_json_str)；不合规返回 None。"""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(name, str) or not name:
        return None
    if args is None:
        return None
    if isinstance(args, str):
        # 已是字符串就原样带走（verl 的 FunctionCall.arguments 是 str）
        return name, args
    try:
        return name, json.dumps(args, ensure_ascii=False)
    except Exception:
        return None


def extract_calls(text: str):
    """返回 (剩余内容, [(name, arguments_str), ...])。

    三种格式依次尝试；同一段文本只按第一种命中的格式解析，
    避免同一个调用被计两次。
    """
    if not text:
        return text, []
    for pat in (_TOOLS_TAG, _JSON_FENCE, _BARE):
        blocks = pat.findall(text)
        if not blocks:
            continue
        calls, spans = [], []
        for m in pat.finditer(text):
            raw = m.group(1) if pat is not _BARE else m.group(0)
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            n = _norm(obj)
            if n:
                calls.append(n)
                spans.append(m.span())
        if calls:
            out, last = [], 0
            for a, b in spans:
                out.append(text[last:a]); last = b
            out.append(text[last:])
            return "".join(out).strip(), calls
    return text, []


def register_into_verl():
    """把本解析器注册进 verl 的 ToolParser registry，注册名 qwen2_5_coder。

    由 sitecustomize.py 在 `verl.experimental.agent_loop.tool_parser`
    导入完成后调用，确保每个 Ray worker 都注册到。
    """
    from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
    from verl.utils.rollout_trace import rollout_trace_op

    if "qwen2_5_coder" in ToolParser._registry:
        return False

    @ToolParser.register("qwen2_5_coder")
    class Qwen25CoderToolParser(ToolParser):
        @rollout_trace_op
        async def extract_tool_calls(self, responses_ids, tools=None):
            text = self.tokenizer.decode(responses_ids, skip_special_tokens=True)
            content, calls = extract_calls(text)
            # 必须按传入 schema 校验函数名。本项目实测：Qwen-1.5B 在 mandatory
            # prompt 下 52/52、加角色消歧 few-shot 后 64/64 调用的都是**题目函数
            # 本身**而非 run_tests。不校验就会把它当成有效调用返回，下游查表
            # KeyError（verl issue #4124 同型），或更糟——被统计成一次成功发起。
            allowed = set()
            for t in (tools or []):
                fn = getattr(t, "function", None) or (t.get("function") if isinstance(t, dict) else None)
                nm = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                if nm:
                    allowed.add(nm)
            out = []
            for n, a in calls:
                if allowed and n not in allowed:
                    continue                       # 未声明的工具名，丢弃
                try:
                    parsed = json.loads(a)
                except Exception:
                    continue
                code = parsed.get("code") if isinstance(parsed, dict) else None
                if not isinstance(code, str) or not code.strip():
                    continue                       # code 必须是非空字符串
                out.append(FunctionCall(name=n, arguments=a))
            return content, out

    return True

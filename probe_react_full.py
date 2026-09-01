#!/usr/bin/env python3
"""ReAct 协议下的完整三层测试。

背景：function calling 下三层全否（0/1000 主动调用、0/256 重试、8-9/540 救回）；
换 ReAct 文本协议后**第一层跳到 100%**。但只验证了"会不会起步"，
后两层未知——若模型发一次 Action 拿到 Observation 就收手，
那 ReAct 只解决了起步，多轮 debug 依然不存在，训练照样没有有效信号。

三层：
  L1 发起率      —— 首轮是否产出 Action
  L2 持续率      —— 喂回**真实**的失败 Observation 后，是否再发 Action
  L3 救回率      —— 完整跑多轮，最终是否通过（对照 function calling 的 8-9/540）

不依赖 vLLM 的 tool parser：ReAct 是纯文本协议，自己解析 Action Input。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from sandbox import run_tests

# 2×2 设计：协议(fc/react) × 指令强度(optional/mandatory)。
# 此前 ReAct 写"必须先用 Action"、FC 写"你可以调用"，两个变量同时变了，
# 100% vs 0% 无法归因到协议本身。四格全跑才能分离。
FC_OPTIONAL = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。"""

FC_MANDATORY = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

**你必须先调用 run_tests 工具验证你的代码**，不要直接给出答案。
工具会返回测试通过情况与报错；若有失败，请修改代码并再次调用 run_tests。"""

REACT_OPTIONAL = """你是一个 Python 编程助手，可以按以下格式工作：

Thought: 你的思考
Action: run_tests
Action Input: ```python
<完整代码>
```
Observation: （由系统填写测试结果）

Thought: 我已确认代码正确
Final Answer: ```python
<最终代码>
```

你可以用 Action 提交代码验证，也可以直接给 Final Answer。"""

SYS = """你是一个 Python 编程助手，按以下格式逐步工作：

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

# 某些 tokenizer（实测 deepseek-coder）在 vLLM 下解码不完整，直接吐出
# GPT-2 字节级 BPE 的内部符号：Ġ=空格、Ċ=换行、ĉ=制表符。
# 已用 curl 确认服务端返回的原始 JSON 就带这些字符，非本地解析问题。
# 不还原的话代码写进 solution.py 就是一整行乱码，pytest 必然 collection error。
_BYTE_BPE = {"\u0120": " ", "\u010a": "\n", "\u0109": "\t"}


def normalize_bpe(text: str) -> str:
    if not text or not any(ch in text for ch in _BYTE_BPE):
        return text
    for k, v in _BYTE_BPE.items():
        text = text.replace(k, v)
    return text


_ACTION = re.compile(r"^\s*Action\s*:\s*run_tests", re.M | re.I)
_CODE = re.compile(r"Action\s*Input\s*:\s*```(?:python|py)?[ \t]*\n?(.*?)```", re.S | re.I)
_FINAL = re.compile(r"Final\s*Answer\s*:\s*```(?:python|py)?[ \t]*\n?(.*?)```", re.S | re.I)
# 裸代码块：DeepSeek 这类模型不产出 Action，直接给 ```python ...```，
# 仍应执行并计入 direct_pass@1，但不能算作 Action
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.S)
_FENCED_CODE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.S | re.I)


# 对照用的「详尽 schema」。原 FC_TOOLS 的 code 参数是裸 {"type": "string"}，
# 没有 description —— 而 ReAct 的 system prompt 给了完整模板。两个协议的
# 说明详尽程度不对等时，协议对比会混入「我把哪个写得更清楚」。
# 这一版把参数说明补到与 ReAct 模板同等详尽，用来检验 schema 混淆是否只是
# 我 schema 写得潦草造成的。
RICH_FC_TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": ("把你写的 Python 代码提交给测试运行器执行，返回测试通过情况与报错。"
                    "注意：这个工具不是你要实现的那个函数，不要把题目函数的参数传进来；"
                    "唯一的参数 code 是你写的完整源代码文本。"),
    "parameters": {"type": "object",
                   "properties": {"code": {
                       "type": "string",
                       "description": ("完整的 Python 源代码文本，包含所需的 import 和完整的函数定义。"
                                       "例如：\"def add(a, b):\\n    return a + b\"")}},
                   "required": ["code"]}}}]


FC_TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]

FC_SYS = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。"""


def parser_adapter(model: str, requested: str) -> str:
    """保留 Qwen 原口径；其他家族允许直接 fenced-code 回答。

    旧矩阵的 Qwen 结果依赖严格的 Action/Final Answer 解析，不能因为补
    DeepSeek 适配而静默改变。跨家族模型则经常直接返回 markdown 代码块，
    这仍是一个有效的首轮答案，但不能误记成 Action。
    """
    if requested != "auto":
        return requested
    return "legacy" if "qwen" in model.lower() else "cross_family"


_PLACEHOLDER = {"<>", "<完整代码>", "<最终代码>", "..."}


def _is_real_code(code: str) -> bool:
    """过滤模型只复述格式模板的情况，如 ```python<>```。

    DeepSeek-Coder 会把 system prompt 里的 ReAct 模板整段抄一遍，
    若不过滤会被当成有效代码，虚高 action 率。
    """
    c = (code or "").strip()
    return bool(c) and c not in _PLACEHOLDER


def extract_react_response(text: str, adapter: str) -> tuple[bool, str, str]:
    """返回 (是否 Action, 代码, 解析路径)。"""
    has_action = bool(_ACTION.search(text or ""))
    if has_action:
        match = _CODE.search(text or "")
        code = match.group(1) if match else ""
        if not _is_real_code(code):
            return False, "", "template_echo"     # 只是抄模板，不算发起 Action
        return True, code, "react_action"

    match = _FINAL.search(text or "")
    if match and _is_real_code(match.group(1)):
        return False, match.group(1), "final_answer"

    if adapter == "cross_family":
        blocks = [b for b in _FENCED_CODE.findall(text or "") if _is_real_code(b)]
        # DeepSeek 常输出 "Solution: <代码>" + "Explanation" + "Test Cases: <断言>"，
        # 取最后一块会拿到测试用例而非实现。优先选含函数/类定义的块。
        defs = [b for b in blocks if re.search(r"^\s*(def|class)\s", b, re.M)]
        if defs:
            return False, defs[-1], "direct_fenced_code"
        if blocks:
            # 解释中可能先引用旧代码，再给修正版；与项目其他提取器一致取最后块。
            return False, blocks[-1], "direct_fenced_code"
    return False, "", "unparsed"


def native_fc_template_status(model: str) -> tuple[str, str]:
    """静态检查本地模型的 chat template 是否真的消费 tools。

    vLLM 接受 ``tools=...`` 不代表模板会把 schema 渲染进 prompt。DeepSeek
    Coder 1.3B/6.7B 的原生模板完全不引用 tools；把这种情况记成 0% 会把
    “协议未提供给模型”误写成“模型拒绝调用”。
    """
    cfg = Path(model) / "tokenizer_config.json"
    if not cfg.is_file():
        return "unknown", f"找不到本地 tokenizer_config.json: {cfg}"
    try:
        template = json.loads(cfg.read_text(encoding="utf-8")).get("chat_template")
    except (OSError, json.JSONDecodeError) as exc:
        return "unknown", f"无法读取 chat template: {type(exc).__name__}"
    if not isinstance(template, str) or not template.strip():
        return "unsupported", "chat_template 为空"
    # Jinja 模板只要读取 tools 变量，就有机会注入 schema；完全不引用则确定不支持。
    if re.search(r"\btools\b", template):
        return "supported", "chat_template 引用了 tools"
    return "unsupported", "chat_template 未引用 tools，schema 不会进入模型 prompt"


# FC 把代码放进 JSON 字符串，\n 全部转义，比 ReAct 的裸 fenced block 更费 token。
# 两协议共用同一上限时，过小的上限会系统性偏袒 ReAct —— 这正是 26 条空代码的
# 首要嫌疑。抬到 2048 并记录 finish_reason，让截断可被观测。
MAX_TOKENS = 2048
LAST_FINISH_REASON = None


_INTENT_PATTERNS = [
    ("json_named_call", re.compile(r'"name"\s*:\s*"run_tests"')),
    ("json_arguments", re.compile(r'"arguments"\s*:\s*\{')),
    ("xml_tool_call", re.compile(r'<tool_call>|<tools>|<function')),
    ("json_code_obj", re.compile(r'\{\s*"code"\s*:\s*"')),
]


# strict 变体。vLLM 文档：tool_choice="auto" 下要施加 schema 级约束，需要
# VLLM_ENFORCE_STRICT_TOOL_CALLING=true（默认）**且**至少一个工具带 strict: true，
# 且所选 parser 支持 structural tags；否则 vLLM 从纯文本里抽取调用，
# "arguments may occasionally be malformed or violate the function's parameter schema"。
# Llama 那 22% 填错参数名正落在这句话描述的情形里 —— 本变体用于检验它。
STRICT_FC_TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "strict": True,
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string"}},
                   "required": ["code"],
                   "additionalProperties": False}}}]


def _detect_tool_intent(content: str):
    """协议无关的调用意图检测。

    只回答「模型有没有想调工具」，**不**回答「为什么没解析出来」。命中这里
    不等于格式合法：Qwen2.5-Coder 确实是输出了合法 JSON 但缺 <tool_call> 标签
    （包装层缺失），而 Qwen2.5-7B-Instruct 是标签齐全、载荷非法（载荷层写坏）。
    两者在本函数下不可区分，要分层请用 analysis/failure_layer.py。
    """
    if not content:
        return None
    for name, pat in _INTENT_PATTERNS:
        if pat.search(content):
            return name
    return None


_TCID_RE = re.compile(r"[^A-Za-z0-9]")


def _normalize_tool_call_id(msg, obs_len_seed=0):
    """把 tool_call id 规整成 9 位字母数字，并同步改写 assistant 消息里的那份。

    vLLM 文档：Mistral 的 chat template 要求 tool call id 恰好 9 位，
    而 vLLM 生成的 id（如 chatcmpl-tool-xxxx）远长于此 —— 回传时会 400。
    对其他家族，9 位字母数字同样合法，所以统一处理，不做家族分支。
    """
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return "c0abcdefg"[:9]
    raw = str(tcs[0].get("id") or "")
    cleaned = _TCID_RE.sub("", raw)
    if len(cleaned) >= 9:
        new = cleaned[-9:]
    else:
        new = (cleaned + "abcdefghi")[:9]
    tcs[0]["id"] = new          # assistant 侧与 tool 侧必须一致
    return new


def gen_fc(port, model, messages, schema="terse", seed=None, temperature=0.0):
    """function calling 一轮。返回 (是否发起调用, 代码, 原始 message)。"""
    body = {"model": model, "temperature": temperature, "max_tokens": MAX_TOKENS,
            "messages": messages,
            "tools": (STRICT_FC_TOOLS if schema == "strict"
                      else RICH_FC_TOOLS if schema == "rich" else FC_TOOLS),
            "tool_choice": "auto"}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            _ch = json.loads(r.read())["choices"][0]
            msg = _ch["message"]
            msg["_finish_reason"] = _ch.get("finish_reason")
            if msg.get("content"):
                msg["content"] = normalize_bpe(msg["content"])
    except urllib.error.HTTPError as e:
        # 只记状态码等于没记：400 的原因（模板渲染失败、schema 被拒、
        # tool_call_id 不合规）全在响应体里。
        try:
            body = e.read().decode("utf-8", "replace")[:800]
        except Exception:
            body = "(响应体读取失败)"
        return False, "", {"content": f"__ERROR__HTTP {e.code}: {body}"}
    except Exception as e:
        return False, "", {"content": f"__ERROR__{e}"}
    tcs = msg.get("tool_calls") or []
    if tcs:
        # 必须校验函数名：parser（尤其是带 few-shot 的第三方插件）只检查 name 是
        # 字符串，不检查它是否属于请求提供的 tools。模型模仿 few-shot 调用一个
        # 不存在的工具时，若不校验就会被记成有效发起，同时因为缺 code 键而落进
        # empty_code —— L1 与「schema 混淆」两个指标会被同时污染。
        _fname = (tcs[0].get("function") or {}).get("name")
        msg["_fc_tool_name"] = _fname
        if _fname != "run_tests":
            _a = tcs[0]["function"].get("arguments")
            msg["_fc_raw_args"] = _a if isinstance(_a, str) else json.dumps(
                _a, ensure_ascii=False)
            msg["_fc_arg_err"] = f"wrong_tool: {_fname!r}"
            msg["_fc_wrong_tool"] = _fname
            return False, "", msg        # 不算发起，也不执行
        args = tcs[0]["function"].get("arguments")
        # 把原始 arguments 与失败原因挂回 msg，供轨迹落盘。
        # 旧版 except -> code="" 会把「JSON 解析失败 / key 名不对 / 模型真给空」
        # 三种完全不同的情况压成同一个空串，导致无法归因。
        msg["_fc_raw_args"] = args if isinstance(args, str) else json.dumps(
            args, ensure_ascii=False)
        code = ""
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except Exception as e:
            msg["_fc_arg_err"] = f"json_decode_error: {e}"
        else:
            if not isinstance(parsed, dict):
                msg["_fc_arg_err"] = f"not_a_dict: {type(parsed).__name__}"
            else:
                msg["_fc_arg_keys"] = sorted(parsed.keys())
                code = parsed.get("code") or ""
                # 插件/模型可能把 code 给成 list/dict/数字。旧代码直接透传进
                # sandbox，一条就能让整个 n=100 崩掉。非字符串一律记录并拒执行。
                if code and not isinstance(code, str):
                    msg["_fc_arg_err"] = f"non_string_code: {type(code).__name__}"
                    msg["_fc_non_string_code"] = True
                    code = ""
                if not code:
                    alt = next((k for k in ("source", "python", "script", "solution",
                                            "program", "input", "arguments")
                                if parsed.get(k)), None)
                    if alt:
                        _v = parsed[alt]
                        if isinstance(_v, str) and _v.strip():
                            msg["_fc_arg_err"] = f"key_mismatch: used {alt!r} not 'code'"
                            code = _v
                        else:
                            msg["_fc_arg_err"] = (
                                f"non_string_code: {alt!r}={type(_v).__name__}")
                            msg["_fc_non_string_code"] = True
                    else:
                        msg["_fc_arg_err"] = "empty_code"
        return True, code, msg
    # 没发起调用时：先记录「是否其实在尝试调用、只是格式不被 parser 认」，
    # 再退回从正文里取代码。
    msg["_fc_intent"] = _detect_tool_intent(msg.get("content") or "")
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", msg.get("content") or "", re.S)
    return False, (m.group(1) if m else ""), msg


def gen(port, model, messages, max_tokens=MAX_TOKENS, seed=None, temperature=0.0):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": messages, "stop": ["Observation:"]}   # 别让它自己编造 Observation
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            _ch = json.loads(r.read())["choices"][0]
            global LAST_FINISH_REASON
            LAST_FINISH_REASON = _ch.get("finish_reason")
            return normalize_bpe(_ch["message"]["content"] or "")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:800]
        except Exception:
            body = "(响应体读取失败)"
        return f"__ERROR__HTTP {e.code}: {body}"
    except Exception as e:
        return f"__ERROR__{e}"


def supports_native_fc(model_path: str) -> bool:
    """chat template 是否真的注入 tools。

    关键区分：Qwen 的 0% 是"注入了工具但模型不用"，DeepSeek 是"压根没注入"。
    后者应记 N/A —— 把两者都写成 0% 会得出完全错误的结论。
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        probe = [{"role": "user", "content": "hi"}]
        t = [{"type": "function", "function": {"name": "run_tests_probe_marker",
              "description": "x", "parameters": {"type": "object", "properties": {}}}}]
        a = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        b = tok.apply_chat_template(probe, tools=t, tokenize=False, add_generation_prompt=True)
        return "run_tests_probe_marker" in b and len(b) > len(a)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--strength", choices=["optional", "mandatory"],
                    default="mandatory", help="指令强度：可选 vs 必须")
    ap.add_argument("--protocol", choices=["react", "fc"], default="react",
                    help="react=ReAct 文本协议；fc=OpenAI function calling。"
                         "两者用同一批题、同一沙箱执行器，唯一差别是交互协议。")
    ap.add_argument("--sys-file", default=None,
                    help="用文件内容覆盖 system prompt（prompt 消融用）")
    ap.add_argument("--fc-schema", choices=["terse", "rich", "strict"],
                    default="terse",
                    help="FC 工具 schema 详尽程度（协议对比的公平性对照）")
    ap.add_argument("--seed", type=int, default=None,
                    help="传给 vLLM 的采样 seed；不传则不发送该字段（行为与旧版一致）")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="采样温度。默认 0.0 与旧版一致；>0 用于方差估计")
    ap.add_argument("--out", default="", help="逐题 JSONL 输出路径")
    ap.add_argument("--only-ids", default="",
                    help="只跑这些 clean_index（逗号分隔），用于精确补跑失败题")
    ap.add_argument("--parser-adapter", choices=["auto", "legacy", "cross_family"],
                    default="auto", help="auto 保留 Qwen 旧口径，其他家族接受普通代码块")
    ap.add_argument("--allow-unsupported-fc", action="store_true",
                    help="仅诊断用：即使 chat template 不注入 tools 也继续 FC；结果不可作 FC 结论")
    a = ap.parse_args()

    adapter = parser_adapter(a.model, a.parser_adapter)
    if a.protocol == "fc":
        fc_status, fc_reason = native_fc_template_status(a.model)
        print(f"FC 模板检查: {fc_status} — {fc_reason}")
        if fc_status == "unsupported" and not a.allow_unsupported_fc:
            # 删除同名旧输出，防止队列把上一次的 100 行误当作本次成功。
            if a.out:
                Path(a.out).unlink(missing_ok=True)
            print("状态: unsupported_native_fc_template")
            print("本组跳过：这是协议不可用（N/A），不是工具调用率 0%。")
            return 3

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
    if a.only_ids:
        want = {int(x) for x in a.only_ids.split(",") if x.strip()}
        idxs = [i for i in clean if i in want]
    else:
        idxs = clean[:a.n]
    items = [(kc[i]["question"], kc[i]["test"], i) for i in idxs]

    l1 = l2_denom = l2 = rescued = n_err = n_direct = n_unparsed = 0
    native_fc = supports_native_fc(a.model)
    if a.protocol == "fc" and not native_fc:
        print(f"\n模型: {a.model}\n⚠️  native_fc_supported = false "
              f"(template_does_not_inject_tools) → 本组记 N/A，不是 0%")
    fout = open(a.out, "w", encoding="utf-8") if a.out else None
    passed_final = passed_turn1 = 0
    turn_hist = {}
    parse_modes: Counter[str] = Counter()
    first_modes: Counter[str] = Counter()   # 逐题首轮口径
    intent_forms: Counter[str] = Counter()
    n_intent = 0

    for qi, (q, test, ds_id) in enumerate(items):
        is_fc = (a.protocol == "fc")
        sysmsg = ({("fc", "optional"): FC_OPTIONAL,
                   ("fc", "mandatory"): FC_MANDATORY,
                   ("react", "optional"): REACT_OPTIONAL,
                   ("react", "mandatory"): SYS})[(a.protocol, a.strength)]
        if a.sys_file:
            sysmsg = open(a.sys_file, encoding="utf-8").read().strip()
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}]
        first_ok = final_ok = False
        turns = 0
        pending = None
        rec = {"i": qi, "clean_index": ds_id, "protocol": a.protocol,
               "adapter": adapter, "sys_file": a.sys_file,
               "fc_schema": a.fc_schema,
               "seed": a.seed,
               "temperature": a.temperature,
               "max_tokens": MAX_TOKENS,
               "strength": a.strength, "model": a.model, "turns": []}
        for t in range(1, a.max_turns + 1):
            if is_fc:
                if pending is not None:
                    has_action, code, msg = pending
                    pending = None
                else:
                    has_action, code, msg = gen_fc(a.port, a.model, msgs, a.fc_schema, a.seed, a.temperature)
                out = msg.get("content") or ""
                if out.startswith("__ERROR__"):
                    n_err += 1
                    rec["turns"].append({"t": t, "action": False, "code": "",
                                         "raw_output": out[:4000], "passed": None,
                                         "parse_mode": "request_error"})
                    parse_modes["request_error"] += 1
                    break
                parse_mode = ("unknown_tool" if msg.get("_fc_wrong_tool")
                              else "fc_tool_call" if has_action
                              else "fc_direct_fenced_code" if code else "unparsed")
                fc_dbg = {k: msg[k] for k in
                          ("_fc_raw_args", "_fc_arg_err", "_fc_arg_keys",
                           "_finish_reason", "_fc_intent",
                           "_fc_tool_name", "_fc_wrong_tool",
                           "_fc_non_string_code")
                          if k in msg}
            else:
                out = pending if pending is not None else gen(a.port, a.model, msgs, seed=a.seed, temperature=a.temperature)
                pending = None
                if out.startswith("__ERROR__"):
                    n_err += 1
                    rec["turns"].append({"t": t, "action": False, "code": "",
                                         "raw_output": out[:4000], "passed": None,
                                         "parse_mode": "request_error"})
                    parse_modes["request_error"] += 1
                    break
                has_action, code, parse_mode = extract_react_response(out, adapter)
            parse_modes[parse_mode] += 1
            if t == 1:
                l1 += has_action
                first_modes[parse_mode] += 1
                if is_fc and not has_action:
                    _iv = msg.get("_fc_intent")
                    if _iv:
                        n_intent += 1
                        intent_forms[_iv] += 1
            if not code:
                _turn_rec = {"t": t, "action": has_action, "code": "",
                                     "raw_output": out[:4000], "passed": None,
                                     "parse_mode": parse_mode}
                if a.protocol == "fc":
                    _turn_rec.update(fc_dbg)
                else:
                    _turn_rec["_finish_reason"] = LAST_FINISH_REASON
                rec["turns"].append(_turn_rec)
                break
            turns = t
            res = run_tests(code, test, mode="pytest")
            final_ok = res.all_passed
            if t == 1:
                first_ok = res.all_passed
            _ok_rec = {"t": t, "action": has_action, "code": code,
                       "raw_output": out[:4000],
                       "parse_mode": parse_mode,
                       "passed": res.passed, "total": res.total,
                       "all_passed": res.all_passed,
                       "obs": res.stderr[-400:]}
            if a.protocol == "fc":
                _ok_rec.update(fc_dbg)
            else:
                _ok_rec["_finish_reason"] = LAST_FINISH_REASON
            rec["turns"].append(_ok_rec)
            if res.all_passed or not has_action:
                break
            if t == 1:
                l2_denom += 1
            obs = f"{res.passed}/{res.total} 个测试通过。\n{res.stderr[-500:]}"
            if is_fc:
                _tcid = _normalize_tool_call_id(msg)
                msgs += [msg, {"role": "tool", "tool_call_id": _tcid,
                               "name": "run_tests", "content": obs}]
            else:
                msgs += [{"role": "assistant", "content": out},
                         {"role": "user", "content": f"Observation: {obs}"}]
            if t >= a.max_turns:
                break              # 最后一轮不必再生成，否则白烧一次推理
            nxt = gen_fc(a.port, a.model, msgs, a.fc_schema, a.seed, a.temperature) if is_fc else gen(a.port, a.model, msgs, seed=a.seed, temperature=a.temperature)
            if t == 1 and (nxt[0] if is_fc else bool(_ACTION.search(nxt))):
                l2 += 1
            pending = nxt          # 下一轮直接消费，不重复生成
        turn_hist[turns] = turn_hist.get(turns, 0) + 1
        passed_turn1 += first_ok
        passed_final += final_ok
        if final_ok and not first_ok:
            rescued += 1
        rec.update(first_ok=first_ok, final_ok=final_ok, n_turns=turns)
        if fout:                      # 逐题落盘并 flush：中途关机也不会全丢
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    if fout:
        fout.close()

    n = len(items)
    if n == 0:
        print("没有选中任何题，检查 --n / --only-ids。")
        return 2
    print(f"\n模型: {a.model}   协议={a.protocol}  强度={a.strength}  n={n}")
    print(f"解析适配: {adapter}   路径统计: {dict(sorted(parse_modes.items()))}")
    print(f"请求错误: {n_err}" + ("   ⚠️ 非零，本组结果不可用" if n_err else ""))
    print(f"native_fc_supported   : {native_fc}")
    print(f"L1 严格 Action 发起率  : {l1}/{n} = {l1/n:.0%}"
          + ("   ← 协议不受支持，应记 N/A" if a.protocol == "fc" and not native_fc else ""))
    # FC 臂的解析标签带 fc_ 前缀（fc_direct_fenced_code），ReAct 臂不带。
    # 只取无前缀的键会让「直接给代码率」对所有 FC 臂恒为 0% —— 轨迹里数据是全的，
    # 丢的只是这一行汇总。
    n_direct = (first_modes.get("direct_fenced_code", 0)
                + first_modes.get("fc_direct_fenced_code", 0))
    n_unparsed = first_modes.get("unparsed", 0) + first_modes.get("template_echo", 0)
    n_wrongtool = first_modes.get("unknown_tool", 0)
    n_final = first_modes.get("final_answer", 0)
    print(f"   直接给代码率        : {n_direct}/{n} = {n_direct/n:.0%}")
    print(f"   Final Answer 率     : {n_final}/{n} = {n_final/n:.0%}")
    print(f"   无法解析率          : {n_unparsed}/{n} = {n_unparsed/n:.0%}")
    if a.protocol == "fc":
        print(f"   ★ 服务端解析出调用   : {l1}/{n} = {l1/n:.0%}")
        # 原来这里写「格式完好但 parser 不认 → 静默低估」，是一句从未校验过的断言：
        # _detect_tool_intent 只看有没有调用意图，不校验 JSON 合法性、也不区分
        # 有没有 <tool_call> 包装层。实测 Qwen2.5-7B-Instruct 的 34 条里格式完好的
        # 是 0 条，全是模型自己把 JSON 写坏了。成因分层交给
        # analysis/failure_layer.py（它逐行重放 hermes），这里只作中性陈述。
        print(f"   ★ 有调用意图但服务端未解析 : {n_intent}/{n} = {n_intent/n:.0%}"
              f"   （成因分层见 analysis/failure_layer.py，勿直接读作 parser 损失）")
        print(f"     意图形式分布       : {dict(intent_forms)}")
        print(f"   ★ 调用了不存在的工具 : {n_wrongtool}/{n} = {n_wrongtool/n:.0%}"
              f"   （已排除，不计入 L1，不执行）")
    print(f"L2 继续发 Action       : {l2}/{l2_denom if l2_denom else 1} "
          f"= {l2/l2_denom if l2_denom else 0:.0%}"
          f"   （分母=首轮**发起 Action 且执行失败**数，非所有首轮失败数）")
    print(f"L3 首轮通过 / 最终通过 : {passed_turn1}/{n} = {passed_turn1/n:.0%}"
          f"  →  {passed_final}/{n} = {passed_final/n:.0%}")
    n_fail1 = n - passed_turn1
    print(f"   绝对救回率           : {rescued}/{n} = {rescued/n:.1%}")
    print(f"   **条件救回率**       : {rescued}/{n_fail1 if n_fail1 else 1} = "
          f"{rescued/n_fail1 if n_fail1 else 0:.1%}   （分母=首轮未通过数）")
    print(f"   ※ 绝对值跨规模不可比：大模型首轮通过率高，剩下的失败题更难")
    print(f"   轮次分布            : {dict(sorted(turn_hist.items()))}"
          f"   （0 = 未产出任何可执行代码）")
    print("\n判读：L2 高 → ReAct 真正解决了多轮，可以据此重跑训练；")
    print("      L2 低 → ReAct 只解决起步，多轮仍不存在，别改训练架构。")
    # 请求错误非零时以非零码退出：调用方不能再静默引用一组被自己判定为
    # 不可用的数据（v2 里 Mistral FC 臂就是这样被引用的）。
    EXIT_DIRTY = 2 if n_err else 0
    if EXIT_DIRTY:
        print(f"\n⚠️ 退出码 {EXIT_DIRTY}：n_err={n_err}，本组数据不可用于比较。")
    return EXIT_DIRTY


if __name__ == "__main__":
    sys.exit(main())

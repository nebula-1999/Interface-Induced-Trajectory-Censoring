"""在 verl **自己的** rollout 路径上记录 emitted → accepted → executed。

要回答的问题：在真正的训练栈里（不是我们的评测 harness），Qwen2.5-Coder-7B
在 hermes 配置下**发出过合规的 run_tests 调用却被丢弃**吗？

  emitted>0 且 accepted=0 且 executed=0 → 因果链闭合，§5.7 可从「分布性缺失」升级
  emitted=0                             → 与 1.5B 同类，是策略问题不是接口问题
  钩子未生效                              → **本轮作废**，不得读作上面任何一种

第三种是本文件一半篇幅的理由：三个计数天然都可能是 0，「没装上」和「真的是 0」
在产物上完全一样。所以每个 patch 点都记录实际触发次数，收尾时任一为 0 就落
P3_INVALID 并在 stderr 喊出来。

**打在哪**（由 p3/discover_verl.py 在机器上实测得出，不是猜的）：
  · verl.experimental.agent_loop.tool_parser.ToolParser._registry 里**每一个**
    子类的 extract_tool_calls —— 它的入参是 responses_ids、返回 (content, calls)，
    所以同一个点既拿得到原始发出文本，也拿得到被接受的调用。
  · verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop._call_tool —— 执行边界。
  · code_tool.CodeTool.execute —— 我们自己工具的执行确认。

注意 verl 的 HermesToolParser 正则是 `<tool_call>(.*?)</tool_call>`，**没有**
vLLM 0.27.1 那条 `|<tool_call>(.*)` 的收尾兜底分支，即 verl 侧比服务端更严。
两者的判定都记下来，差异本身是结果的一部分。
"""
from __future__ import annotations

import functools
import inspect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

OUT = Path(os.environ.get("P3_OUT", "/root/autodl-tmp/runs/p3_rollout_probe"))
OUT.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()

FIRED: dict[str, int] = {"extract_tool_calls": 0, "call_tool": 0, "code_tool_execute": 0}

# 判据与 analysis/failure_layer.py 同源：照抄 vLLM 0.27.1 的正则（含兜底分支）
VLLM_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
VERL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TIGHT_RE = re.compile(
    r'"name"\s*:\s*"run_tests".{0,200}?"arguments"\s*:\s*\{.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}', re.S)
REAL_RE = re.compile(r"\\n|def |class |return |import |lambda ")


def _accepts(rx, text: str) -> bool:
    if "<tool_call>" not in text:
        return False
    try:
        caps = [m[0] if m[0] else (m[1] if len(m) > 1 else "") for m in rx.findall(text)]
        calls = [json.loads(c) for c in caps if c]
        return bool(calls) and all("name" in c and "arguments" in c for c in calls)
    except Exception:
        return False


def _write(kind: str, **f) -> None:
    with _LOCK, (OUT / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.time(), "pid": os.getpid(), "kind": kind, **f},
                            ensure_ascii=False) + "\n")


def install() -> bool:
    """装解析器钩子。返回是否至少装上一个——装不上必须让调用方知道。

    与 install_agentloop 分开：本函数在 tool_parser 导入完成时调用，而那一刻
    tool_agent_loop 往往**正在导入中**（它自己 import 了 tool_parser），
    此时取 ToolAgentLoop 会拿到 partially initialized module。
    """
    ok = False
    try:
        from verl.experimental.agent_loop import tool_parser as tp
    except Exception as e:
        print(f"[p3] ✗ import tool_parser 失败: {e!r}", file=sys.stderr, flush=True)
        return False

    for name, cls in list(getattr(tp.ToolParser, "_registry", {}).items()):
        fn = cls.__dict__.get("extract_tool_calls")
        if fn is None or getattr(fn, "_p3", False):
            continue

        @functools.wraps(fn)
        async def wrapper(self, responses_ids, tools=None, _fn=fn, _name=name):
            try:
                text = self.tokenizer.decode(responses_ids, skip_special_tokens=False)
            except Exception:
                text = ""
            res = await _fn(self, responses_ids, tools)
            content, calls = (res if isinstance(res, tuple) and len(res) == 2 else ("", []))
            FIRED["extract_tool_calls"] += 1
            _write("extract",
                   parser=_name,
                   n=FIRED["extract_tool_calls"],
                   accepted=len(calls or []),
                   accepted_names=[getattr(c, "name", None) for c in (calls or [])],
                   has_envelope="<tool_call>" in text,
                   names_run_tests='"run_tests"' in text,
                   vllm_would_accept=_accepts(VLLM_RE, text),
                   verl_would_accept=_accepts(VERL_RE, text),
                   tight=bool((lambda m: m and REAL_RE.search(m.group(1)))(TIGHT_RE.search(text))),
                   text=text[:4000])
            return res

        wrapper._p3 = True
        setattr(cls, "extract_tool_calls", wrapper)
        ok = True
        print(f"[p3] ✓ 已挂 {cls.__name__}.extract_tool_calls", file=sys.stderr, flush=True)

    try:
        import code_tool
        orig_e = code_tool.CodeTool.__dict__.get("execute")
        if orig_e is not None and not getattr(orig_e, "_p3", False):
            @functools.wraps(orig_e)
            async def execute(self, instance_id, parameters, *a, _o=orig_e, **kw):
                FIRED["code_tool_execute"] += 1
                _write("execute", n=FIRED["code_tool_execute"],
                       has_code=bool((parameters or {}).get("code")))
                return await _o(self, instance_id, parameters, *a, **kw)
            execute._p3 = True
            code_tool.CodeTool.execute = execute
            print("[p3] ✓ 已挂 CodeTool.execute", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[p3] ! CodeTool.execute 未挂上: {e!r}", file=sys.stderr, flush=True)

    _write("install", ok=ok, fired=dict(FIRED))
    return ok


def install_agentloop() -> bool:
    """装 ToolAgentLoop._call_tool。必须等 tool_agent_loop 自己导入完成后再调。

    这个信号不是冗余：若 accepted>0 而工具名是错的，_call_tool 会触发而
    CodeTool.execute 不会——两者之差正好区分「解析成功但叫错工具」与
    「根本没解析出来」，而这正是 1.5B 那条结论的关键分歧点。
    """
    try:
        from verl.experimental.agent_loop import tool_agent_loop as tal
        cls = getattr(tal, "ToolAgentLoop", None)
        if cls is None:
            return False
        orig = cls.__dict__.get("_call_tool")
        if orig is None or getattr(orig, "_p3", False):
            return False

        @functools.wraps(orig)
        async def call_tool(self, *a, _o=orig, **kw):
            FIRED["call_tool"] += 1
            _write("call_tool", n=FIRED["call_tool"], args=repr(a)[:400])
            return await _o(self, *a, **kw)

        call_tool._p3 = True
        cls._call_tool = call_tool
        print("[p3] ✓ 已挂 ToolAgentLoop._call_tool", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[p3] ! _call_tool 未挂上: {e!r}", file=sys.stderr, flush=True)
        return False

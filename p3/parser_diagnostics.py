"""JSON/envelope diagnostics, not a substitute for invoking native parsers."""
import json
import re

DIAGNOSTIC_VERSION = 2
VLLM_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
VERL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def accepts(rx, text):
    try:
        captures = [m if isinstance(m, str) else next((s for s in m if s), "")
                    for m in rx.findall(text)]
        calls = [json.loads(c) for c in captures]
        return bool(calls) and all(isinstance(c, dict) and
                                   "name" in c and "arguments" in c for c in calls)
    except (ValueError, TypeError):
        return False

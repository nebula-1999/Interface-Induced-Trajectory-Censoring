#!/usr/bin/env python3
"""Locate *which layer* a would-be tool call was lost at.

`analysis/intent.py` answers **was there an attempt**; this module answers **where it
died**. The two are orthogonal and neither replaces the other. It exists because the probe
originally labelled every strong-intent item "well-formed but the parser refused it -> silent
undercount", and that sentence had never been checked against JSON validity. Measured on
Qwen2.5-7B-Instruct's 34 such items, the number that were actually well-formed was **zero**.

The criterion is not a re-implementation. It **replays vLLM 0.27.1's hermes parser line for
line** (the non-streaming branch of `vllm/tool_parsers/hermes_tool_parser.py`), so that "the
parser should have succeeded and did not" is falsifiable rather than asserted.

Four tiers:

``server_parsed``  The server parsed it. Not a loss.
``no_envelope``    No ``<tool_call>`` tag in the output. hermes not seeing a call here is
                   **correct behaviour**, not a bug. Every Qwen2.5-Coder size lands here:
                   the payload is valid JSON, what is missing is the envelope.
``bad_payload``    Envelope present, but the payload is not valid JSON and lenient decoding
                   cannot recover it either. The usual cause is Python source dropped into a
                   JSON string unescaped: docstring ``\"\"\"``, raw newlines, bad escapes.
``strict_only``    The payload itself is valid; ``json.loads`` merely refuses the trailing
                   text left in the capture group. ``raw_decode`` recovers it --
                   **this tier, and only this one, is caused by parser strictness.**
``parser_loss``    Replaying hermes verbatim succeeds where the server did not. Only this
                   tier deserves to be called a silent undercount.

Usage::

    from analysis.failure_layer import classify, TIERS
    tier = classify(raw_output, server_parsed=turn["parse_mode"] == "fc_tool_call")
"""
from __future__ import annotations

import json
import re

# Copied verbatim from vLLM 0.27.1 hermes_tool_parser.py, lines 38-40.
# Editing this changes the criterion: any change must bump the cited vLLM version too.
HERMES_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)
HERMES_START_TOKEN = "<tool_call>"
_DECODER = json.JSONDecoder()

TIERS = ("server_parsed", "no_envelope", "parser_loss", "strict_only", "bad_payload")

TIER_LABEL = {
    "server_parsed": "parsed by the server",
    "no_envelope": "no <tool_call> envelope (parser cannot see it -- correct behaviour)",
    "parser_loss": "* genuine parser loss (replay should have succeeded)",
    "strict_only": "payload valid, rejected by json.loads only for trailing text",
    "bad_payload": "payload itself malformed (the model wrote it wrong)",
}


def _hermes_strict(model_output: str) -> bool:
    """What hermes does today: hand the whole capture to json.loads; any failure fails all."""
    try:
        captures = HERMES_TOOL_CALL_REGEX.findall(model_output)
        calls = [json.loads(m[0] if m[0] else m[1]) for m in captures]
        for call in calls:
            _ = call["name"], call["arguments"]
        return bool(calls)
    except Exception:
        return False


def _lenient_recoverable(model_output: str) -> bool:
    """Decode only the first complete JSON object, tolerating trailing text."""
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

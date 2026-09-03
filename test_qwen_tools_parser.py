from __future__ import annotations

import json

from qwen_tools_parser import extract_calls


def _valid(text: str):
    _, calls = extract_calls(text)
    return [(name, json.loads(arguments)) for name, arguments in calls]


def test_bare_json_call():
    text = '{"name":"run_tests","arguments":{"code":"def f():\\n    return 1"}}'
    assert _valid(text) == [("run_tests", {"code": "def f():\n    return 1"})]


def test_json_fence_and_tools_tag():
    call = '{"name":"run_tests","arguments":{"code":"return 1"}}'
    assert _valid(f"```json\n{call}\n```")[0][0] == "run_tests"
    assert _valid(f"<tools>{call}</tools>")[0][0] == "run_tests"


def test_schema_echo_is_not_a_call():
    text = ('{"type":"function","function":{"name":"run_tests",'
            '"parameters":{"type":"object"}}}')
    assert _valid(text) == []


def test_malformed_json_is_not_accepted():
    assert _valid('{"name":"run_tests","arguments":{"code":"unterminated}}') == []

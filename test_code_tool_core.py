"""code_tool 纯逻辑的测试。"""

from __future__ import annotations

from code_tool_core import extract_code, format_observation, suspicious


def test_extract_last_block():
    """模型常先复述旧代码再给新版——必须取最后一个。"""
    t = "先看原来的:\n```python\ndef f(): return 0\n```\n改成:\n```python\ndef f(): return 1\n```"
    assert extract_code(t) == "def f(): return 1"


def test_extract_bare_fence():
    assert extract_code("```\ndef f(): pass\n```") == "def f(): pass"


def test_extract_without_fence():
    """小模型经常忘记打围栏，不能直接判错。"""
    assert extract_code("def f(): return 1") == "def f(): return 1"


def test_extract_empty():
    assert extract_code("") == ""


def test_suspicious_detects_exit():
    assert "sys_exit" in suspicious("import sys\nsys.exit(0)")
    assert "sys_exit" in suspicious("os._exit(0)")


def test_suspicious_detects_pytest_skip():
    assert "pytest_skip" in suspicious("import pytest\npytest.skip('x')")


def test_suspicious_detects_test_file_access():
    assert "touch_test_file" in suspicious("open('test_solution.py','w')")


def test_suspicious_clean_code():
    assert suspicious("def f(x):\n    return x + 1") == []


def test_observation_all_pass():
    assert format_observation(5, 5, "ok", "") == "全部 5 个测试通过。"


def test_observation_partial():
    obs = format_observation(3, 5, "ok", "E   AssertionError: assert 1 == 2")
    assert "3/5" in obs and "2 个失败" in obs
    assert "AssertionError" in obs


def test_observation_timeout():
    assert "超时" in format_observation(0, 0, "timeout", "")


def test_observation_no_tests():
    assert "没有运行起来" in format_observation(0, 0, "no_tests", "")


def test_observation_is_bounded():
    """报错再长也不能撑爆上下文。"""
    huge = "\n".join(f"E   AssertionError: line {i}" for i in range(5000))
    assert len(format_observation(0, 9, "ok", huge)) <= 1200

"""沙箱的边界测试。

照 probes/ 那套做法：先把环境本身测穿，再拿它去测模型。
沙箱漏一个边界，训练时就是一个静默的脏奖励，事后极难查。
"""

from __future__ import annotations

import time

from sandbox import Result, run_many, run_tests, safe_workers

TEST_TWO = """
from solution import f

def test_a():
    assert f(1) == 1

def test_b():
    assert f(2) == 2
"""


# --- 正常路径 -------------------------------------------------------------

def test_all_pass():
    r = run_tests("def f(x): return x", TEST_TWO)
    assert r.status == "ok"
    assert (r.passed, r.failed) == (2, 0)
    assert r.pass_ratio == 1.0
    assert r.all_passed


def test_partial_pass():
    """partial credit 的核心用例：一半对一半错要拿到 0.5，不是 0。"""
    r = run_tests("def f(x): return 1", TEST_TWO)
    assert (r.passed, r.failed) == (1, 1)
    assert r.pass_ratio == 0.5
    assert not r.all_passed        # 训练给 0.5，评测判不通过


def test_bare_test_style_without_import():
    """KodCode 的第二种 test 风格：不写 import，直接用函数名。

    实测占样本的 16.5%。不分流的话这一整类全是 NameError，
    会被误当成数据质量问题剔掉。
    """
    bare = "def test_a():\n    assert f(1) == 1\n\ndef test_b():\n    assert f(2) == 2\n"
    r = run_tests("def f(x): return x", bare)
    assert r.all_passed, r.stderr[-300:]
    assert (r.passed, r.failed) == (2, 0)


def test_bare_style_partial_credit_still_works():
    bare = "def test_a():\n    assert f(1) == 1\n\ndef test_b():\n    assert f(2) == 2\n"
    r = run_tests("def f(x): return 1", bare)
    assert r.pass_ratio == 0.5


def test_model_output_cannot_spoof_pytest_summary():
    """Only pytest's final summary line may determine the reward."""
    test = '''
from solution import f

def test_a():
    print("999 passed")
    assert f(1) == 1

def test_b():
    assert f(2) == 2
'''
    r = run_tests("def f(x): return 1", test)
    assert (r.passed, r.failed, r.total) == (1, 1, 2)
    assert r.pass_ratio == 0.5


def test_skipped_and_xfailed_are_in_reward_denominator():
    test = '''
import pytest
from solution import f

def test_pass():
    assert f(1) == 1

@pytest.mark.skip(reason="not implemented")
def test_skip():
    assert f(2) == 2

@pytest.mark.xfail(reason="known gap")
def test_xfail():
    assert f(3) == 3
'''
    r = run_tests("def f(x): return 1", test)
    assert (r.passed, r.skipped, r.xfailed, r.total) == (1, 1, 1, 3)
    assert r.pass_ratio == 1 / 3
    assert not r.all_passed


# --- 坏代码 ---------------------------------------------------------------

def test_syntax_error():
    r = run_tests("def f(x) return x", TEST_TWO)
    assert not r.all_passed
    assert r.pass_ratio == 0.0
    assert "SyntaxError" in r.stderr or r.status in ("no_tests", "crash")


def test_import_missing_module():
    r = run_tests("import nonexistent_pkg_xyz\ndef f(x): return x", TEST_TWO)
    assert not r.all_passed
    assert r.pass_ratio == 0.0


def test_name_error_at_runtime():
    r = run_tests("def f(x): return undefined_name", TEST_TWO)
    assert (r.passed, r.failed) == (0, 2)
    assert r.pass_ratio == 0.0


# --- 资源边界 -------------------------------------------------------------

def test_infinite_loop_times_out():
    t0 = time.time()
    r = run_tests("def f(x):\n    while True: pass\n", TEST_TWO, timeout=3.0)
    el = time.time() - t0
    assert r.status == "timeout"
    assert not r.all_passed
    assert el < 12, f"超时后没有及时收尾，耗时 {el:.1f}s"


def test_memory_bomb():
    r = run_tests("def f(x):\n    return [0] * (10**9)\n", TEST_TWO,
                  timeout=8.0, mem_mb=256)
    assert not r.all_passed
    assert r.status in ("ok", "timeout")     # MemoryError 记为普通失败即可


def test_output_flood():
    """无限输出不能把父进程拖垮，也不能把返回值撑爆。"""
    sol = "def f(x):\n    for _ in range(10**7): print('x' * 200)\n    return x\n"
    r = run_tests(sol, TEST_TWO, timeout=5.0)
    assert not r.all_passed
    assert len(r.stderr) <= 1600 + 64


def test_child_process_is_killed():
    """派生的子进程必须跟着死。

    只 kill 父进程的话它会留在后台吃 CPU——112 核上攒几百个就废了。
    """
    sol = ("import subprocess, sys\n"
           "def f(x):\n"
           "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
           "    while True: pass\n")
    t0 = time.time()
    r = run_tests(sol, TEST_TWO, timeout=3.0)
    assert r.status == "timeout"
    assert time.time() - t0 < 12


def test_cannot_write_huge_file():
    sol = ("def f(x):\n"
           "    with open('big.bin','wb') as g:\n"
           "        for _ in range(10000): g.write(b'0'*(1<<20))\n"
           "    return x\n")
    r = run_tests(sol, TEST_TWO, timeout=8.0)
    assert not r.all_passed


# --- 隔离性 ---------------------------------------------------------------

def test_runs_are_isolated():
    """一次运行写的文件不能被下一次看到。"""
    w = ("def f(x):\n"
         "    open('leak.txt','w').write('1')\n"
         "    return x\n")
    run_tests(w, TEST_TWO)
    rd = ("import os\n"
          "def f(x):\n"
          "    assert not os.path.exists('leak.txt')\n"
          "    return x\n")
    r = run_tests(rd, TEST_TWO)
    assert r.all_passed


# --- 并行 -----------------------------------------------------------------

def test_safe_workers_respects_memory():
    w = safe_workers()
    assert w >= 1
    assert w <= (__import__("os").cpu_count() or 1)


def test_run_many():
    items = [("def f(x): return x", TEST_TWO),
             ("def f(x): return 1", TEST_TWO),
             ("def f(x) return x", TEST_TWO)]
    rs = run_many(items, workers=2)
    assert len(rs) == 3
    assert [round(r.pass_ratio, 2) for r in rs] == [1.0, 0.5, 0.0]


def test_result_no_tests_is_zero():
    r = Result(status="no_tests")
    assert r.pass_ratio == 0.0
    assert not r.all_passed

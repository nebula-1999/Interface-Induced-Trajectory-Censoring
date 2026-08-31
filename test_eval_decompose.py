"""分解评测的不变量测试。

主图三条曲线全部由这里算出，逻辑错了结论就错了，而且是那种"数字看着合理、
其实是错的"的错法——所以每条不变量都要钉死。

用假 policy 和假沙箱，不碰模型也不跑真代码。
"""

from __future__ import annotations

import eval_decompose as ed
from eval_decompose import Task, Turn, summarize


class ScriptedPolicy:
    """按预定脚本回话：outs[i] 是第 i 轮所有题的输出。"""

    def __init__(self, per_turn: list[str]):
        self.per_turn = per_turn
        self.calls = 0

    def __call__(self, convs, max_tokens=1024):
        out = self.per_turn[min(self.calls, len(self.per_turn) - 1)]
        self.calls += 1
        return [out] * len(convs)


def _fake_batch(results):
    """把 _run_batch 换掉：results 是每轮的 all_passed 值。"""
    state = {"i": 0}

    def f(recs, codes, workers):
        ok = results[min(state["i"], len(results) - 1)]
        state["i"] += 1
        return [type("R", (), {"passed": 1 if ok else 0, "total": 1,
                               "all_passed": ok, "status": "ok",
                               "stderr": ""})() for _ in recs]
    return f


REC = ("HumanEval+", "HumanEval/0",
       {"entry_point": "f", "prompt": "def f(x):\n    ...\n",
        "canonical_solution": "    return x", "test": "def check(c): pass"})


def test_pass_on_first_turn(monkeypatch):
    monkeypatch.setattr(ed, "_run_batch", _fake_batch([True]))
    ts = ed.evaluate(ScriptedPolicy(["```python\ndef f(x): return x\n```"]),
                     [REC], workers=1)
    t = ts[0]
    assert t.turn1_passed and t.final_passed
    assert t.n_turns == 1, "第一轮就过了不该再有第二轮"


def test_fixed_on_second_turn(monkeypatch):
    """turn-1 失败、turn-2 修好——这正是"真 debug"要计入的那一类。"""
    monkeypatch.setattr(ed, "_run_batch", _fake_batch([False, True]))
    ts = ed.evaluate(ScriptedPolicy(["```python\nbad\n```",
                                     "```python\ndef f(x): return x\n```"]),
                     [REC], workers=1)
    t = ts[0]
    assert not t.turn1_passed
    assert t.final_passed
    assert t.n_turns == 2


def test_never_fixed_stops_at_max_turns(monkeypatch):
    monkeypatch.setattr(ed, "_run_batch", _fake_batch([False]))
    ts = ed.evaluate(ScriptedPolicy(["```python\nbad\n```"]), [REC],
                     max_turns=3, workers=1)
    t = ts[0]
    assert not t.final_passed
    assert t.n_turns == 3, "失败也不能超过 max_turns 轮"


def test_summarize_separates_turn1_from_final():
    """核心不变量：final 包含 turn-1，两者之差才是多轮带来的增量。"""
    def mk(t1, fin, n):
        turns = [Turn(i + 1, 1 if (i == 0 and t1) or (i == n - 1 and fin) else 0,
                      1, (i == 0 and t1) or (i == n - 1 and fin), "ok")
                 for i in range(n)]
        return Task("t", "HumanEval+", turns)

    multi = [mk(True, True, 1), mk(False, True, 2), mk(False, False, 3)]
    s = summarize(multi, [])
    assert abs(s["turn1_pass"] - 1 / 3) < 1e-9
    assert abs(s["final_pass"] - 2 / 3) < 1e-9
    assert s["final_pass"] >= s["turn1_pass"], "final 必然 >= turn-1"


def test_repair_rate_by_kind():
    rep = [Task("a", "HumanEval+", [Turn(1, 1, 1, True, "ok")], "cmp_boundary"),
           Task("b", "HumanEval+", [Turn(1, 0, 1, False, "ok")], "cmp_boundary"),
           Task("c", "HumanEval+", [Turn(1, 1, 1, True, "ok")], "drop_guard")]
    s = summarize([Task("x", "HumanEval+", [Turn(1, 1, 1, True, "ok")])], rep)
    assert abs(s["repair_rate"] - 2 / 3) < 1e-9
    assert s["repair_by_kind"]["cmp_boundary"]["rate"] == 0.5
    assert s["repair_by_kind"]["drop_guard"]["rate"] == 1.0


def test_scalars_expose_the_gap():
    s = summarize([Task("x", "HumanEval+", [Turn(1, 1, 1, True, "ok")])],
                  [Task("a", "HumanEval+", [Turn(1, 0, 1, False, "ok")], "k")])
    sc = ed.scalars(s)
    assert "code/gap_final_minus_turn1" in sc
    assert "code/repair_rate" in sc


def test_mbpp_question_includes_signature():
    """MBPP 只给一句自然语言，不附断言的话模型不知道函数名——必须附上。"""
    rec = {"prompt": "Write a function to add two numbers.",
           "test_list": ["assert add(1,2)==3", "assert add(0,0)==0"]}
    q = ed.question_of(rec)
    assert "add(1,2)" in q


def test_humaneval_question_is_prompt_as_is():
    rec = {"entry_point": "f", "prompt": "def f(x):\n    '''doc'''\n"}
    assert ed.question_of(rec) == rec["prompt"]

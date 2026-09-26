"""Offline teaching examples for PROJECT_LEARNING_GUIDE.md.

Uses the standard library only. Does not contact models, execute submitted
programs, edit experiment artifacts, or reproduce production parser behavior.
Run from the repository root: python3 -B writeup/learning_lab.py
"""

from __future__ import annotations

import ast
import json
import math
import re
import statistics


def valid_candidate(text: str) -> bool:
    """Teaching criterion: JSON, allowed name, nonempty code, Python syntax.

    Syntax validity is not functional correctness. ast.parse does not run code.
    """
    try:
        call = json.loads(text)
        if not isinstance(call, dict) or call.get("name") != "run_tests":
            return False
        args = call.get("arguments")
        if not isinstance(args, dict) or not isinstance(args.get("code"), str):
            return False
        if not args["code"].strip():
            return False
        ast.parse(args["code"])
    except (ValueError, SyntaxError, TypeError):
        return False
    return True


def envelope_parser(text: str) -> bool:
    """A deliberately small teaching parser; NOT a copy of Hermes."""
    match = re.fullmatch(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    return bool(match and valid_candidate(match.group(1)))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar for nonnegative discordant counts."""
    if b < 0 or c < 0:
        raise ValueError("Counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def main() -> None:
    print("所有示例为教学构造；不调用模型，不运行候选程序。\n")
    good = json.dumps({
        "name": "run_tests",
        "arguments": {"code": "def add(a, b):\n    return a + b"},
    })
    wrong_tool = json.dumps({"name": "add", "arguments": {"a": 1, "b": 2}})
    wrong_answer = json.dumps({
        "name": "run_tests",
        "arguments": {"code": "def add(a, b):\n    return a - b"},
    })
    # A literal newline in a JSON string makes the JSON invalid.
    malformed = '{"name":"run_tests","arguments":{"code":"def add(a,b):\nreturn a+b"}}'
    weak_regex = re.compile(r'"name"\s*:\s*"run_tests".*?"code"', re.DOTALL)

    print("1. 调用候选、信封和正确性")
    print(f"合法裸调用: 候选判据={valid_candidate(good)}, 信封parser={envelope_parser(good)}")
    print(f"同一payload加信封: {envelope_parser('<tool_call>' + good + '</tool_call>')}")
    print(f"错工具名: 候选判据={valid_candidate(wrong_tool)}")
    print(f"JSON损坏: regex命中={bool(weak_regex.search(malformed))}, 候选判据={valid_candidate(malformed)}")
    print(f"add写成减法: 候选判据={valid_candidate(wrong_answer)}；语法通过不保证答对题。\n")

    assert valid_candidate(good) and not envelope_parser(good)
    assert envelope_parser("<tool_call>" + good + "</tool_call>")
    assert not valid_candidate(wrong_tool) and not valid_candidate(malformed)
    assert valid_candidate(wrong_answer)

    y00, y10, y01, y11 = 0.0, 0.0, 0.0, 0.98
    simple_a, simple_b = y10 - y00, y01 - y00
    marginal_a = (y10 + y11 - y00 - y01) / 2
    marginal_b = (y01 + y11 - y00 - y10) / 2
    interaction = y11 - y10 - y01 + y00
    print("2. 四格0/0/0/0.98的效应")
    print(f"baseline简单效应 A={simple_a:.2f}, B={simple_b:.2f}")
    print(f"平均主效应 A={marginal_a:.2f}, B={marginal_b:.2f}")
    print(f"交互对比={interaction:.2f}\n")

    rewards = [0.0, 0.5, 0.5, 1.0]
    mean = statistics.mean(rewards)
    # Population std is an explicit choice for this teaching example.
    std = statistics.pstdev(rewards)
    grpo = [(x - mean) / std for x in rewards]
    rloo = [x - (sum(rewards) - x) / (len(rewards) - 1) for x in rewards]
    print("3. 组内相对奖励（教学公式，不代表某版框架全部loss）")
    print("reward:", rewards)
    print("GRPO:", [round(x, 4) for x in grpo])
    print("RLOO:", [round(x, 4) for x in rloo])
    print("组内全相等时，减均值项为0；这不代表KL等全部项为0。\n")

    first, rescued, regressed = 53, 9, 4
    final = first + rescued - regressed
    print("4. 救回和回退")
    print(f"首轮={first}, 救回={rescued}, 回退={regressed}, 最终={final}")
    print(f"final-turn1={final-first}，并不等于rescued={rescued}\n")

    print("5. 配对检验")
    print(f"15 vs 3: exact p={exact_mcnemar(15, 3):.8f}")
    print(f"3 vs 0: exact p={exact_mcnemar(3, 0):.8f}")
    print(f"12次检验的Bonferroni示例阈值={0.05/12:.8f}")
    assert math.isclose(exact_mcnemar(3, 0), 0.25)
    assert exact_mcnemar(0, 0) == 1.0
    assert 0.05 / 12 < exact_mcnemar(15, 3) < 0.05
    assert math.isclose(marginal_a, 0.49) and math.isclose(interaction, 0.98)
    print("\n教学示例检查通过。请把每个输出解释一遍，再看学习文档答案。")


if __name__ == "__main__":
    main()

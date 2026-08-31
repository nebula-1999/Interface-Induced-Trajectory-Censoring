#!/usr/bin/env python3
"""程序化 bug 注入——scripted 修复探针的核心。

Step 0 推翻了"用自然发生的 turn-1 失败当分母"的做法：base 和 RL 的失败集
不是同一批题，RL 的失败集更小更难，条件修复率下降可能纯粹来自分母变难。
替代方案是给两个模型**逐字相同**的 buggy 初稿，看谁能修好。

对注入的要求，缺一不可：

  1. **可复现** —— 按 task_id 播种，同一道题永远得到同一个 bug。
  2. **语法正确** —— 走 AST 变换 + ast.unparse，不做字符串替换。
  3. **确实让测试失败** —— 注入完必须过一遍沙箱确认，改了但测试照过的
     不算数（生成端不做这件事，见 build_probes.py）。
  4. **像真实的小错误** —— 不是乱码。下面每一种都是人真会写出来的 bug。

每条只注入一处。变换类型会记录下来，让分析能按 bug 类型拆开修复率——
"能修 off-by-one 但修不了漏掉的边界判断"本身就是一条结论。
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 变换定义

CMP_BOUNDARY = {ast.Lt: ast.LtE, ast.LtE: ast.Lt,
                ast.Gt: ast.GtE, ast.GtE: ast.Gt}
CMP_INVERT = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
              ast.Lt: ast.Gt, ast.Gt: ast.Lt,
              ast.LtE: ast.GtE, ast.GtE: ast.LtE}
BINOP_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add,
              ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}


@dataclass
class Mutation:
    kind: str
    description: str


MUTATIONS = [
    Mutation("cmp_boundary", "比较符的边界改动（< ↔ <=），经典 off-by-one"),
    Mutation("cmp_invert", "比较方向反了（== → !=、< → >）"),
    Mutation("binop_swap", "运算符用错（+ → -、* → //）"),
    Mutation("const_off_by_one", "数值常量差一（range(n) → range(n-1)）"),
    Mutation("boolop_flip", "and / or 用反"),
    Mutation("drop_guard", "漏掉一个提前返回的边界判断"),
    Mutation("swap_args", "调用时前两个实参写反"),
]


# ---------------------------------------------------------------------------
# 单点变换器
#
# 共同约定：先遍历一遍数出所有候选点，再按种子选中**其中一个**下手。
# 一次只改一处——改多处的话，模型修好一个仍然失败，修复率就没法解释了。

class _SinglePoint(ast.NodeTransformer):
    def __init__(self, target: int):
        self.target = target
        self.seen = 0
        self.applied = False

    def _hit(self) -> bool:
        """当前候选点是不是被选中的那个。"""
        hit = self.seen == self.target
        self.seen += 1
        if hit:
            self.applied = True
        return hit


class CmpMut(_SinglePoint):
    def __init__(self, target: int, table: dict):
        super().__init__(target)
        self.table = table

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            if type(op) in self.table and self._hit():
                node.ops[i] = self.table[type(op)]()
                break
        return node


class BinOpMut(_SinglePoint):
    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if type(node.op) in BINOP_SWAP and self._hit():
            node.op = BINOP_SWAP[type(node.op)]()
        return node


class ConstMut(_SinglePoint):
    def visit_Constant(self, node: ast.Constant):
        # bool 是 int 的子类，必须排掉，否则 True 会变成 2
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if self._hit():
                node.value = node.value - 1 if node.value > 0 else node.value + 1
        return node


class BoolOpMut(_SinglePoint):
    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        if self._hit():
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node


class DropGuardMut(_SinglePoint):
    """删掉一个只含 return/raise/continue 的 if —— 漏掉边界判断。

    这类 bug 最贴近真实：处理空输入、越界、None 的那一行忘了写。
    """

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if node.orelse:
            return node
        if len(node.body) == 1 and isinstance(
                node.body[0], (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            if self._hit():
                return None            # 整个 if 删掉
        return node


class SwapArgsMut(_SinglePoint):
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if len(node.args) >= 2 and not any(
                isinstance(a, ast.Starred) for a in node.args[:2]):
            if self._hit():
                node.args[0], node.args[1] = node.args[1], node.args[0]
        return node


def _make(kind: str, target: int) -> _SinglePoint:
    return {
        "cmp_boundary": lambda: CmpMut(target, CMP_BOUNDARY),
        "cmp_invert": lambda: CmpMut(target, CMP_INVERT),
        "binop_swap": lambda: BinOpMut(target),
        "const_off_by_one": lambda: ConstMut(target),
        "boolop_flip": lambda: BoolOpMut(target),
        "drop_guard": lambda: DropGuardMut(target),
        "swap_args": lambda: SwapArgsMut(target),
    }[kind]()


def _count_sites(kind: str, tree: ast.Module) -> int:
    """数候选点：拿 target=-1 跑一遍，谁都不会命中，seen 就是总数。"""
    m = _make(kind, -1)
    m.visit(ast.parse(ast.unparse(tree)))
    return m.seen


# ---------------------------------------------------------------------------

def candidates(source: str, seed: int) -> list[tuple[str, str]]:
    """列出这段代码所有可行的 (kind, mutated_source)，按种子打乱顺序。

    返回多个而不是一个：注入了不代表测试会挂（比如改的是不影响结果的分支），
    调用方要依次试到真正让测试失败的那一个为止。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    kinds = [m.kind for m in MUTATIONS]
    rng.shuffle(kinds)

    for kind in kinds:
        n = _count_sites(kind, tree)
        if n == 0:
            continue
        order = list(range(n))
        rng.shuffle(order)
        for t in order[:3]:              # 每种最多试 3 个点，避免组合爆炸
            m = _make(kind, t)
            new = m.visit(ast.parse(source))
            if not m.applied:
                continue
            try:
                ast.fix_missing_locations(new)
                text = ast.unparse(new)
            except (ValueError, RecursionError):
                continue
            if text.strip() and text != ast.unparse(ast.parse(source)):
                out.append((kind, text))
    return out

#!/usr/bin/env python3
"""阶段 0：探明 verl 的 rollout 路径长什么样，再决定往哪里打钩子。

**为什么不能跳过这一步。** 本探针要的结论是「发出了合规调用、但服务端没接受、
也没执行」，即 emitted>0 而 accepted=0、executed=0。可是「钩子根本没装上」
产生的也是一串 0，两者在产物上无法区分。所以必须先确认要 patch 的类和方法
真实存在、名字对得上，patch 之后还要能自证被调用过。

本脚本只读不写，不占 GPU，几秒钟跑完。把输出贴回来再定 patch 目标。
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil


def show(mod_name: str, want: tuple[str, ...] = ()) -> None:
    try:
        m = importlib.import_module(mod_name)
    except Exception as e:
        print(f"  ✗ {mod_name}: {type(e).__name__}: {e}")
        return
    print(f"  ✓ {mod_name}  ({getattr(m, '__file__', '?')})")
    for name, obj in vars(m).items():
        if name.startswith("_"):
            continue
        if inspect.isclass(obj) and obj.__module__ == mod_name:
            meths = [n for n, _ in inspect.getmembers(obj, inspect.isfunction)
                     if not n.startswith("__")]
            print(f"      class {name}: {', '.join(meths[:14])}")
        elif want and name in want:
            print(f"      {name} = {obj!r}")


def main() -> None:
    import verl
    print(f"verl {getattr(verl, '__version__', '?')}  @ {verl.__file__}\n")

    print("[1] agent loop 相关模块")
    for sub in ("verl.experimental.agent_loop", "verl.workers.agent_loop",
                "verl.experimental.agent_loop.agent_loop",
                "verl.experimental.agent_loop.tool_agent_loop"):
        show(sub)

    print("\n[2] 全包扫描：模块名里带 agent_loop / tool_parser 的")
    for mi in pkgutil.walk_packages(verl.__path__, prefix="verl."):
        if any(k in mi.name for k in ("agent_loop", "tool_parser", "tool_call")):
            print("   ", mi.name)

    print("\n[3] 工具解析器：verl 自己解析还是走 vLLM 的？")
    for sub in ("verl.workers.rollout.schemas",
                "verl.tools.utils.tool_registry",
                "verl.experimental.agent_loop.tool_parser"):
        show(sub)

    print("\n[4] 我们自己的两个可 patch 点（已在 PYTHONPATH 上）")
    for sub in ("code_tool", "react_agent_loop"):
        show(sub)


if __name__ == "__main__":
    main()

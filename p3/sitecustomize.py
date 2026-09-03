"""在**每个** Python 进程（含 Ray actor）里装上 p3 探针，并接力原有的 sitecustomize。

两件事都必须做：

1. verl 的 rollout 跑在 Ray actor 里，是独立进程。只在 driver 里 patch 完全无效——
   症状是日志里有「已挂载」而产物一条不出，训练看起来一切正常。这个坑本项目
   踩过（见 PROJ_DIR/sitecustomize.py 的说明）。

2. **接力**。把 p3 放在 PYTHONPATH 最前面会让 Python 只加载本文件、
   不再加载 PROJ_DIR/sitecustomize.py，于是原有的分解评测钩子被静默关掉。
   所以这里显式按路径把它加载一遍。宁可多写十行，也不要又制造一个静默失效。

时机：等 `verl.experimental.agent_loop.tool_parser` 这个模块**本身**导入完成再动手，
不能在父包阶段——那会撞循环导入。
"""
from __future__ import annotations

import builtins
import importlib.util
import os
import sys

# 两个目标各等各的：tool_agent_loop 会 import tool_parser，所以 tool_parser
# 导入完成的那一刻 tool_agent_loop 还是 partially initialized，此时取
# ToolAgentLoop 会 AttributeError（实测过）。
_T_PARSER = "verl.experimental.agent_loop.tool_parser"
_T_LOOP = "verl.experimental.agent_loop.tool_agent_loop"
_T_VLLM_LORA = "vllm.lora.layers.column_parallel_linear"
_orig_import = builtins.__import__
_done = {"parser": False, "loop": False, "vllm_lora": False}


def _chain_original() -> None:
    """加载 PROJ_DIR 下原来的 sitecustomize.py（本文件把它挡住了）。"""
    proj = os.environ.get("P3_PROJ_DIR", "/root/autodl-tmp/code-agent")
    path = os.path.join(proj, "sitecustomize.py")
    if not os.path.isfile(path) or os.path.samefile(path, __file__):
        return
    try:
        spec = importlib.util.spec_from_file_location("_orig_sitecustomize", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[p3] ! 接力原 sitecustomize 失败: {e!r}", file=sys.stderr, flush=True)


def _hook(name, g=None, l=None, fromlist=(), level=0):
    mod = _orig_import(name, g, l, fromlist, level)
    try:
        # Check sys.modules after *every* import.  ``from package import child``
        # invokes __import__ with the parent name, so matching ``name`` exactly
        # misses a successfully loaded target (observed for ToolAgentLoop).
        if not _done["parser"] and _T_PARSER in sys.modules:
            _done["parser"] = True
            import rollout_probe
            rollout_probe.install()
        if not _done["loop"] and _T_LOOP in sys.modules:
            import rollout_probe
            if rollout_probe.install_agentloop():
                _done["loop"] = True
        if not _done["vllm_lora"] and _T_VLLM_LORA in sys.modules:
            import rollout_probe
            if rollout_probe.install_vllm_lora_debug():
                _done["vllm_lora"] = True
        if all(_done.values()):
            builtins.__import__ = _orig_import       # 三个目标都装完才复原
    except Exception as e:
        print(f"[p3] ✗ 装载 rollout_probe 失败: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
    return mod


builtins.__import__ = _hook
_chain_original()

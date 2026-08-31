"""让每个 Python 进程启动时都装上分解评测钩子——包括 Ray worker。

**为什么需要这个**：verl 的训练主循环跑在 `TaskRunner` 这个 Ray actor 里，
是独立进程。launch_ppo.py 只 patch 了 driver 进程的 RayPPOTrainer，
TaskRunner 那边完全没被 patch —— 表现为日志里有"已挂载"却从头到尾
不产出任何评测数据，而训练一切正常，极难察觉。
（train/probe_patch.py 有完全相同的问题，主论文那条线要一并修。）

sitecustomize.py 会被 Python 解释器在启动时自动导入（只要在 sys.path 上，
这里靠 run_code_grpo.sh 把 PROJ_DIR 放进 PYTHONPATH）。Ray worker 也是普通
Python 进程，照样生效。

时机很讲究：**必须精确等到 `verl.trainer.ppo.ray_trainer` 这个模块本身导入
完成**，而不是它的父包。在父包阶段就动手会撞上循环导入
（verl.trainer.distillation.losses 尚处于 partially initialized）。
而且要直接从 sys.modules 取已经装好的类，不能重新 import。
"""

from __future__ import annotations

import builtins
import sys

_TARGET = "verl.trainer.ppo.ray_trainer"
_orig_import = builtins.__import__
_installed = False


def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _installed
    mod = _orig_import(name, globals, locals, fromlist, level)
    if not _installed and name == _TARGET:
        rt = sys.modules.get(_TARGET)
        cls = getattr(rt, "RayPPOTrainer", None) if rt else None
        if cls is not None:
            _installed = True
            builtins.__import__ = _orig_import      # 先复原，避免递归
            try:
                import code_patch
                code_patch.install(cls)             # 显式传类，不再自己 import
            except Exception as e:
                print(f"[sitecustomize] 装载 code_patch 失败: "
                      f"{type(e).__name__}: {e}", flush=True)
    return mod


builtins.__import__ = _hooked_import

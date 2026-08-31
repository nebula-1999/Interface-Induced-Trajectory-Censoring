"""把分解评测挂进 verl 的验证环节。

手法照搬 train/probe_patch.py（主论文那条线已验证）：包住
RayPPOTrainer._validate，先跑 verl 自带的验证，再用**同一个 rollout vLLM
服务**跑一遍分解评测。

为什么这样接：rollout 引擎本来就维护着与当前策略同步的 vLLM 服务并暴露
OpenAI 兼容端点，而评测只需要生成——所以**权重一次都不用落盘**。
1.5B 的完整训练态 checkpoint 约 21 GB，数据盘只有 50 GB。

被 import 即生效（顶层调用 _install），引导见 launch_ppo.py。
"""

from __future__ import annotations

import os
import traceback

# 输出目录必须能按 run 区分：dump 是 append 模式，多个 seed 写同一个目录会
# 让 step_00000.jsonl 里混进不同 run 的记录，事后无法拆分。
# 由 run_code_grpo.sh 按 seed/算法设置。
_OUT = os.environ.get("CODE_EVAL_OUT") or "/root/autodl-tmp/runs/code-eval"


def _find_addresses(trainer) -> list[str]:
    """找 rollout 的 OpenAI 端点地址。

    verl 各版本把 LLMServerManager 藏在不同位置，逐个候选试，
    失败时把实际对象结构打出来——省得反复猜属性路径。
    """
    mgr = getattr(trainer, "async_rollout_manager", None)
    candidates = [
        mgr,
        getattr(mgr, "llm_client", None),
        getattr(mgr, "server_manager", None),
        getattr(getattr(mgr, "llm_client", None), "server_manager", None),
        getattr(trainer, "llm_server_manager", None),
        getattr(trainer, "server_manager", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        for attr in ("get_addresses", "server_addresses"):
            got = getattr(obj, attr, None)
            if got is None:
                continue
            addrs = got() if callable(got) else got
            if addrs:
                print(f"[code-eval] 地址来源: {type(obj).__name__}.{attr} -> {addrs}",
                      flush=True)
                return list(addrs)

    def _dump(name, obj):
        if obj is None:
            return f"  {name}: None"
        pub = [a for a in dir(obj) if not a.startswith("__")][:40]
        return f"  {name}: {type(obj).__name__} -> {pub}"

    raise RuntimeError(
        "找不到 rollout 服务地址。对象结构如下：\n"
        + _dump("trainer.async_rollout_manager", mgr) + "\n"
        + _dump("  .llm_client", getattr(mgr, "llm_client", None)) + "\n"
        + _dump("  .server_manager", getattr(mgr, "server_manager", None))
    )


def install(RayPPOTrainer=None) -> None:
    """装钩子。

    RayPPOTrainer 由调用方传入时**不再自己 import** —— sitecustomize 是在
    verl 的导入链中途被触发的，那时 verl.trainer.distillation.losses 还处于
    partially initialized 状态，重新 import 会撞循环导入
    （ImportError: cannot import name 'is_distillation_enabled'）。
    """
    if RayPPOTrainer is None:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer, "_code_patched", False):
        return
    original_validate = RayPPOTrainer._validate

    def _validate_with_code_eval(self, *args, **kwargs):
        metrics = original_validate(self, *args, **kwargs)
        try:
            addrs = _find_addresses(self)
            step = int(getattr(self, "global_steps", 0) or 0)
            from code_eval_hook import CodeEvalHook

            hook = CodeEvalHook(
                addrs,
                model_name=self.config.actor_rollout_ref.model.path,
                out_dir=_OUT,
                probes_path="/root/autodl-tmp/code-agent/probes_repair.jsonl",
                # 冒烟时由 run_code_grpo.sh 设成小数，正式 run 不设即全量
                limit=int(os.environ.get("CODE_EVAL_LIMIT", "0")) or None,
            )
            m = hook.run(step=step)
            brief = {k: v for k, v in m.items()
                     if k in ("code/final_pass", "code/turn1_pass",
                              "code/repair_rate", "code/gap_final_minus_turn1",
                              "code/elapsed_s", "code/request_failures")}
            print(f"[code-eval] step {step} -> {brief}", flush=True)
            if isinstance(metrics, dict):
                metrics.update(m)
        except Exception:
            # 首次失败必须炸掉训练：评测静默失效 = GPU 白烧十几小时、
            # 产出零份分解数据，而那正是这个项目唯一要采集的东西。
            first = not getattr(self, "_code_eval_ok_once", False)
            print(f"[code-eval] 本轮评测失败（首次={first}）：\n"
                  + traceback.format_exc(), flush=True)
            if first:
                raise RuntimeError(
                    "首次分解评测执行失败，已中止训练——继续跑只会产出没有分解"
                    "数据的无用 checkpoint。请修复钩子后重跑。") from None
        else:
            self._code_eval_ok_once = True
        return metrics

    RayPPOTrainer._validate = _validate_with_code_eval
    RayPPOTrainer._code_patched = True
    print("[code-eval] 已挂载到 RayPPOTrainer._validate", flush=True)


def _try_install_now() -> None:
    """import 本模块即尝试装载（driver 进程走这条）。

    worker 进程走 sitecustomize.py 的 import 钩子，会显式把类传进 install()。
    """
    try:
        install()
    except Exception:
        # 挂载失败不能影响训练启动本身；worker 那条路径稍后还会再试一次
        print("[code-eval] 挂载失败：\n" + traceback.format_exc(), flush=True)


_try_install_now()

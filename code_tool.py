"""verl 的代码执行工具：多轮 debug 循环的环境侧。

由 train/search_tool.py 改造而来——接口、生命周期、"工具失败不炸轨迹"的
处理方式全部沿用，改的只是把 BM25 检索换成沙箱跑测试。

纯逻辑在 code_tool_core.py（不 import verl，可单独测）；沙箱在 sandbox.py。

奖励口径：**outcome-only + partial credit**。
  - execute 一律返回 0.0，不给过程奖励——给了就等于往里塞信用分配假设，
    而本项目要测的正是纯 outcome reward 下增益落在哪一轮。
  - calc_reward 返回**最后一次**运行的通过比例。用比例而非 0/1 是因为
    1.5B 在难题上全 0 会让 GRPO 一整组 advantage 归零、没有梯度。
  - 评测另走 sandbox 的 script 模式取严格全通过，不用这里的比例。
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from code_tool_core import extract_code, format_observation, suspicious
from sandbox import run_tests

SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "完整的 Python 代码（函数定义及其依赖的 import）",
                },
            },
            "required": ["code"],
        },
    },
})


# 工具真被执行过的凭据。CodeTool 跑在 Ray worker 里，进程内计数器共享不出去，
# 所以落成文件——文件系统是同机共享的。
# 存在的意义：2026-08-28 那次 150 步训练里工具**一次都没被调用**
# （vLLM 默认 parser 与 Qwen2.5-Coder-1.5B 不兼容，finish_reason 恒为 "stop"），
# 而日志唯一的破绽是 timing_s/agent_loop/tool_calls/mean 那个 0.0。
# 4 小时 ¥20 全程没有任何报警。guard_tool_calls.sh 靠这个文件兜底。
TOOL_CALLED_FLAG = Path("/root/autodl-tmp/runs/.tool_called")


class CodeTool(BaseTool):
    def __init__(self, config: dict,
                 tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config, tool_schema or SCHEMA)
        self.timeout = float(config.get("timeout", 10.0))
        self.mem_mb = int(config.get("mem_mb", 512))
        # 截断观测，理由同 search_tool：长轨迹会撑爆上下文
        self.max_chars = int(config.get("max_chars", 1200))
        self._inst: dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs):
        """开一条轨迹。

        该样本的 test 由数据集的 extra_info["tools_kwargs"] 传进来。
        两种嵌套形式都兼容，免得因为 verl 版本差异静默拿到空 test——
        那会让所有测试"通过"，是最危险的一种脏奖励。
        """
        iid = instance_id or str(uuid4())
        test = kwargs.get("test") or (kwargs.get("create_kwargs") or {}).get("test", "")
        self._inst[iid] = {"test": test or "", "turns": 0,
                           "last_ratio": 0.0, "flags": set(), "ever_passed": False}
        return iid, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        # 工具调用专用计数（FC/tool_agent 路径）。不能用 sandbox 里的总数：
        # 那个把奖励评测也算进去了。
        try:
            _c = os.environ.get('TOOL_CALL_COUNTER',
                                '/root/autodl-tmp/runs/.tool_call_count')
            with open(_c, 'a') as _f:
                _f.write('codetool\n')
        except Exception:
            pass
        st = self._inst.get(instance_id)
        if st is None:
            return ToolResponse(text="error: 会话不存在"), 0.0, {}
        if not st["test"]:
            # 宁可显式报错也不要静默判全过
            return ToolResponse(text="error: 本题缺少测试"), 0.0, {"no_test": 1}

        code = extract_code((parameters or {}).get("code", ""))
        if not code:
            return ToolResponse(text="error: 没有收到代码"), 0.0, {"empty_code": 1}

        st["turns"] += 1
        # 落一次凭据即可，后续 touch 开销可忽略
        try:
            TOOL_CALLED_FLAG.parent.mkdir(parents=True, exist_ok=True)
            TOOL_CALLED_FLAG.touch()
        except OSError:
            pass          # 记录失败不该影响训练
        flags = suspicious(code)
        st["flags"].update(flags)

        try:
            # 沙箱是阻塞的，必须丢到线程里——直接调用会卡死整个 rollout 的事件循环
            res = await asyncio.to_thread(
                run_tests, code, st["test"],
                timeout=self.timeout, mem_mb=self.mem_mb)
        except Exception as e:
            # 沙箱本身出问题不该炸掉轨迹，交给策略自己处理
            return (ToolResponse(text=f"error: 执行失败 {type(e).__name__}"),
                    0.0, {"sandbox_error": 1})

        st["last_ratio"] = res.pass_ratio
        st["ever_passed"] = st["ever_passed"] or res.all_passed

        obs = format_observation(res.passed, res.total, res.status,
                                 res.stderr, self.max_chars)
        metrics = {"turn": st["turns"], "pass_ratio": res.pass_ratio,
                   "all_passed": int(res.all_passed), "status": res.status,
                   "duration": round(res.duration, 3)}
        if flags:
            metrics["suspicious"] = ",".join(flags)
        return ToolResponse(text=obs), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        st = self._inst.get(instance_id)
        return float(st["last_ratio"]) if st else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # 必须清，否则长训练下这个 dict 会一直涨
        self._inst.pop(instance_id, None)

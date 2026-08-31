"""ReAct 协议的 verl AgentLoop —— 让训练端的多轮真正生效。

**为什么需要它**：verl 的 tool_agent 完全绑定 OpenAI function calling
（靠 `tool_calls` 字段判断是否继续下一轮）。而实测表明 Qwen2.5-Coder 全系列
在 `tool_choice=auto` 下**从不发起 function call**（0/1000，跨 1.5B–32B），
换成 ReAct 文本协议则 95–100% 主动发起。因此用 tool_agent 训练时，
`timing_s/agent_loop/tool_calls/mean` 恒为 0——模型从未收到过真实的工具反馈，
150 步训练的多轮信号完全无效。

本类改用文本协议：解析 `Action: run_tests` + ```python 代码块```，
执行后把 `Observation:` 拼回序列。

**mask 的处理是关键**（照搬 tool_agent_loop 的做法）：
    prompt_ids   += 注入内容      # 环境产生的 token 进入上下文
    response_mask += [0] * len   # 但不参与 loss，否则模型会去学预测工具输出

注册名 react_agent；数据侧把 extra_info["agent_name"]="react_agent" 即可生效。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

_ACTION = re.compile(r"^\s*Action\s*:\s*run_tests", re.M | re.I)
_CODE = re.compile(r"Action\s+Input\s*:\s*```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
_FINAL = re.compile(r"Final\s+Answer\s*:", re.I)


@register("react_agent")
class ReActAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mt = self.rollout_config.multi_turn
        self.max_turns = mt.max_assistant_turns or 4
        self.max_obs_len = mt.max_tool_response_length or 1200
        self.response_length = self.rollout_config.response_length
        # 停在 Observation 处，否则模型会自己幻觉出工具结果，整条轨迹作废
        self.stop_strs = ["Observation:"]
        # guard_tool_calls.sh 靠这个文件判断工具是否真的被执行过。
        # ReAct 绕开了 CodeTool.execute，必须自己落凭据，否则 15 分钟后训练会被误杀。
        self.flag = Path("/root/autodl-tmp/runs/.tool_called")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra = kwargs.get("extra_info") or {}
        test_src = ((extra.get("tools_kwargs") or {}).get("run_tests", {})
                    .get("create_kwargs", {}).get("test", ""))

        metrics: dict[str, Any] = {}
        def _encode():
            # tokenize=True 在部分 transformers 版本返回 BatchEncoding 而非 list，
            # 直接 += 会 TypeError；统一取成 python list。
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                enable_thinking=False)
            if hasattr(enc, "input_ids"):
                enc = enc.input_ids
            if enc and isinstance(enc[0], (list, tuple)):
                enc = enc[0]
            return list(enc)

        prompt_ids = await self.loop.run_in_executor(None, _encode)

        response_mask: list[int] = []
        sp = dict(sampling_params)
        sp["stop"] = list(sp.get("stop") or []) + self.stop_strs
        # vLLM 默认 include_stop_str_in_output=False 会把 "Observation:" 整个删掉，
        # 导致训练序列与推理时模型实际看到的不一致
        sp["include_stop_str_in_output"] = True

        turns = 0
        n_actions = 0
        truncated = False
        response_logprobs: list[float] = []
        extra_fields: dict[str, Any] = {}
        for turn in range(self.max_turns):
            if len(response_mask) >= self.response_length:
                truncated = True
                break
            with simple_timer("generate_sequences", metrics):
                out = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sp)
            gen_ids = out.token_ids
            prompt_ids += gen_ids
            response_mask += [1] * len(gen_ids)      # 模型生成 → 计入 loss
            # 无论本轮有没有 logprobs 都要补满，否则与 response_mask 错位，
            # GRPO 会拿错 token 的 logprob
            if out.log_probs:
                lp = list(out.log_probs)
                lp = lp[: len(gen_ids)] + [0.0] * max(0, len(gen_ids) - len(lp))
            else:
                lp = [0.0] * len(gen_ids)
            response_logprobs += lp
            if getattr(out, "extra_fields", None):
                extra_fields.update(out.extra_fields)
            turns += 1

            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if _FINAL.search(text) or not _ACTION.search(text):
                break
            m = _CODE.search(text)
            if not m:
                break

            with simple_timer("tool_calls", metrics):   # 让 tool_calls/mean 不再恒为 0
                obs = await self.loop.run_in_executor(
                    None, self._run_tests, m.group(1), test_src)
            n_actions += 1
            # stop 串已被保留在模型输出里，这里只补内容，避免重复 "Observation:"
            obs_ids = self.tokenizer.encode(f" {obs}\n", add_special_tokens=False)
            if len(response_mask) + len(obs_ids) >= self.response_length:
                truncated = True
                break
            prompt_ids += obs_ids
            response_mask += [0] * len(obs_ids)      # 环境注入 → 不计入 loss
            response_logprobs += [0.0] * len(obs_ids)   # 环境注入恒为 0
            turns += 1                               # Observation 也算一轮

        n_prompt = len(prompt_ids) - len(response_mask)
        return AgentLoopOutput(
            prompt_ids=prompt_ids[:n_prompt],
            response_ids=prompt_ids[n_prompt:][: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(response_logprobs[: self.response_length]
                               if any(response_logprobs) else None),
            num_turns=turns + 1,                     # +1 为初始 user 轮
            metrics=metrics,
            extra_fields={**extra_fields, "turn_scores": [], "tool_rewards": [],
                          "react_actions": n_actions, "truncated": truncated},
        )

    def _run_tests(self, code: str, test_src: str) -> str:
        """同步执行沙箱。跑在 executor 线程里，不阻塞事件循环。"""
        # 工具调用专用计数（ReAct 路径），与 CodeTool 写同一文件、不同标签
        try:
            _c = os.environ.get('TOOL_CALL_COUNTER',
                                '/root/autodl-tmp/runs/.tool_call_count')
            with open(_c, 'a') as _f:
                _f.write('react\n')
        except Exception:
            pass
        if not test_src:
            return "error: 本题缺少测试"
        try:
            self.flag.parent.mkdir(parents=True, exist_ok=True)
            self.flag.touch()
        except OSError:
            pass
        try:
            from sandbox import run_tests
            r = run_tests(code, test_src, mode="pytest")
        except Exception as e:
            return f"error: 执行失败 {type(e).__name__}"
        if r.status == "timeout":
            return "测试超时：代码可能有死循环。"
        if r.total == 0:
            return "测试没有运行起来（语法错误或 import 失败）。"
        head = (f"全部 {r.total} 个测试通过。" if r.all_passed
                else f"{r.passed}/{r.total} 个测试通过，{r.total - r.passed} 个失败。")
        return (head + "\n" + r.stderr)[: self.max_obs_len]

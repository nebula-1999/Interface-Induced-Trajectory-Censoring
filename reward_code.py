"""GRPO 的奖励函数：从轨迹里取出最终代码，跑测试，返回通过比例。

**为什么不用 CodeTool.calc_reward**：verl 0.9.0 里 `calc_reward` 只出现在
base_tool.py 的 docstring 中，**没有任何代码调用它**——是个死接口。
把奖励放在那里会让训练全程拿 0 分，而且日志上看不出任何异常。
奖励必须完全由 custom_reward_function 给出。（train/reward_em.py 同理。）

口径：**outcome-only + partial credit**。
  - 只看最后一次提交的代码，不给过程奖励——本项目要测的正是纯 outcome
    reward 下增益落在哪一轮，给过程奖励等于先把答案掺进假设里。
  - 用通过比例而非 0/1：1.5B 在难题上整组全 0 会让 GRPO 的 advantage
    归零、没有梯度。
  - 评测另走 sandbox 的 script 模式取严格全通过，与公开 pass@1 可比。
"""

from __future__ import annotations

from code_tool_core import extract_submitted_code
from sandbox import run_tests

def extract_final_code(traj: str) -> str:
    """取轨迹里**最后一次**提交的代码。

    优先 hermes 工具调用里的 arguments.code；模型没调工具就退回最后一个
    markdown 代码块。两者都没有返回空串（记 0 分）。

    取最后一个而不是第一个：轨迹是"写→测→改"的循环，第一个是初稿。
    """
    return extract_submitted_code(traj)


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  **kwargs) -> float:
    """ground_truth 就是该题的 pytest 源码（见 prepare_code_data.py）。"""
    code = extract_final_code(solution_str)
    if not code:
        return 0.0
    test = ground_truth if isinstance(ground_truth, str) else ""
    if not test.strip():
        # 没有测试就一律 0。返回正分会制造出"免费满分"的题，
        # GRPO 会迅速学会往那些题上靠。
        return 0.0
    cfg = (extra_info or {}).get("sandbox", {}) if isinstance(extra_info, dict) else {}
    try:
        res = run_tests(code, test,
                        timeout=float(cfg.get("timeout", 10.0)),
                        mem_mb=int(cfg.get("mem_mb", 512)))
    except Exception:
        return 0.0          # 沙箱异常不该炸掉整个 batch
    return float(res.pass_ratio)

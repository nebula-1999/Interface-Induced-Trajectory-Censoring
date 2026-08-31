"""verl 训练的引导器：先 import code_patch 装好评测钩子，再进 verl 的 main。

不用 `python -c "import code_patch; ..."`：hydra 的 @hydra.main 依赖 sys.argv[0]，
`-c` 模式下会把参数解析弄乱。用真正的脚本文件最稳。

RayPPOTrainer 跑在 driver 进程，所以在这里 patch 就够，不必进 Ray worker。
"""

from __future__ import annotations

import sys

# flash_attn_shim 必须在最前：verl 的 log-prob 计算硬依赖
# flash_attn.bert_padding，而 flash-attn 最高只有 torch 2.9 的预编译轮子，
# 与本机 torch 2.13 ABI 不符（主论文那条线已踩过）。
sys.path.insert(0, "/root/autodl-tmp/code-agent/flash_attn_shim")

import code_patch  # noqa: F401  顶层 _install() 即生效

from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()

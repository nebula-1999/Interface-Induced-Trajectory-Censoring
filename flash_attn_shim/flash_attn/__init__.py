"""最小 flash_attn 包：只提供 verl 用到的 bert_padding 工具函数。

为什么需要：verl 的 _compute_old_log_prob 无条件调用 left_right_2_no_padding
-> unpad_input -> `from flash_attn.bert_padding import ...`。CUDA 分支没有回退
（NPU 分支用的是 transformers 的纯 PyTorch 等价实现）。而这台机器到 GitHub
release 的路径不通，预编译轮子拿不到；即使拿到，那是按 torch 2.9 编的，
本机 torch 2.13，ABI 大概率对不上。

为什么安全：bert_padding 里**没有任何 CUDA kernel**，全是 PyTorch 张量操作
（gather / scatter / cumsum）。这里的实现就是上游那几十行的标准写法，
带完整 autograd 支持——log-prob 计算要反向传播，缺了会静默出错。

**不提供任何 attention kernel。** 训练走 sdpa，推理走 vLLM 自带的 FlashInfer，
两者都不经过这里。如果哪天有代码真去 import flash_attn.flash_attn_interface，
会直接 ImportError 而不是给出错误结果——这是刻意的。
"""

__version__ = "0.0.0+shim"

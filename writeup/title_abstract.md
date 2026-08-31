# 题目与摘要（草案 v1）

## 主选题目

**Silent Interface Mismatch Masquerades as Missing Agentic Capability:
Evidence from Four Model Families and an RL Training Run**

中文：**静默的工具接口错配会伪装成智能体能力缺失——四个模型家族与一次 RL 训练的证据**

### 备选

1. *When the Parser Fails, the Model Looks Incapable: Interface Mismatch in Agentic Evaluation and RL Training*
2. *Zero Tool Calls, 80% Well-Formed: Quantifying What Agent Interfaces Silently Discard*
3. *A Taxonomy of Silent Failures in LLM Tool-Calling Stacks, and Their Cost to RL*

---

## Abstract（英文，约 250 词）

Evaluations of LLM agents routinely report tool-call rates taken from the serving
stack. We show that this number can be zero while the model emits well-formed
calls, and that the resulting misattribution contaminates both benchmarks and
reinforcement learning.

Configuring four model families for function calling on vLLM following official
documentation, we find that **each fails at a different layer and all fail
silently**: DeepSeek-Coder's chat template never injects tools; Qwen2.5-Coder
emits JSON that the recommended hermes parser cannot see; Llama-3.1-8B
hallucinates the *task function itself* as a callable tool in 23% of items;
Mistral-7B repeats its `[TOOL_CALLS]` marker and triggers HTTP 400. Only one
missing flag produces an error — the rest return HTTP 200 with an empty
`tool_calls` array.

For Qwen2.5-Coder the undercount **grows monotonically with scale**: across a
21× range the server parses 0/100 calls at every size, while well-formed calls
the model actually emits rise from 0 to 80/100 at 32B. Llama's failure is
eliminated by a single `strict: true` flag (23→0, p=0.0001), a remedy that three
prior single-variable controls failed to identify.

We then show the same mismatch reaching **RL training**: under function calling,
a 10-step GRPO run executes **zero** tool calls and `num_turns` stays pinned at
its minimum, while a matched ReAct run executes 2.05 calls per rollout. Because
the reward admits direct answers, no trajectory ever rewards tool use — the
tool-using branch is *unreachable by gradient*. This explains our own training
result, where RL improved first-draft quality (+2.6–2.8 pp across three runs)
while multi-turn rescues stayed flat at 6–9/540.

Swapping in a dedicated adapter restores the mechanism (parse 0→84, multi-turn
0→37, rescues 0→9), yet a protocol gap remains: with interfaces fully repaired,
ReAct still beats function calling on two of three families
(80 vs 61, p=0.0019; 74 vs 62, p=0.0169). Interface problems *mask* protocol
problems; only after fixing the former does the latter become visible.

---

## 摘要（中文）

智能体评测普遍直接引用服务栈报告的工具调用率。本文表明：**该数字可以为零，
而模型实际输出的是格式完好的调用**；由此产生的归因错误同时污染基准数字与强化学习。

按官方文档在 vLLM 上为四个模型家族配置 function calling，我们发现**每个家族在不同层
失败，且全部静默**：DeepSeek-Coder 的 chat template 根本不注入工具；Qwen2.5-Coder
输出的 JSON 不被推荐的 hermes parser 识别；Llama-3.1-8B 在 23% 的题目上把**待实现的
任务函数本身**当作可调用工具；Mistral-7B 重复输出 `[TOOL_CALLS]` 标记触发 400。
四类中只有一个缺失的 flag 会报错，其余一律返回 200 与空 `tool_calls` 数组。

在 Qwen2.5-Coder 上，这一低估**随规模单调增长**：跨 21 倍规模，服务端每档均解析出
0/100，而模型实际产出的合格调用由 0 升至 32B 的 80/100。Llama 的失败可由单个
`strict: true` 消除（23→0，p=0.0001）——此前三个单变量对照均未能识别该原因。

我们进一步证明同一错配延伸至 **RL 训练**：function calling 下的 10 步 GRPO 训练
执行了**零次**工具调用、`num_turns` 恒等于最小值，而条件匹配的 ReAct 训练每条 rollout
执行 2.05 次。由于奖励允许直接作答，**不存在任何奖励工具使用的轨迹——该策略分支
在梯度上不可达**。这解释了我们自己的训练结果：RL 提升了初稿质量（三个 run 一致
+2.6~2.8 个百分点），而多轮救回始终横盘在 6–9/540。

换用专用适配器可恢复机制（解析 0→84、多轮 0→37、救回 0→9），但协议差距依然存在：
在接口完全修复后，ReAct 在三个家族中的两个上仍优于 function calling
（80 vs 61，p=0.0019；74 vs 62，p=0.0169）。**接口问题掩盖了协议问题；
只有修好前者，后者才可见。**

---

## 写作时必须保留的限定

1. Qwen2.5-Coder 不产出 hermes 格式这一现象，**hanXen 于 2026-01 已在 vLLM
   issue #32926 报告**（提案被 close as not planned）。本文贡献是量化、分层、
   训练侧因果与闭环，**不是发现该现象**。摘要正文中应显式引用。
2. 训练侧为 **10 步机制演示**，非 150 步效果对比；措辞须为
   "parser 接受并执行的调用为 0"，**不可写成"模型从未尝试调用"**。
3. 每臂 n=100，`clean[:100]` **非随机取样**；主表为 `temperature=0` 单次采样。
4. 专用适配器同时更换 parser、chat template 与 few-shot，只能称"适配器组合"。
5. Mistral 无可用于通过率比较的 FC 臂（四配置错误数 2/42/3/39），
   摘要中不得出现其 p 值。

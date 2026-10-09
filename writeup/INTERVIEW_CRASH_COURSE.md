# 面试速通：工具接口如何改变 Coding Agent 的评测与训练

> 给面试前快速复习用。先看 1 分钟讲法和“必须记住的数字”，再按面试方向补细节。所有数字以当前长版论文和 2026-09-26 的结果更新为准。
>
> 论文：*Interface-Induced Trajectory Censoring*，arXiv:2609.03966。预印本不等于同行评审通过。

## 先花一分钟回忆整篇工作

### 一句话

我最初做的是一个基于 verl 和 GRPO 的代码智能体：模型提交代码，工具运行测试，模型根据反馈修改。排查训练和评测时，我发现工具调用率不是只由模型决定的；模型生成的文本还要经过模板、解析器和执行器，才能成为实际工具动作。接口不匹配会让模型输出里的调用没有进入被测或训练的轨迹。

### 60 秒讲法

> 我做的是一个用强化学习训练的代码智能体，模型可以调用 `run_tests`，根据测试反馈修改代码。最初我们看到最终通过率有上升，但多轮救回没有同步改善，所以我开始追踪模型输出是怎么变成工具动作的。
>
> 我把生成、解析、执行、反馈和最终评分分开记录，发现服务端返回 HTTP 200 并不代表工具调用真的发生。BFCL v4 上，同一个模型和题目只改变 serving adapter 配置，官方分数在 `simple_python` 上从 0.00 到 0.96，在 `multi_turn_base` 上从 0.00 到 0.19。进一步的模板与 parser 2×2 显示，基线下单独换其中一个都没有恢复解析，二者一起换后首请求有 196/200 被解析。
>
> 我还把这个问题带到 verl 真正的训练 rollout 中：7B、150 步的单变量对照里，只修 verl 的 parser，工具执行从 10 次变成 16,844 次。但在两种共同评估协议下，都没有检测到 held-out 成功率或 rescue 的提升。因此我把结论限定为：接口会决定训练轨迹里能出现什么；轨迹恢复了，不代表学习收益就自动出现。

如果时间只有半分钟，保留三点：**问题是什么、最强证据是什么、结论边界在哪里。**

## 这篇工作到底研究什么

一次工具调用要经过多道门：

```text
模型生成文本
  → 模板/序列化格式
  → parser 识别调用
  → 工具执行
  → 反馈进入下一轮
  → 任务评分或 RL 更新
```

如果任何一层没接上，系统可能只看到 `tool_calls=[]`。单看这个空数组，无法判断模型没尝试、输出格式不合、参数坏了，还是工具根本没执行。

论文用 **interface-induced trajectory censoring** 描述这种测量和轨迹形成问题。它不是在说“某个 parser 一定有 bug”。Qwen2.5-Coder 的 bare JSON 没有 Hermes 要求的 `<tool_call>` envelope 时，Hermes 按自身规则拒绝它可能完全正确；失败在模型训练格式、模板期望与 parser 规则之间的契约。

面试时可以说：

> 我们测量的是固定模型在具体协议和接口下发出了什么、系统接受并执行了什么。不能把接口观察到的工具调用数直接当成脱离系统配置的“模型真实能力”。

## 第 6 节：四条容易混淆的代码路径

先记住这四个问题：**谁发起测试？谁解析调用？反馈给谁？最终分数给谁？**

| 路径 | 谁发起测试 | parser 在哪里 | 报错会给模型吗 | 主要测量什么 |
|---|---|---|---|---|
| serving probe | 模型发出调用后由 probe/工具执行 | vLLM serving parser | 会，若循环继续 | 服务接口能否把生成变成动作 |
| 训练 rollout | 模型必须发出调用 | verl 的 ToolParser | 会，进入训练轨迹 | 训练时实际获得了什么工具经验 |
| 程序反馈评测 | evaluator 自动测试每轮代码 | 不需要工具调用 parser | 会，以 user message 返回 | 模型收到指定反馈后能否修复 |
| 终局 reward | trainer 对最终代码测试 | 不需要工具调用 parser | 不回到当前轨迹 | 最终答案给训练器多少奖励 |

### A. 自研 serving probe：检查 vLLM API 这一侧

probe 发一个带 `messages`、`tools`、`tool_choice` 的 HTTP 请求。vLLM 先用 chat template 把消息变成模型输入，再生成 token；随后 vLLM 的 serving parser 试着把输出变成结构化的 `tool_calls`。

如果模型输出裸 JSON，而 parser 只认 Hermes envelope，HTTP 仍可能是 200，但 `tool_calls` 为空。probe 只有在后续检查到工具名和参数、执行代码、并构造下一轮消息时，才完成了工具交互。

注意：本项目 probe 的某些配置也能从普通输出中提取代码来测最终答案。因此“sandbox 执行过”不必然等于“模型发起了有效工具调用”。应看 action、parse、execution、observation 各自的记录。

### B. 真正的 verl rollout：检查训练时的经验

数据行中的 `agent_name` 选择 AgentLoop。vLLM 在这里主要负责生成 token；verl 自己的 `ToolParser` 再识别生成结果。调用只有经过 verl parser 后，才会进入 `ToolAgentLoop._call_tool` 和 `CodeTool.execute`。

```text
verl 取题目和对话
  → rollout backend 生成 token
  → verl ToolParser 识别调用
  → AgentLoop 分派 CodeTool
  → CodeTool 跑测试
  → observation 加回对话
  → 模型继续生成
  → 轨迹结束后计算 reward 并更新参数
```

因此，给 vLLM 加一个 parser 插件，不会自动修好 verl 的 parser 注册表。即使两个实现的名字都叫 `hermes`，也不能推断它们用了同一份注册表、同一段解析代码或相同返回类型。还要确认 Ray worker 进程实际加载了注册逻辑。

**这也是 P3 的关键设计：**两条 7B、150 步 LoRA 训练臂只改变 verl parser registry，检查“parser 修复是否改变训练轨迹里的工具经验”。

### C. 分解评测器：考官主动把测试反馈递给模型

这条路径没有提交 `tools`，模型不需要自主发起 function call。evaluator 每一轮都提取模型写的代码，主动跑测试，再把结果作为新的 user 消息发回去：

```text
模型写代码
  → evaluator 自动运行测试
  → evaluator 把通过数和报错发给模型
  → 模型修改代码
  → evaluator 再测
```

它回答的是：“给模型相同的程序测试反馈时，它能否利用反馈？”它不能回答：“模型是否学会主动调用原生工具？”

这条评测仍有价值：训练前后的模型可以在同一套反馈规则下比较。但训练中的反馈是 tool-role 消息，旧的共享评测把反馈发成 user message；这是格式迁移，结果要按该协议解释。

### D. 终局奖励：测试结果给优化器，不给当前模型看

`reward_code.compute_score` 从完整轨迹提取最终代码，运行测试，返回通过比例。这个结果用于算训练奖励，不是当前 AgentLoop 中的一条 observation。

```text
轨迹结束
  → reward 函数取最终代码
  → sandbox 跑测试
  → 得到例如 0.7
  → GRPO 用它计算相对优势并更新参数
```

这条路径解释了为什么“日志里测试运行很多次”不能证明“模型训练时看到了很多测试反馈”。sandbox 可能只是被 reward 函数调用了。

## 奖励、工具反馈和多轮学习

当前奖励是 **outcome-only + partial credit**：只从最终代码的测试通过比例算分，没有按工具调用次数加分，也没有逐轮给 reward。

例子：最终通过 7/10 个测试，reward 是 0.7。这样比全对才得 1 分更容易产生组内差异；对 GRPO 来说，如果一组候选都得 0，组内优势可能没有有用差异。

但部分通过比例仍是**终局奖励**，不是过程奖励。它告诉优化器最后得 0.7，却没有直接标明：哪次调用拿到了有用报错，哪次代码改动让通过数提高。

因此这套奖励可能更容易强化“首轮就答得好”，而没有充分教会“读反馈后修复”。但这是待验证的机制假设，不能说当前实验已经证明 reward 是多轮学习失败的原因。

为什么不按调用次数奖励？因为模型可能学会重复测试、无意义地增加轮次，或在已正确时继续调用。我们真正想奖励的是**更好的任务结果和有效反馈利用**，不是次数本身。

Credit assignment 可以改善“把结果归给哪一步”，但它不会自动定义什么结果值得奖励，也不能弥补根本没有采到有效修复轨迹的问题。可以考虑的下一步是：保持最终任务目标不变，加入轮级价值估计或差分信号，并同时评估最终通过、固定错误初稿修复率和调用成本。若直接奖励每轮通过率增量，要防止反复试探刷分、测试集过拟合和为了赚修复分而故意写差初稿。

## 面试前必须记住的数字

| 结果 | 正确说法 | 解释边界 |
|---|---|---|
| BFCL v4 | `simple_python` 0.00→0.96；`multi_turn_base` 0.00→0.19 | 同模型、固定 cases/decoding/executor/scorer，只改变 serving adapter 配置；不是 parser-only 对照 |
| BFCL 2×2 | Hermes+文档模板 0/200；Hermes+专用模板 0/200；专用 parser+文档模板 0/200；专用 parser+专用模板 196/200 | 基线下单独替换任一组件都无恢复；联合替换恢复。不要说“统计学主效应为零” |
| BFCL 固定原文重放 | 对原始 100 条 `simple_python` 响应不改字节，宽松 JSON extractor 重建 97 个允许工具名/结构的调用；官方 AST scorer 通过 92/100 | 说明一些原始输出确实包含可恢复、可评分的调用；不是把 0→0.96 的全部差异都归为 parser |
| Qwen2.5-Coder 梯子 | 100 题原始 tight 数为 0、4、21、36、80；Hermes parsed 均为 0 | 分类器是 call-shaped 启发式；32B 存储截断及分类误差限制规模解释 |
| τ-bench retail | 115 题配对；parsed/executed 0 对 636；有任意执行的任务 0 对 103 | 任务解出 7 对 10，exact McNemar p=0.25；user simulator 是本地 Llama-3.1-8B，所以绝对分数不能与排行榜比较 |
| P3 训练轨迹 | broken：8,689 tight-classified generation events、10 accepted calls、10 executions；repaired：16,912 accepted calls、16,844 generation events/执行/observation | 不是同一列的计数单位；repaired 单次 generation 可含多个 calls，one-call cap 下多出的 68 个没有派发 |
| P3 旧的共同程序反馈评测 | broken 430/542；repaired 427/542；rescues 两端都是 12；exact p=0.25 | 共享 evaluator 自动跑测试、user-message 反馈；未检测到收益，不等于证明有害或完全等价 |
| P3 后来的共同原生 FC 评测 | base 239/542；broken 248/542；repaired 248/542；broken/repaired 16 得、16 失，p=1 | 事后评测、同一可用 parser；不等于重新训练或原样复刻训练 AgentLoop。通过集不相同，等总分不代表策略相同 |

### 两个最容易说错的统计口径

1. **BFCL 2×2 的“主效应为零”说法不正确。** 如果按标准 factorial design 计算边际主效应，每个组件的平均效应并非零。准确结论是：在文档配置这个基线下，单独替换模板或 parser 的 simple effect 都是 0；效果需要匹配的一对配置共同替换，统计结构主要体现为交互。
2. **P3 的 16,912 与 16,844 不是互相矛盾。** 前者是 parser 接受的 call 数；后者是 generation event 数，也是 one-call cap 下执行/返回 observation 的次数。40 个 generation 含多个 calls，额外 68 个未派发。

## 常见追问：可以这样答

### “这不就是配置错了吗？学术贡献是什么？”

> 具体问题确实是模型格式和 serving 配置不匹配，修复也使用现有工具。本文不声称发明新 parser。我们测量了这个问题如何改变标准 benchmark 的分数，用 template×parser 对照拆开组件关系，把原始输出固定后重放，并在真实 verl rollout 中验证它改变了训练经验。贡献是对测量后果和训练轨迹后果做可分解的实证分析。已知 issue 也在论文中承认。

### “BFCL 的分数变化能证明都是 parser 遮住了已有能力吗？”

> 不能。主 adapter intervention 同时改变 parser 和配套模板，模板会改变模型输入和生成。2×2 能说明二者互补，固定原始字节重放又显示原始输出里已有一批可被重解析并通过官方 scorer 的调用；但 0.00 到 0.96 的全部增益不能被解释成纯 parser recovery。

### “为什么修好 RL 接口，成功率没涨？”

> 我们直接确认工具交互经验被大幅恢复，但两套 held-out 评估都没有检测到成功率或 rescue 提升。这个实验是单 seed、7B、LoRA、150 步。它没有区分信用分配不足、预算不足、任务迁移、奖励目标或探索问题，也不证明更长训练或其他算法都不会获益。

### “通过比例奖励是不是削弱多轮？”

> 它不直接惩罚多轮，但它只看最终结果，没有告诉模型每轮反馈和修改分别贡献多少；所以首轮改善可能比反馈利用更容易被学到。现有对照没有隔离奖励函数，这只能作为后续假设。Credit assignment 可以细化结果归因，但要先定义可靠的过程信号并防止 reward hacking。

### “为什么不奖励工具调用次数？”

> 调用次数不是目标，调用有效性才是。固定每次调用加分会鼓励冗余测试。更合理的目标是最终正确性、在固定失败初稿上的修复成功，以及调用成本之间的权衡。

### “你们证明 parser 有 bug 了吗？”

> 没有。重放 vLLM 自己的 Hermes extractor 后，archive 中没有发现 parser 按自身规则本该接受却漏掉的调用。主要发现是模型输出格式、模板期望和 parser 接受格式没有对上；parser 可能在做它被设计去做的事，但这个组合仍会让评测低估可观察行为。

### “这个结果能推广到非代码任务吗？”

> BFCL 包含非代码工具任务，τ-bench retail 是有状态的客服环境；两者都看到 serving adapter 改变了调用解析或 benchmark 结果。规模梯子和受控 RL 训练仍是代码任务，所以不能据此声称规模趋势和学习效应已经跨领域验证。

### “你接下来会做什么？”

> 我会先把 reward 和 credit assignment 当成独立因素，保持任务、接口和训练预算可比，设计轮级信号并评估最终成功、固定错误初稿修复和调用成本。也会用多个 seed 看差异是否稳定；不会只看 reward 曲线或工具调用次数。

## 不要这样说

- 不要说“论文证明了模型真实能力被隐藏”。说“给定输出判据和接口，原始生成中存在服务端没有观察到的 call-shaped 内容”。
- 不要把 BFCL 称为 parser-only intervention；它改变了 serving adapter 配置，2×2 才分解模板和 parser。
- 不要说“BFCL 两个主效应精确为零”。说“在基线下单独替换任一组件没有恢复解析”。
- 不要说“RL 修复后学会/没学会工具使用”。说“工具交互经验恢复；在两个指定 held-out 协议下未检测到收益”。
- 不要把所有 `sandbox.run_tests` 计数当作模型发起的工具调用。
- 不要把 7/10 τ-bench 说成 leaderboard 分数，或说差异显著。
- 不要把预印本说成论文已被 ICLR 接收。

## 最后十秒检查

面试前能不看文档回答这五句，就够进入细节追问：

1. 我的 agent 原本要做什么？——根据测试反馈多轮修改代码。
2. 发现的问题是什么？——工具调用要经过模板、parser 和执行器，模型输出不一定进入动作轨迹。
3. 最强的控制是什么？——BFCL template×parser 2×2，加固定原文重放。
4. 训练里修复了什么？——verl parser 修复把交互从 10 次执行提高到 16,844。
5. 结论到哪里为止？——观察轨迹改变了；当前 7B 单 seed、150 步设置下没有检测到 held-out 学习收益。

## 可继续复习

- [**面试排练：怎么讲 + 30 问**](INTERVIEW_30Q.md)（2026-10-09，按 ICLR 投稿版口径）
- [完整学习文档](PROJECT_LEARNING_GUIDE.md)
- [最初 Coding Agent 是什么](ORIGINAL_CODING_AGENT_GUIDE.md)
- [项目复盘时间线](PROJECT_RETROSPECTIVE.md)
- [P3 最新结果](../paper/NATIVE_FC_UPDATE_20260926.md)
- [论文摘要](../paper/sections/abstract.tex)

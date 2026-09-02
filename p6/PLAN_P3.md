# P3 方案：broken-FC vs repaired-FC 的 RL 对照

## 一、先纠正诊断：卡点不是「verl 不暴露 guided decoding」

复盘 §12.3 把 P3 的障碍记为「卡在 verl 不暴露 guided decoding」。**这个诊断
与本文自己的论点不一致，而且被今天的 rollout 探针推翻。**

guided decoding 是**强迫模型输出某种格式**。但本文全篇主张的修法不是强迫模型，
而是**让解析器接受模型实际发出的东西**（§5.3 的 2×2：换模板或换 parser 单独无效，
匹配才有效）。所以 repaired-FC 不需要 guided decoding。

今天的探针还给出了具体路径：verl 的 AgentLoop **自带一套 ToolParser 注册表**
（`verl.experimental.agent_loop.tool_parser`，注册名有 hermes / qwen3_coder / glm /
kimi / deepseek_v4 等），由 `actor_rollout_ref.rollout.multi_turn.format` 选择；
注册用的是 `@ToolParser.register(name)` 类装饰器，`get_tool_parser` 按名字查表。
我们已经在这套机制上成功装过钩子（`p3/rollout_probe.py` + sitecustomize，
115 次生成全部截获）。

**因此 repaired-FC 臂 = 向 verl 注册一个接受 Qwen2.5-Coder 实际输出格式（裸 JSON）
的 ToolParser，并把 `multi_turn.format` 指向它。** 这是本文 §5.3 那个 adapter 在
训练侧的对应物。

> **步骤 0（未验证，必须先做）**：确认 `multi_turn.format` 不是被校验成固定枚举，
> 且自定义注册名能被 `get_tool_parser` 取到。机器当时已关机，无法现场验证。
> 若被枚举卡死，退路是直接 monkey-patch `HermesToolParser.extract_tool_calls`
> ——今天的探针已证明该点可稳定拦截。

## 二、真正的卡点：**规模**

这才是拦住 P3 的东西，而且今天两个实验各提供了一半证据：

- 论文里的 RL run 是 **1.5B**。§5.7.1 的探针显示：该 checkpoint 在训练提示词下
  **0/52 的输出点名 `run_tests`**——它根本不发出合规调用。
  **修好 parser 也没有东西可接受**，repaired-FC 在 1.5B 上必然仍是零工具执行。
  这与今天 Granite 臂被判为「无信息量」是同一个道理。
- 会发出合规调用的是 **7B**：rollout 探针实测 115 次生成里 **45 条**是合法完整的
  裸 JSON 调用。**P3 必须在 7B 上做。**
- 而 7B 全参 RL 在单张 80 GB 卡上 **CUDA OOM**（今天实测，OOM 发生在 rollout 之后
  的 actor 更新阶段）。

**所以 P3 = 7B 的 RL，卡点是显存，不是 verl。**

## 三、让 7B RL 跑起来的三条路

| | 做法 | 代价 | 风险 |
|---|---|---|---|
| **A** | **LoRA/PEFT 代替全参**（verl 支持） | 单卡可行，显存降一个量级 | 与历史 150 步全参 run **不可直接比**；但 P3 是**臂内对照**（broken vs repaired 都用 LoRA），组内可比即可 |
| B | 更激进的 offload + micro-batch=1 | 零额外成本 | 今天已用 param/optimizer offload 仍 OOM；余量只有 890 MiB，希望不大 |
| C | 双卡 FSDP | 翻倍机时 | 主论文那台是 PCIe 无 NVLink，P2P 仅 17.3 GB/s；可行但慢 |

**建议 A。** 理由是 P3 要回答的是**臂间差**（修好接口是否恢复多轮学习），不是
复现历史绝对值；两条臂用同一套 LoRA 配置，对照就成立。这也和 §5.2 那条纪律一致：
组内对照不需要与外部数字可比。

## 四、实验设计

固定：同模型（Qwen2.5-Coder-7B）、同数据（`train_fc3.parquet`，tool_agent +
FC_MANDATORY 提示词——**注意 `train.parquet` 已被 ReAct run 覆盖**）、同种子、
同步数、同 LoRA 配置、同奖励函数。

| 臂 | `multi_turn.format` | 预期 |
|---|---|---|
| broken-FC | `hermes` | 工具执行 ≈ 0（复现 §5.7） |
| repaired-FC | 自定义 bare-JSON parser | 工具执行 > 0 |

**每步必须记录**：工具执行次数、`num_turns` 分布、多轮救回数、以及
emitted/parsed/executed 三层计数（复用 `p3/rollout_probe.py` 的钩子）。

## 五、它能决定什么

- **repaired-FC 恢复了多轮学习** → §5.8 的
  「a working channel is necessary, not sufficient」**是错的**，必须撤回。
  这是本文唯一一处能被自己推翻的主张，做这个实验就是为了给它一次被推翻的机会。
- **没有恢复** → 该主张从一条臂（ReAct，同时改了协议与接口）加强到两条臂，
  且第二条是**单变量**的（只改 parser）。这正是审稿意见要的那个干净对照。

两种结果都值得报，且**第一种对论文更有价值**——它会纠正一个已发表的判断。

## 六、成本与前置条件

- 两条臂各 75–150 步。按今天 7B 的吞吐估，LoRA 下单臂约 4–8 小时，合计 8–16 小时。
- **磁盘**：`autodl-code` 数据盘只剩 3.9 GB。LoRA checkpoint 小得多，但仍需先腾空间
  或把 `default_local_dir` 指到系统盘（还有 17 GB）。
- **必须先做的两件事**（今天写在 §7 里的欠账）：训练前修掉验证器的两处缝隙
  （正则锚定、`skipped` 计入分母），并**保留完整 rollout 文本**——这次能查到什么
  全靠日志碰巧带了代码片段。

## 七、若步骤 0 失败

若自定义 parser 注册不通、monkey-patch 也不稳，则退回原诊断，P3 需要换
rollout backend（SGLang）。但在花那个工程量之前，**先做步骤 0**——它只要几分钟。

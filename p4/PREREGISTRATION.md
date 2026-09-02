# 预注册 v2：固定 parser 下的前瞻机制检验（写于运行之前）

> v1 有四处实质错误，已按外部审阅意见更正，更正内容见文末「v1 的错误」。
> v1 保留在 git 历史里（commit `10eb9b7`），不删。

## 这个实验要检验的假设（写窄，不写大）

**H：给定一个固定的 parser，能否在运行之前、仅凭 chat template 的序列化约定，
预测该 parser 是否观测得到模型发出的工具动作。**

被检验的是**「模板 → 可观测性」这条预测规则**，不是本文的 censoring 机制本身。
两者必须分开：Qwen2.5-Coder 恰恰证明了**模板与训练行为可以脱钩**（它继承
Instruct 的模板却没在该格式上训练过），所以预测失败完全可能，而那只推翻这条
规则的当前形式。

## 与论文主结果的关键区别（必须写在正文里）

| | Qwen2.5-Coder（§5.2/§5.3） | 本实验 |
|---|---|---|
| 怎么走到 hermes 的 | **按官方文档配置就会走到** | **我们故意指定** |
| 性质 | 真实的 measurement-validity 失效 | 受控的机制检验 |

Granite 的 vLLM 文档推荐 `--tool-call-parser granite`，GLM 亦有自己的路径。
所以 **Granite+hermes=0 是预期内的不兼容，不是"又发现一个官方栈的 bug"**。
本实验绝不作后一种表述。

## 臂与预测（提交时间早于运行）

固定：同一题集（`clean_ids.json[:100]`）、`temperature=0`、`seed=0`、
`fc-schema terse`、`tool_choice=auto`、同一 tool schema、同一 max_tokens。

### A. 阳性对照 + 反事实规模梯子：Qwen3 在**匹配**接口下

Qwen3 的模板注入 tools 并要求 `<tool_call>`，与 hermes 一致；vLLM 文档也对
Qwen3 示范 hermes。**预测：各尺寸解析率均显著 > 0，且不随尺寸出现递增的
silent fraction。**

| 臂 | parser | 预测 |
|---|---|---|
| Qwen3-0.6B / 1.7B / 4B / 8B（视磁盘可加 14B） | hermes | 解析率 > 0，undercount 不随规模增长 |

**这条梯子的价值在于它是反事实对照**，直接冲 §5.2 最大的混淆：
「emitted 列随规模增长，可能只是大模型更会写 JSON 的一般能力趋势」。
若 Qwen3 在**匹配**接口下不出现递增的 undercount，则
「规模 + 匹配接口 ⇒ 无递增 undercount」对照「规模 + 错配接口 ⇒ 递增 undercount」
成立，混淆被切断。它**不会**给出第二条 0→80 式的 censoring 梯子，也不打算给。

### B. 负向臂 + rescue 对照：Granite-3.1-8B

模板注入 tools，信封是 `<|tool_call|>`（与 hermes 的 `<tool_call>` 是不同 token）。

| 臂 | parser | 预测 |
|---|---|---|
| Granite-3.1-8B | hermes | 解析率 ≈ 0 |
| Granite-3.1-8B | **granite** | 解析率 **> 0** |

**rescue 臂是必需的，不是加分项。** 只跑 hermes 得到 0，无法排除"模型压根没调
工具"。只有同一批输出在专用 parser 下被解析出来，才能证明 0 来自序列化契约
而非能力缺失。

### C. GLM-4-9B：**本轮不跑**

模板把工具写成中文 markdown「# 可用工具」，无 hermes 式信封，预测会是 0。
但 **vLLM 0.27.1 只带 `glm47_moe`（面向 GLM-4.7 MoE），没有适配 GLM-4-9B 的
parser**，因此**做不出 rescue 对照**。按上一条的标准，没有 rescue 的 0 不可解释，
故本轮不跑。列在此处是为了说明取舍，不是留白。

## 什么算证伪

- Qwen3 在 hermes 下解析率 ≈ 0 → 推翻「模板信封一致 ⇒ 可观测」这一半。
- Granite 在 hermes 下解析率显著 > 0 → 推翻「模板信封不一致 ⇒ 不可观测」这一半。
- Granite 在 granite parser 下仍 ≈ 0 → 该臂的 0 无法归因于契约，整条臂作废。

以上任一发生，**被推翻的是本文件开头那条预测规则**，不是论文的 censoring 机制。

## 局限（先写下来）

1. **本次普查只看了 chat template，没有普查各家的推荐 parser**，因此
   **不能**据此说"接口错配普遍、模板与 parser 相符罕见"。v1 里那句话已删。
2. 判据只看模板文本，不看模型实际训练成什么。模板与训练脱钩是已知现象。
3. Granite 只有 2B/8B，GLM-4-9B 单尺寸；A 组的 Qwen3 是梯子但属于**匹配**接口，
   所以本实验**不提供**第二条 undercount 梯子。原意义上的 P2 仍然开着。
4. 普查的九个家族里有六个（Phi-4 / InternLM2.5 / Yi-1.5 / StarCoder2 /
   DeepSeek-Coder-V2 / OLMo-2）的默认模板不注入 OpenAI 式 tools。
   **这不等于它们不会 censor**——按本文分类学，模板不注入本身就是失败的一层。
   正确表述是：**它们不适合用来检验 envelope/parser 层的 censoring**。

## v1 的错误（对照留证）

1. 写了「六个不注入 tools → 不可能表现出 censoring」。**错**：模板缺失本身就是
   本文分类学中的一层失败，应为「不适合检验 envelope/parser 层的 censoring」。
2. 写了「接口错配不是罕见事故，稀少的反而是模板与 parser 恰好同意」。
   **证据不足**：只普查了模板，未普查各家推荐 parser。已删。
3. 写了「剩下三个各自只有一两个尺寸」。**事实错误**：Qwen3 有完整 dense 梯子
   0.6B/1.7B/4B/8B/14B/32B，六档均已核实存在。
4. 写了「任一条不符即推翻机制」。**说大了**：被推翻的是模板预测规则的当前形式。

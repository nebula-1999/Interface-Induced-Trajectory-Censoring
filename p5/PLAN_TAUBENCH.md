# P5 准备：τ-bench 上的接口错配复现

## 为什么做这个（它攻的是哪条局限）

§7 局限第 1 条：

> **One task family, one tool.** 所有臂都是 KodCode 编程题 + 单一工具
> `run_tests`（单个 `code` 参数）。多工具 schema、在多个工具间选择、非代码域
> 均未测试。单工具 schema 是对 parser **最容易**的情况，因此测得的 censoring
> 若有偏差也是偏小——**但那是论证，不是测量**。

τ-bench 是**多工具、需要工具选择、非代码域、带用户模拟器的交互式环境**。
它把上面那句「是论证不是测量」变成测量。这是清单上唯一能碰这条局限的实验。

## 关键发现：不需要写适配层

τ-bench 的 `ToolCallingAgent.solve()`：

```python
res = completion(messages=..., tools=self.tools_info, ...)
next_message = res.choices[0].message.model_dump()
action = message_to_action(next_message)      # 读 message["tool_calls"]
```

**它直接读服务端解析后的 `tool_calls` 字段。** 解析不出来 → 空数组 →
`message_to_action` 判定为 RESPOND（自然语言回复）→ 环境收不到任何工具动作。

对比 BFCL：那次出厂路径走 `/v1/completions` 并在 benchmark 进程内正则抠标签，
我们必须写 `bfcl_registration.py` 把它掰回 `/v1/chat/completions`。
**τ-bench 本来就在我们研究的那条路上，零适配代码。**

这也意味着：若结果为阳性，它比 BFCL **更贴近真实用法**——没有任何我们加的东西。

## 架构

```
τ-bench (litellm)
   ├── agent      → 记录代理 :8001 → vLLM :8000   ← 本实验的自变量在这里
   └── user sim   → vLLM :8010                    ← 跨臂完全固定
```

- **agent 模型**：Qwen2.5-Coder-7B-Instruct（与 §5.3 的 BFCL 臂同一权重）
- **用户模拟器**：固定一个模型、固定 `temperature`/种子，**两条臂完全一致**
- **记录代理**：复用 `p1/toolcall_proxy.py`，逐请求记录
  emitted → parsed，与 BFCL 那次同源

### litellm 如何分别指向两个后端

优先：agent 用 `hosted_vllm` provider（读 `HOSTED_VLLM_API_BASE`），
user 用 `openai` provider（读 `OPENAI_API_BASE`）——两个不同环境变量，天然分流。
**开跑前必须先验证** `hosted_vllm` 在该版 litellm 的 `provider_list` 里。

退路：给 `toolcall_proxy.py` 加**按 `model` 字段路由**，单端点分发到两个后端。
改动不大，且顺带把用户模拟器的流量也记下来。

## 臂

与 §5.3 的 BFCL 完全对齐，便于并排：

| 臂 | agent 侧配置 | 预期 |
|---|---|---|
| documented | `--tool-call-parser hermes` | 解析 ≈ 0，任务成功率 ≈ 0 |
| repaired | 社区 Coder parser + 配对模板 | 解析 > 0，成功率 > 0 |

若资源允许，再补 2×2 的两个非对角格（模板与 parser 各换一侧），
与 BFCL 那次的「主效应为零、效应全在交互项」相互印证。

## 必须写进论文的局限（现在就定，不等结果）

1. **用户模拟器不是 gpt-4o。** τ-bench 默认用 gpt-4o 当用户模拟器，而 AutoDL
   连不上 OpenAI。我们用本地模型。**因此绝对分数不可与 τ-bench 排行榜比较**，
   本实验只主张**臂间差**——两条臂用的是同一个用户模拟器、同一配置、同一种子，
   所以差值仍然干净。这一点必须在报告数字的同一段里写明，不能放在脚注。
2. 环境是 mock（`MockRetailDomainEnv`），数据随仓库发布，不联网。
3. 任务数：retail 约 115、airline 约 50。多轮对话，7B 本地推理下耗时可观。

## 资源与风险

- **两个 vLLM 实例同时在线**（agent 7B + user sim）。单卡 80 GB 可行，
  各给 0.4 左右显存；若用户模拟器换更大的模型则需要第二张卡。
- **磁盘**：agent 模型已在盘上；用户模拟器若选已有模型则零新增下载。
- **主要风险**：litellm 版本与 provider 名称、以及两个后端的分流方式。
  这是唯一需要现场调试的环节，**建议开卡后先用 2--3 个 task 冒烟**再跑全量。

## 开跑前的检查清单

1. `pip install -e .` τ-bench，确认 litellm 版本与 `provider_list` 含 `hosted_vllm`
2. 起两个 vLLM，`curl /v1/models` 各自可达
3. `run.py --task-ids 0 1 2` 冒烟，确认：
   - agent 的请求经过记录代理（代理日志非空）
   - 请求体里 `tools` 非空、`tool_choice` 为 auto
   - 两条臂的用户模拟器输出一致（同种子下应当逐字相同）
4. 冒烟通过再跑全量；**冒烟不过不要跑全量**

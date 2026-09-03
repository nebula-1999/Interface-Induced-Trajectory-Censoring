# P3 结果落地后的论文改动地图

结果到手后按本文件逐条改。**不要凭记忆找位置**——这些行号是 2026-09-03
核对过的，改前请先确认行号未漂移（搜索引文里的关键词更稳）。

---

## 零、动手前必读：两个 "repair" 不是一回事

论文里**已经有一处** repair 结果，别和 P3 混为一谈：

| | §5.3 推理侧修复（**已在论文里**） | P3 训练侧对照（**本次新增**） |
|---|---|---|
| 问的问题 | 换 adapter 后，能力信号回来多少 | 修好接口后，**多轮学习是否随训练恢复** |
| 设置 | Qwen2.5-Coder-7B 推理，`tool_choice: auto` 固定 | 7B + LoRA，两臂只差 verl parser 注册 |
| 数字 | parse 0→84、二轮 0→37、rescues 0→9、pass 53→62 (n.s.) | 见 `p3/results/*_formal_summary.json` |
| 位置 | `05_results.tex:213-238`（`\ref{sec:repair}`） | §5.7.2（`\ref{sec:sufficiency}`） |

摘要那句 "Repair restores the mechanism and not the outcome" 说的是**左边**。
P3 的结论**不改** §5.3 的任何数字，只在 §5.7.2 及其下游位置新增训练侧的证据。

---

## 一、必改位置清单

### 1. `05_results.tex:296-337` — §5.7.2（**主战场**）
标题：`Why no repaired-FC arm exists yet, and what a working channel did not buy`
label：`sec:norepair` / `sec:sufficiency`

**分支 A（repaired 恢复了多轮学习）→ 标题和小节整体重写**：
- L300-301「The natural control is repaired FC rather than ReAct, and we did not
  have it at the time of writing」→ 改为「该对照现已存在」并给出结果
- L311-318「Correcting our own prescription」段：guided decoding 那段诊断更正
  **保留**（它是真实历史，且体现自我修正），但要补「现已跑通」的后续
- L320-337 的 sufficiency 结论 → **撤回**。语气沿用撤回 p=0.648 那处：
  干脆、不辩解、不铺垫

**分支 B（未恢复）→ 局部改写**：
- L300-301 同样改为「该对照现已存在」
- L333-334「This is a ReAct arm, not a repaired-FC arm…cannot separate…」
  → 删掉这句限制，改为「该分离现已由单变量对照完成，结论为 X」
- L330-332 的 narrower statement 保留并加强（从一条臂升级为两条臂）

### 2. `H_limitations.tex:44-51` — item 8
现标题「There is no clean repaired-FC training control」。
两个分支都要改标题和内容；分支 B 下它从「没有对照」变成「对照存在且结论为 X」。
L46-48 关于 guided decoding 的历史更正**保留**。

### 3. `H_limitations.tex:69-73` — item 12
「Opening the channel is tested only through ReAct」→ 按分支改写。

### 4. `H_limitations.tex:63-68` — item 11（**与分支无关，必须改**）
「The FC training arm did not save raw rollout text」——Codex 已在 `e5613a9`
修好（原文完整保留 + SHA256）。这条现在是过时的，无论哪个分支都要更新。
注意 `07_limitations.tex:45-46` 有同一处表述，一并改。

### 5. `07_limitations.tex:63-69` + `H_limitations.tex:116-123`
「What would change our conclusions」第一条 broken-vs-repaired 对照
→ 从**待办**移入**已完成**，并写明结果落在哪个分支。

### 6. `08_conclusion.tex:10`
「recovers only part of the outcome gap」一句，按分支调整。

### 7. `06_discussion.tex:38` 起的 cold-start 段
按分支调整。分支 A 下这段的论证力度要减弱（因为通道打开后确实学会了）。

### 8. 摘要（`abstract.tex` 末段）
末段现有「at this model scale and training budget, a working channel was not
sufficient」类表述——分支 A 下必须撤回。注意**只改训练侧那句**，
§5.3 推理侧的 "Repair restores the mechanism and not the outcome" 不动。

---

## 二、绝对不许动的东西（改前对照一遍）

- **§5.3（`05_results.tex:213-238`）的所有数字**：84 / 37 / 9 / 53 / 62 /
  p=0.093 / 160/300 / 178/300 / p=0.0385 / Bonferroni ≈0.0042 / p=3.1e-06 /
  +8.4pp / p=0.118
- **Appendix A（Data Errata）全文**：撤回 p=0.648、七个 arm 的 max_tokens
  记录错误、Llama initiation 97→74
- **§7 的 16 条 limitation（附录 H 全量）**：可压缩每条到 2-3 句，**不可删条目**
- **预注册记录**（`p4/PREREGISTRATION.md`）与「跑之前提交了预测」的表述
- **所有对自己不利的披露**：n=300 复现中 7B 从 21 变 30、7B rollout probe 的
  CUDA OOM、Granite 臂 uninformative、Qwen3 ladder 的 parsed 列非单调、
  enable_thinking=false 协议偏离、4000 字符截断、κ=0.713 是仪器产物
- **验证门控规则（§3.3）**

---

## 三、新增内容必须写清的三件事

1. **PEFT 说明**：两臂都用 LoRA（rank 32 / alpha 32 / all-linear），
   因为 7B 全参 RL 在单卡 80 GB 上于 actor 更新阶段 OOM。
   **必须写明与历史 150 步全参 baseline 不严格可比**——P3 是臂内对照，
   组内可比即可，但不能默认可与外部数字比。
2. **单变量性质**：两臂只差 `actor_rollout_ref.rollout.multi_turn.format`
   一个配置键（hermes ↔ qwen2_5_coder），模型/数据/种子/LoRA/奖励全同。
3. **自证**：三个钩子（parser / `_call_tool` / `CodeTool.execute`）均有安装
   记录，parser 产生运行时事件；broken 臂的零计数是真实的零，
   不是钩子没挂上（这正是第 11 条欠账还清后才敢说的话）。

---

## 四、怎么判分支

```bash
# 两臂各跑一遍，输出每步趋势
python3 p3/parse_steps.py <broken_driver.log>   --arm broken
python3 p3/parse_steps.py <repaired_driver.log> --arm repaired
```

判据：
- broken：`tool_calls_always_zero = true`（`timing_s/agent_loop/tool_calls/mean`
  每步恒为 0.0）、`num_turns/mean` 恒为 2.0
- repaired：`tool_calls_time > 0`，看 `score` 的**前后四分之一均值差**：
  - `delta` 明显为正且曲线上升 → **分支 A**
  - `delta` ≈ 0 或为负 → **分支 B**

统计上要克制：本脚本给的是效应量为主、Welch t 为辅。论文在 §5.6 已因多重
比较吃过亏（p=0.093 不通过 Bonferroni），这里别只报 p 值。

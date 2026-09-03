# P3 结果落地后的论文改动地图（2026-09-03 重写）

> 旧版的分支 A / 分支 B 框架**已作废**——它按 `critic/score/mean` 判分支，
> 而那是本文第 1 节自证为混淆的聚合量。作废理由见
> `p3/PREREG_AMENDMENT_20260903.md`。本文件是替代版。

---

## 零、动手前必读

**两个 "repair" 不是一回事。** 论文里已经有一处 repair 结果，别混：

| | §5.7 推理侧修复（**已在论文里**） | P3 训练侧对照（**本次新增**） |
|---|---|---|
| 问的问题 | 换 adapter 后能力信号回来多少 | 修好接口后**多轮学习是否随训练恢复** |
| 设置 | 7B 推理，`tool_choice: auto` 固定 | 7B + LoRA，两臂只差 `multi_turn.format` |
| 数字 | parse 0→84、rescues 0→9、pass 53→62 (n.s.) | 见下 |

摘要那句 "Repair restores the mechanism and not the outcome" 说的是**左边**。
P3 **不改** §5.7 的任何数字。

---

## 一、无条件要改的（不依赖 repaired 的走势）

这三处只要两臂机制数字到手就能改，**与 sufficiency 的结论无关**。

### 1. §7 item 8 / `H_limitations.tex`
现文：「**There is no clean repaired-FC training control.** …ReAct is a positive-control
interaction channel, not a parser-repair control.」

→ 改为：该对照**现已存在**。两条臂只差 `actor_rollout_ref.rollout.multi_turn.format`
一个键（`hermes` vs `qwen2_5_coder`），模型、数据、种子、LoRA 配置、奖励函数逐字一致。
保留 guided decoding 那段自我更正（它是真实历史）。

### 2. §5.8 `sec:norepair`
现文：「The natural control is *repaired* FC rather than ReAct, and we did not have it
at the time of writing.」

→ 改为给出两臂 150 步的训练内机制对比：

| | broken (`hermes`) | repaired (`qwen2_5_coder`) |
|---|---|---|
| 语义完整调用 | 8591 | 待填 |
| 带 `<tool_call>` 信封 | 20 | 待填 |
| **被 parser 接受** | **7 (0.08%)** | 待填 |
| 工具执行 | 7 | 待填 |
| `num_turns/mean` | 恒 2.000 | 3.x（step 5 实测） |

**措辞要点**：不能写「hermes 接受 0」。是 **7/8591 = 0.08%**，
或「99.9% 的合规调用被丢弃」。那 7 次执行是**本 run 内部的阳性对照**——
解析器与执行器都证明可用，所以其余 8584 条不是管线故障而是契约不匹配。
别重蹈早期 "never" 的绝对化覆辙。

### 3. §5.6 `sec:rl` 的 "Wording bound"
现文：「FC rollouts were not saved, so direct answers, malformed calls and dropped
intent cannot be separated from the training logs.」

→ **这句话已不成立**。P3 保留了完整 rollout 文本（实测 `text_chars` 最大 12943，
30 条超 4000 字符，每条带 sha256），三者现在分得开。同步改
§7 item 11（零执行 ≠ 零尝试）与 item 14 的审计边界
（"full rollout text was not retained"）。

---

## 二、依赖 repaired 走势的（sufficiency）

**判据**：repaired 臂 **multi 通道**的 `rescued` **率**在 step 0/30/60/90/120/150 的走势。
`rescued` 按 `analyze_step3.py` 口径逐点重算（最终过 且 首轮没过），
**不要直接读 `gap_final_minus_turn1`**——回退数在 step 0 是 0，但不保证一直是 0。

**基线**：step 0 = **20/1108 = 1.80%**（二项 SE ≈ 0.40pp，95% 区间 [1.03%, 2.58%]）。

**分母警告**：论文历史 run 报的是 **6–9/540**，这里 n=1108。
**绝对计数不可并排比，只能比率与趋势。**

| 走势 | §5.9 怎么改 | 还要不要补跑 |
|---|---|---|
| **平**（留在区间内） | sufficiency 主张**保留并加强**：从「一条 ReAct 臂，同时改了协议与接口」升级为「外加一条单变量臂」。同时必须写明功效边界——平的结果排除大效应，不排除小效应 | 不用 |
| **上升**（>2.6%，即 ≥29/1108） | **撤回** sufficiency 主张。语气沿用撤回 p=0.648 那处：干脆、不辩解、不铺垫 | **要**。broken 无救回数，归因不干净。补跑 broken 带评测，6.2h + 42min |
| 崩在中途 | 用已落盘的评测点看趋势（结果逐步落盘，崩了不亏） | 视点数而定 |

**两个分支共同**：必须写清两臂都用 LoRA（7B 全参不进单卡 80 GB），
配置完全相同，**但与 150 步全参历史基线不严格可比**。

---

## 三、产物位置

```
服务器 /root/p3_formal/{broken,repaired}/events/summary.json   机制计数
       /root/autodl-tmp/runs/code-eval-p3-{arm}/step_*.jsonl    逐题评测（新路径，分臂）
本地   p3/results/                                              回传后
       p3/results/code-eval-p3-repaired-ABORTED-2008/           20:08 那次中断的 step0
```

**注意**：`code_patch.py` 的默认输出路径 `/root/autodl-tmp/runs/code-eval` 是**共享的**，
2026-09-03 已因此覆盖过一次历史数据。`run_p3_arm.sh` 现在设 `CODE_EVAL_OUT` 分臂，
但任何绕过该脚本的调用仍会写共享路径。

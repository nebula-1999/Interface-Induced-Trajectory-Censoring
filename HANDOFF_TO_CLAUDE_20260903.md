# P3 交接（→ Claude），2026-09-03 19:30

**任务**：broken-FC vs repaired-FC 的 RL 单变量对照，两条臂只差
`actor_rollout_ref.rollout.multi_turn.format` 一个键。完整方案见
`p6/PLAN_P3.md`；上一版交接见 `HANDOFF_P3_20260903.md`（Codex）与
`p3/STATUS_20260903.md`（Bro）。本文件只写**接手现在需要的东西**，不重复历史。

---

## 0. 一句话状态（19:30）

**正式两臂训练进行中。** broken 臂 148/150（99%，约 5 分钟后跑完）；
repaired 臂由本机 watch_p3.sh 自动接力（约 9 小时）。两笔「开跑前欠账」已核实
**都还清了**，broken 臂的数据是干净的。

---

## 1. 你上次提的三点，我逐条核实过

**① 是谁的 run？** —— 是 Bro（我）起的，单进程干净。证据：broken_driver.log 里
「数据自检」「解释器」各只出现 1 次；launch_ppo 进程 pid 25319 启动于 13:11。
watch_p3.sh 从 13:24 起一直是**观察**它（日志「运行中，本轮不动」），没起过
competing run。

**② 两笔欠账？** —— 都还了，且是在**这条 run** 里还的（服务器代码与本地
`e5613a9` 逐字节 diff 一致）：
- 验证器：`sandbox.py` 的 `_SUMMARY_LINE` 锚定 pytest 摘要行，`total` 分母含
  `skipped/xfailed/xpassed`。
- rollout 全文：`p3/rollout_probe.py` 存完整 `text` + sha256，实测事件里
  `text_chars` 最大 12943、30 条超 4000 字符，无截断。
→ 「零执行 ≠ 零尝试」现在区分得了。§7 item 11 的欠账已补上。

**③ 论文 14→9 页 + 分支改稿？** —— 未做，这是你接手后的主任务。改哪一节见
`p3/PAPER_EDIT_MAP.md` 与 `HANDOFF_P3_20260903.md` 第 5 节。

---

## 2. broken 臂半程数据（148/150 步，最终以 summary 为准）

| 指标 | 数值 |
|---|---|
| extract 检查 | 19063 |
| 语义完整调用 `emitted_tight` | 8528（**下界**，正则保守） |
| **hermes 接受** | **7**（不是 0，见下方 ⚠） |
| 工具执行 / `_call_tool` | 7 / 7 |
| `num_turns` | 恒 2.0 |
| reward | 前后 1/4 段均值 0.312 → 0.287（Welch t=-1.61，不显著） |

**⚠ 措辞要点（对论文很关键）**：broken 臂**不是绝对零执行**。模型在 150 步里
有约 7 次吐出了 `<tool_call>` 信封，hermes 正确解析并执行了这 7 次。所以论文
不能写「hermes 接受 0」，要写成「**8528 条合规调用里接受 7 条（0.08%）**」，
或等价地「99.9%+ 的合规调用被丢弃」。这与全文「量化截断、不绝对化」的纪律一致，
别重蹈 §5.8 早期「never」的措辞覆辙。

repaired 臂冒烟（单步）：140 emitted → 139 accepted → 139 executed（99%），
且已出现真实多轮修复。详见 `p3/STATUS_20260903.md` 第三节。

---

## 3. 基础设施（接手者要知道的）

- **接力是自动的**：本机 `p3/watch_p3.sh` 被定时任务「P3 两臂训练监控与自动接力」
  （id `5d108503`，每小时）调起。它靠 `/root/p3_formal/{arm}.launched` marker
  防重复启动：broken 跑完（`summary.json` 落盘）→ 自动 `launch_arm repaired` →
  把每步 CSV + summary 回传 GitHub（`p3/results/*_steps.csv`）。
  **不要再加第二个看门狗**（Bro 曾误建一个，已移除，会重复启动 repaired）。
- **判分支**：用 `p3/parse_steps.py <driver.log> --arm X`。它报告前后 1/4 段的
  `score / turns_mean / tool_calls_time` 差（Welch t + 效应量），判据见脚本头注释。
- **产物**：服务器 `/root/p3_formal/{broken,repaired}/`（events + driver log），
  回传后本地 `p3/results/`。
- **改稿地图**：`p3/PAPER_EDIT_MAP.md`（分支 A 撤回 / 分支 B 升级，逐节列出）。

---

## 4. 你接下来要做的（按优先级）

1. **等两臂结果**（broken 19:36 完 → repaired 约次日 04:30 完）。
2. 跑 `p3/parse_steps.py` 判分支 A/B，读 `p3/PAPER_EDIT_MAP.md` 逐节改论文。
3. **9 页压缩**（正文现 14 页）——这是零算力的活，可以趁 repaired 跑批时并行做。
4. Overleaf 重编译（今天动了 §5.2/§5.6/§7/摘要/标题，一次都没编过）。
5. 你自己的未提交改动：`writeup/PROJECT_RETROSPECTIVE.md`、
   `writeup/interview_narrative.md` 两个文件有未提交修改，**我特意没替你提交**，
   留给你自己收尾。

---

## 5. 关键数字（别重算）

- repaired 臂 **222 秒/步**（update_actor 125s 是大头），150 步 ≈ 9.2h。
- broken 臂 147 秒/步，150 步 ≈ 6.2h（已接近尾声）。
- 显存：actor 峰值 28.7 GB + vLLM 预留 44 GB ≈ 79 GB（A800 80 GB）。
  **没有余量提高 `ppo_micro_batch_size_per_gpu`**（现为 1，硬提必 OOM）。
- 磁盘：数据盘剩 19 GB；`save_freq=-1`（不存权重），本实验要训练动力学不要权重。
- 模型/数据/种子/LoRA/奖励：两臂逐字一致，只有 `multi_turn.format` 不同。

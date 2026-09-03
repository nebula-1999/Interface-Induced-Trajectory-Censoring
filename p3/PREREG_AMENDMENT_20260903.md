# P3 预注册修正：这条 run 能回答什么、不能回答什么

**写于 2026-09-03 19:40 —— broken 臂已完成（summary 已落盘），repaired 臂尚未启动。
在看到 repaired 任何结果之前提交。**

理由和全文其余预注册一致：判据必须在数据之前定。这一条尤其重要，因为
`p3/parse_steps.py` 现有的分支判据**会重蹈本文开篇批评的那个错误**。

---

## 一、问题：分支判据用错了量

`parse_steps.py` 头注释里的判据是：

> · `tool_calls_time > 0` 且 **score 上升** → 通道打开后学会了多轮修复（分支 A）
> · `tool_calls_time > 0` 但 score 不上 → 通道打开但没学会（分支 B）

这里的 `score` 是 `critic/score/mean`——**训练侧的聚合奖励**。

但 §5.9 的 sufficiency 主张不是关于聚合奖励的。它的原话依据是：

> 「rescues by turn ≥2 over steps 0/15/30/45/60 were 8, 6, 6, 8, 6」——
> **保留集 540 题上、按评估 checkpoint 计的多轮救回数。**

**这两个量不是一回事，而且本文第 1 节的全部意义就在于它们不是一回事。**
历史那三条 run 的核心观察正是：pass@1 涨了 2.6–2.8 分，而多轮救回数
纹丝不动——**聚合分可以完全由首轮质量提升驱动**，91–94% 的新通过题是首轮过的。

所以：**若因 `critic/score/mean` 上升而宣布分支 A（撤回 sufficiency 主张），
就是用本文自己证明为混淆的量去推翻本文自己的结论。**

## 二、这条 run 的配置决定了 sufficiency 不可测

`p3/run_p3_probe.sh`：

```
trainer.val_before_train=False
trainer.test_freq=-1
trainer.save_freq=-1
```

**全程不做验证、不存权重。** 因此：

- 保留集救回数在训练中**没有被测过**；
- 没有 checkpoint，**事后也补不回来**。

`p3/rollout_probe.py` 记录的是逐次生成的解析/执行事件（`kind`、名字、
`text_chars`、sha256），**不含每条 rollout 的最终奖励与轮次归属**，
所以也无法从事件文件里事后算出"首轮失败、后续轮次救回"的比例。

## 三、为什么不给 repaired 臂补上验证

补 `test_freq` 只能补给尚未启动的 repaired 臂。但 broken 臂是在
**没有验证**的配置下跑完的，只给一条臂加验证会**破坏单变量设计**——
而单变量正是 P3 存在的全部理由（§7 item 8 要的就是这个）。

要有验证，两条臂都得重跑（约 15 GPU 小时）。在当前预算下不做。

**结论：保持两臂配置逐字一致，接受 sufficiency 不可测。**

## 四、因此，P3 能主张与不能主张的（在看结果之前定）

**能主张（机制，单变量）：**
- 在训练循环内部，只改 `multi_turn.format` 一个键，工具通道是否打开。
- broken 臂已测得：8591 条语义完整调用中，**20 条带 `<tool_call>` 信封、
  7 条被 hermes 接受、7 条执行**（0.08%）。那 7 次执行是**这条 run 内部的
  阳性对照**——解析器与执行器都证明可用，所以其余 8584 条不是管线故障。
- 这正是 §7 item 8 点名要的 parser-repair 单变量对照，它现在存在了。

**不能主张：**
- **不能**因 `critic/score/mean` 的走势而撤回或加强 §5.9 的 sufficiency 主张。
  该主张关于保留集救回数，本 run 未测该量。
- **不能**把 `num_turns` 上升读作"学会了多轮修复"——它只说明工具被调用得更频繁，
  不说明后续轮次修好了任何东西。

## 五、对 `PAPER_EDIT_MAP.md` 的修正

原改稿地图的分支 A/B 框架**按本 run 的数据无法触发**：

| 原分支 | 状态 |
|---|---|
| A：repaired 恢复了多轮学习 → 撤回 sufficiency | **本 run 无法判定**。没有该量的测量 |
| B：未恢复 → sufficiency 从一条臂升级为两条臂 | **本 run 无法判定**。同上，第二条臂没测同一个量 |

**实际应走的第三条路（分支 C）**：
- §5.9 的 sufficiency 主张 **既不撤回也不加强**，原样保留，依据仍是那条 ReAct 臂。
- §5.8「Why no repaired-FC arm exists yet」**改写**：该对照现已存在，
  并给出两臂的机制数字（broken 7/8591 vs repaired 的对应数字）。
- §7 item 8 从「没有干净的 repaired-FC 训练对照」改为
  「该对照现已存在，确立了 X；但它未测保留集救回数，故不改变 item 12」。
- §7 item 12（"Opening the channel is tested only through ReAct"）**保留不变**，
  并补一句说明 P3 为何没有取代它。

## 六、下一次要怎么做才能回答 sufficiency

两臂都开 `test_freq`（与历史 run 同一个评估钩子、同一个 540 题保留集），
或者在 rollout 探针里补记每条 rollout 的 (轮次, 最终奖励)，
使"首轮失败→后续救回"的比例可以逐步计算。二选一，但必须**两臂同时**做。

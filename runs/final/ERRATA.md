# 数据勘误（必须与轨迹文件一同阅读）

## 1. 七个 ReAct 臂的 `max_tokens` provenance 记录错误

以下文件的记录值为 `2048`，**实际生成上限是 1024**。原因：`probe_react_full.py`
的 `gen()` 默认参数长期为 `max_tokens=1024`，而 `gen_fc()` 用全局 `MAX_TOKENS=2048`；
落盘的 provenance 统一写 `MAX_TOKENS`，因此 ReAct 侧记录与实际不符。
2026-08-31 11:20 修复（`gen` 默认改为 `MAX_TOKENS`）。

```
traj_v3_Llama8B_react.jsonl        记 2048 / 实 1024
traj_v3_Qwen7B_react.jsonl         记 2048 / 实 1024
traj_v3_Mistral7B_react.jsonl      记 2048 / 实 1024
traj_v11_DS1.3b_react.jsonl        记 2048 / 实 1024
traj_v11_DS6.7b_react.jsonl        记 2048 / 实 1024
traj_v11_Llama32_1B_react.jsonl    记 2048 / 实 1024
traj_v11_Llama32_3B_react.jsonl    记 2048 / 实 1024
```

**影响方向**：同期 FC 臂均为真实 2048，故此前所有 ReAct vs FC 对比中，
**FC 一方获得更长的生成预算**。ReAct 仍在这些对比中胜出，因此已报告的协议差距
是**保守估计**；修正后只会扩大，不会缩小。

**替代数据**：
- v11 四臂 → 由 `traj_v13_*_react.jsonl` 取代（真实 2048）
- v3 三臂 → 由 `traj_v14_*_react.jsonl` 取代（真实 2048）

引用时一律使用 v13 / v14 版本；v3 / v11 的 ReAct 臂仅作历史记录保留。

## 2. Llama-3.1-8B FC 的 L1 需人工更正

`traj_v3_Llama8B_fc.jsonl` 记录的 L1（发起率）为 97/100，但其中 **23 条实为调用了
题目函数本身**而非 `run_tests`。该批数据产生于加入函数名校验之前，`has_action`
未区分工具名。**更正后真实 L1 = 74/100**。
（`traj_v8_Llama8B_recheck.jsonl` 以相同配置重跑这 23 题，23/23 复现为 `wrong_tool`，
工具名为 `can_form_word`、`check_password_strength` 等题目函数。）

同批次的 `traj_v3_Qwen7B_fc_nosuffix.jsonl` L1=0，无需更正；
`traj_v3_Mistral7B_fc.jsonl` L1=2，该 2 条未逐条复核工具名。

## 3. 不可用于通过率比较的臂

```
traj_v5_Mistral7B_fc_official.jsonl        n_err=42  （rc=2）
traj_v9_Mistral7B_fc_official_strict.jsonl n_err=39  （rc=2）
traj_v9_Mistral7B_fc_strict.jsonl          n_err=3   （rc=2）
traj_v3_Mistral7B_fc.jsonl                 n_err=2   （rc=2）
traj_v11_Llama32_3B_fc_strict.jsonl        rc=2
```
上述臂的**错误率普查有效**（那正是被研究的现象），
但**通过率不可比**（存在缺失数据）。

## 4. provenance schema 跨代漂移

字段随开发逐步增加，共四代 schema：早期臂缺 `seed` / `temperature` / `fc_schema`。
关键字段（`model` / `protocol` / `adapter` / `max_tokens` / `clean_index`）全代齐备。
`script_sha256` 字段始终未成功加入（补丁两次被中断），脚本一致性改由外部
`v13_pinned_hashes.txt` 与验证器比对保证。

## 5. 已核验为正确的事项

- **37 个正式臂共用同一个 100 题集合**（`clean[:100]`，唯一集合数 = 1），配对有效
- 所有 FC 臂的 `max_tokens` 确为 2048
- A9 的采样确实生效：temp 0.6 下两个 seed 间首轮代码 95/100 不同、最终结果 16/100 翻转

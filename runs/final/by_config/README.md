# 按配置分目录（硬链接，原文件仍在 `runs/final/` 根下）

引用数据前请先读 [`../ERRATA.md`](../ERRATA.md)。

| 目录 | 配置特征 | 可用性 |
|---|---|---|
| `v2_underconfigured/` | vLLM 缺 `--enable-auto-tool-choice` 或缺家族 parser | **仅作历史记录**，勿用于结论 |
| `v3_v5_baseline/` | parser 配对、`max_tokens` FC=2048 而 **ReAct 实为 1024** | FC 臂有效；ReAct 臂已由 v14 取代 |
| `v6_v9_official/` | 官方 chat template、`strict:true`、专用适配器 | 有效（Mistral 各臂含请求错误，见 ERRATA §3） |
| `v11_v14_unified/` | **真实统一 2048**、意图检测器、工具名校验齐备 | **主表引用的就是这批** |
| `training/` | verl 训练日志（FC / ReAct rollout 对照） | 有效 |
| `smoke/` | n=3 冒烟，仅用于启动前门槛检查 | **不得用于任何结论** |

## 主表取数来源

```
ReAct 臂  →  v11_v14_unified/traj_v14_{Llama8B,Qwen7B,Mistral7B}_react.jsonl
FC 臂     →  v6_v9_official/traj_v6_Llama8B_fc_strict.jsonl
             v6_v9_official/traj_v6b_Qwen7B_fc_plugin.jsonl
             v6_v9_official/traj_v9_Mistral7B_fc_strict.jsonl（**含 3 条请求错误，通过率不可比**）
规模曲线  →  v3_v5_baseline/traj_v5_Qwen{1.5B,3B,7B,14B,32B}_fc_intent.jsonl
```

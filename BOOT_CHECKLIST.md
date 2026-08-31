# 开机清单（下次 GPU 启动时按序执行）

机器：`autodl-code`（单卡 A800 80G，数据盘 4.6 GB 可用，**不要再下模型**）

---

## 0. 先拉数据（2 分钟，最先做，防止后续实验覆盖）

**目的两个**：补齐本地缺口，并让仓库达到"可上传 GitHub、可复现"的状态。

本地已核对齐全的：全部 19 个复现链条文件、插件三件套（含 `.orig`）、
49 个正式臂轨迹、35 个 step 评测产物、v7 三份训练日志。

**仍缺、必须拉的**：

```bash
D="$HOME/RL Project/code_agent"

# ① 三个 150 步 run 的原始训练日志（含 loss/entropy/kl/grad_norm 与 resolved config）
rsync -az --include='full.log' --include='full_seed1.log' --include='full_rloo.log' \
  --exclude='*' autodl-code:/root/autodl-tmp/code-agent/ "$D/runs/final/"

# ② flash_attn_shim（verl 的 log-prob 计算硬依赖，复现训练必需）
rsync -az autodl-code:/root/autodl-tmp/code-agent/flash_attn_shim "$D/"

# ③ 训练数据 parquet（由 prepare_code_data.py 生成，但含 fc2/fc3 实验变体）
rsync -az --include='*.parquet' --exclude='*' \
  autodl-code:/root/autodl-tmp/train/code-data/ "$D/train/code-data/"

# ④ 服务器上的一次性分析脚本（若要完整审计轨迹）
rsync -az --include='diag26.py' --include='tight_curve.py' --include='qwen_attempt.py' \
  --include='echo_check.py' --include='intent_split.py' --include='verify_sys.py' \
  --exclude='*' autodl-code:/root/autodl-tmp/code-agent/ "$D/analysis/server_scripts/"

python3 analysis/training_curves.py     # 拉回后立即出图，见 §3
```

**不要拉**：`*.bak_*`（10 个）、`fix_*.py` / `add_*.py`（14 个补丁脚本，
其效果已合并进主文件）、`vllm_*.log`（服务日志，体积大且无分析价值）。

## 1. VAL-02 离线 hermes 对照（5 分钟，零训练）

重解析矩阵的 hermes 列全为 0，需要一个"会亮"的对照，否则可读成
"你的离线 hermes 实现有 bug"。二选一：

```bash
# 选项 1（首选）：直接 import vLLM 自己的 parser，而不是重写一份
ssh autodl-code 'cd /root/autodl-tmp/code-agent && source /root/autodl-tmp/env.sh && python - <<PY
from vllm.tool_parsers import ToolParserManager
print(sorted(ToolParserManager.tool_parsers.keys())[:10])
PY'
# 然后用真实 Hermes2ProToolParser 离线跑一遍 traj_v5_Qwen32B_fc_intent.jsonl 的 raw_output
```

选项 2（若 import 路径不通）：起 Qwen2.5-**Instruct**（在 hermes 脚手架上训过）
跑 n=100，该列应当非零 —— 但这需要模型，磁盘只剩 4.6 GB，先确认本地有权重。

## 2. TRAIN-01 修好接口后的 150 步（唯一昂贵项，约 3 小时）

两位审稿人一致认定的阻断项：目前只证明"坏的还是坏的"，
没证明"修好就能学会"。

```bash
scp fixes/run_train01.sh autodl-code:/root/autodl-tmp/code-agent/
ssh autodl-code 'cd /root/autodl-tmp/code-agent && setsid --fork ./run_train01.sh </dev/null >train01_run.log 2>&1'
```

验收：多轮救回是否从 6–9/540 的横盘抬起来。
**单 seed 也行——有方向远好过没有。**

## 3. 收尾

```bash
# 拉回全部产物
rsync -az --include='traj_*.jsonl' --include='train*.log' --include='*_manifest.txt' \
  --exclude='*' autodl-code:/root/autodl-tmp/code-agent/ "$D/"
python3 validate_arms.py           # 逐臂验收
python3 analysis/training_curves.py
ssh autodl-code 'shutdown -h now'  # 关机
```

## 不要做的

- 不要再下载任何模型（磁盘 4.6 GB）
- 不要在实验运行期间修改 `probe_react_full.py`（provenance 会不一致）
- EXP-02（14B/32B 适配器）已降级为可选，算力优先给 TRAIN-01

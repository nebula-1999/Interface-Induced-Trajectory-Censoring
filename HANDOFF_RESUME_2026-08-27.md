# Code Agent GPU handoff — 2026-08-27（resume，卡已开）

上一份 `HANDOFF_2026-08-27.md` 的停点之后，`verify_evalplus` 的障碍已诊断并收尾。
本文件是当前最新状态和接续顺序。**照着做即可，不用重新排查。**

---

## 服务器当前状态

- 实例：`1× NVIDIA A800 80GB PCIe`，**卡已开**，GPU 空闲（0%/0 MiB），按小时计费。
- SSH 别名 `autodl-code` → 端口 `56557`（当前可用）。项目在 `/root/autodl-tmp/code-agent`。
- 开卡后 cgroup 内存上限 **120 GiB**、112 核（无卡模式才只有 2 GiB，别再用那个数字）。
- **尚未启动任何训练。** 模型两条已下载：
  `/root/autodl-tmp/models/Qwen2.5-Coder-1.5B-Instruct` 与 `Qwen2.5-1.5B-Instruct`（各约 2.9 GB）。
- `verified_ids.json`：**7414** 条可用训练数据（要求 ≥2600）。
- `probes_repair.jsonl`：**456** 条修复探针（要求 ≥450）。
- 37 个测试全过（`test_sandbox.py` / `test_code_tool_core.py` / `test_eval_decompose.py`）。
- venv：`/root/code-venv`；`env.sh` 已把各 cache 指到数据盘。

## EvalPlus 验证 —— 已解决（原先的阻塞点）

**结果：官方解 540/542 通过。** 剩余 2 条为**已确认的 EvalPlus 数据缺陷**，不是我们拼错：

| 题 | 性质 | 依据 |
|---|---|---|
| `HumanEval/32` | test 断言本身写错 | 末尾 `_poly(*candidate(*inp), inp)<=0.0001` 把 float 用 `*` 展开（TypeError）且参数顺序反了；正确应为 `_poly(inp, candidate(*inp))`。官方解 `find_zero` 数学正确（能收敛到测试给出的预期根，`abs(poly(xs,root))<=1e-4` 通过）。 |
| `Mbpp/599` | 测试输入对参考解不可行 | 参考解是 `sum(range(1,n+1))`，但生成测试含 n≈5e11 的输入，参考解自身要几十亿次迭代，4 GB / 30 s 仍是 timeout。 |

已应用并同步的修复（**只改资源上限，不改基准语义**）：
- `sandbox.py`：`MEM_MB` 512 → **4096**。原因：`Mbpp/255` 的参考解需要约 3 GB，512 MB 会让它误报 `MemoryError`（旧 `evalplus_bad.json` 里的第 3 条现已消掉）。

## 待确认的方法学决定（建议用户/Claude 拍板）

**建议把评测集定为 540 条**（剔除 `HumanEval/32` + `Mbpp/599`，并在报告披露这两条缺陷）。
- 否则评测集仍是 542，但这 2 题对任何模型（包括满分参考解）都必然算失败，绝对 pass@1 会和公开 EvalPlus 口径不可比。
- 对项目要做的**同 set 归因**（RL vs base，同一评测集）而言，用 542 也能做，只是有 2 分的恒定天花板。
- 若剔除，N 从 542 → 540，需重算 Step 0 的 CI/MDD 表（MDD≈6.0 基本不变，仍是唯一可用配置）。
- **不要宣称 542/542。** 权威缺陷清单见 `evalplus_bad.json`。

## 后续顺序（这些需要 GPU，卡现在是开的）

1. ~~验证 EvalPlus 542/542~~ → 改为 **540/540 可验证 + 2 条已确认缺陷**（已完成）。
2. **装依赖栈**（1-1.5 h，唯一必须开卡装的东西）
   - 先 `pip install --dry-run` 看 vllm 会不会把 torch 降级（旧实例踩过）。
   - 装完 `pip cache purge`（系统盘只有 30 GB）。
   - 复现版本：Python 3.12 / torch `2.13.0+cu130` / vLLM `0.27.1` / verl `0.9.0` / transformers `5.10.4` / ray `2.57.0` / datasets `5.0.1`。
3. **生成训练数据**（5 min）：`python prepare_code_data.py --out-dir /root/autodl-tmp/train/code-data`。
4. **冒烟**（0.5 h）：`./run_code_grpo.sh smoke`（只跑 20 步），确认 vLLM colocate、工具调用、显存、**奖励不是全 0**。
5. **baseline**（1.5 h）：两个候选模型各跑一次 `CodeEvalHook.run(step=0)`，看 `turn1_pass` 与 `repair_rate`，选“留下足够提升空间”的 base（太强会撞天花板，6 点止损线就涨不出来）。
6. **冻结超参**写进 `configs/`，之后除 seed 外不动。
7. **Step 3 正式 150 步 run** 单独开卡跑（进 Step 3 前先确认单卡 A800 单价，预算 ≤¥200）。

## 预算 / 时点提醒

- Step 2（装栈 + 冒烟 + baseline）约 4 h 约 ¥20-25；Step 3 约 12 h 约 ¥60；整体计划约 ¥145（单卡 A800）。
- **卡现在是开的且空闲。** 如果 Claude 额度要等一会儿，建议先把卡关掉（模型和 data disk 都会保留，重开即可），避免空转计费；等要跑装栈/冒烟/训练时再开。

## 已踩过、别再踩的坑

- **`calc_reward` 在 verl 0.9.0 是死接口。** 奖励必须由 `reward_code.py` 的 `custom_reward_function` 给出——放在 `CodeTool.calc_reward` 里会让训练全程拿 0 分，而日志看不出异常。
- `trainer.save_freq=-1`，靠 `probe_hook.py` 就地评测拿曲线，最后只存一次推理权重（数据盘 50 GB）。
- `HF_HUB_DISABLE_XET=1`（`env.sh` 里已有，不设会 401）。
- **BLAS 线程数钉成 1**（`OPENBLAS/OMP/MKL/NUMEXPR/VECLIB` threads=1），否则 112 核 OpenBLAS 一 `import numpy` 就 `Memory allocation still failed`，全部 numpy 题失败。
- KodCode 测试有两种风格（带/不带 `from solution import`）；**EvalPlus 不是 pytest**，走 `mode="script"` 只看退出码，严格 0/1。别混。

## 本地与服务器同步

- 本地项目：`/Users/wangwenbo/RL Project/code_agent/`；服务器：`/root/autodl-tmp/code-agent/`。
- 本文件 `HANDOFF_RESUME_2026-08-27.md` 和 `OPEN_GPU_CHECKLIST.md` 本地与服务器各留一份。

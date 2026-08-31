# 开卡执行清单

**开着卡现写代码是这个项目里最贵的花钱方式。** 下面每一步都只是执行，
不含任何需要现场编写的东西。预算 ≈ 4 h ≈ ¥20–25。

## 开卡前（无卡模式确认，全部 ¥0）

- [ ] `cat runall.done` 有 kodcode / evalplus / probes 三条完成记录
- [ ] `verified_ids.json` 条数 ≥ 2600（训练需要 2400，低于这个数要回头看）
- [ ] `probes_repair.jsonl` 条数 ≥ 450（542 题里注入成功的）
- [ ] `verify_evalplus` 报官方解 **540/540 可验证通过**（540/542；剩余
      `HumanEval/32` 与 `Mbpp/599` 为**已确认的 EvalPlus 数据缺陷**，
      见 `evalplus_bad.json`。不要宣称 542/542。）
- [ ] 37 个测试全过：`pytest test_sandbox.py test_code_tool_core.py test_eval_decompose.py`

> 方法学决定待拍板：评测集建议定为 **540 条**（剔除上述 2 条缺陷并披露），
> 需重算 Step 0 的 CI 表（N 542→540，MDD≈6.0 基本不变）。若坚持 542，
> 2 条缺陷对任何模型都必然算失败，绝对 pass@1 与公开口径不可比。

## 开卡后按顺序执行

1. **装栈**（1–1.5 h，唯一必须开卡装的东西）
   - 先 `pip install --dry-run` 看 vllm 会不会把 torch 降级——旧实例踩过
   - 装完 `pip cache purge`（系统盘只有 30 GB）
2. **生成训练数据**（5 min）
   `python prepare_code_data.py --out-dir /root/autodl-tmp/train/code-data`
3. **下模型**（15 min）
   两个候选：`Qwen2.5-Coder-1.5B-Instruct` 与 `Qwen2.5-1.5B-Instruct`
   （`HF_HUB_DISABLE_XET=1` 已在 env.sh 里，不设会 401）
4. **冒烟**（0.5 h）
   - ⚠️ verl 默认往 **wandb** 打日志，没配 API key 会直接
     `UsageError: No API key configured` 崩在启动阶段（2026-08-27 实测）。
     已在 `run_code_grpo.sh` 里钉死 `trainer.logger='[console]'`。
   `./run_code_grpo.sh smoke` —— 只跑 20 步，验证 vLLM colocate、
   工具调用、奖励函数、显存都正常
   - **重点看奖励不是全 0**。`calc_reward` 在 verl 0.9.0 里是死接口，
     奖励全靠 `reward_code.py`，这一条错了整个 run 白跑而且日志看不出来
5. **baseline**（1.5 h）
   两个候选模型各跑一次 `CodeEvalHook.run(step=0)`，看 `turn1_pass` 与
   `repair_rate` 定 base model。
   **选择标准是"谁留下足够的提升空间"**——太强会撞天花板，
   6 个点的止损线就涨不出来
6. **冻结超参**写进 `configs/`，之后除了 seed 什么都不许动
7. **关卡**。Step 3 的 12 h 正式 run 单独开一次——
   Step 2 的产出正是用来决定 Step 3 怎么配的

## 止损

总 pass@1 提升 < 6 个点（542 题、80% power 下的 MDD）就不要加钱硬救，
转 fallback：对 `swe-bench/experiments` 公开的 per-instance 结果做配对检验
与 scaffold/model 归因拆分，零 GPU。

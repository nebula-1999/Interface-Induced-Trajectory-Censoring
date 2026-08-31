# 工具接口错配伪装成模型缺乏智能体能力

代码智能体的多轮 RL 项目，及其过程中发现的一类评测/训练污染。

**结论文档**：[`RESULTS.md`](RESULTS.md) → 详细证据 [`writeup/section_config.md`](writeup/section_config.md)
**数据勘误**：[`runs/final/ERRATA.md`](runs/final/ERRATA.md) —— **引用任何数字前请先读这份**
**面试叙述**：[`writeup/interview_narrative.md`](writeup/interview_narrative.md)

## 一句话

按 vLLM 官方文档为四个模型家族配置 function calling，**每个家族都会在不同层静默失败**，
表现为"模型不会用工具"。该错配同时污染评测数字与 RL 训练信号——在我们自己的训练中，
它使工具调用在梯度上不可达。

![意图-解析缺口](figures/fig1_intent_parse_gap.png)

## 目录

```
RESULTS.md                顶层结论（12 节）
writeup/
  section_config.md       完整证据与对照设计（11 节）
  interview_narrative.md  5 分钟叙述与追问弹药
analysis/intent.py        调用意图的**唯一**判据（正文/表/图共用）
figures/make_figs.py      三张主图
runs/final/
  traj_*.jsonl            58 个实验臂的逐题轨迹
  ERRATA.md               7 个 provenance 错误臂、5 个通过率不可比臂
  final_table.py          主表与方差表（含错误的臂自动标 N/A）
  validate_arms.py        逐臂严格验收
probe_react_full.py       协议探针（ReAct / FC，四种 schema 变体）
sandbox.py                受限沙箱（10s / 512MB / cgroup 感知并行度）
code_tool.py              verl 工具（FC 路径）
react_agent_loop.py       自定义 verl AgentLoop（ReAct 路径）
analyze_all.py            EvalPlus 训练结果的唯一来源
```

## 复现

分析与制图（**无需 GPU**，本地可跑）：

```bash
pip install -r requirements.txt
python analysis/intent.py          # 规模曲线的四档计数
python runs/final/final_table.py   # 主表 / 方差表 / 错误率表
python figures/make_figs.py        # 三张主图
```

探针（需要一台起着 vLLM 的机器）：

```bash
python probe_react_full.py --model <path> --port 8000 --n 100 \
  --protocol {react,fc} --strength {optional,mandatory} \
  --fc-schema {terse,rich,strict} --parser-adapter cross_family \
  --temperature 0.0 --seed 0 --out traj.jsonl
python validate_arms.py            # 逐臂验收：行数/rc/n_err/provenance/脚本 hash
```

## 已知限制

见 [`RESULTS.md` §10](RESULTS.md)。最重要的三条：单一题库、每臂 100 题且**非随机取样**；
主表为 `temperature=0` 单次采样；专用适配器同时更换 parser/模板/few-shot 三者，
只能称"适配器组合"。

**`sandbox.py` 依赖 Linux 的 `preexec_fn` 与 `RLIMIT_AS`，在 macOS 上单元测试会失败**；
沙箱相关功能需在 Linux 下运行。

## 第三方组件

`plugin/` 下的 Qwen2.5-Coder `<tools>` parser 来自
[hanXen/vllm-qwen2.5-coder-tool-parser](https://github.com/hanXen/vllm-qwen2.5-coder-tool-parser)
（Apache 2.0）。其 chat template 的 few-shot 示例原为 `get_weather`，与本实验唯一合法工具
`run_tests` 不符，会诱导模型调用不存在的工具，**已改写**；原件保留为 `.orig`，
两版 SHA256 见 `writeup/section_config.md` §8。

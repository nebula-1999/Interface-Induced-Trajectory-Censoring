# 仓库整理任务（→ Codex）

**结论先行：不要新建仓库，不要全量翻译。** 理由在下面，请先读完第一节再动手。

---

## 一、两条硬约束（违反任一，本次整理判定失败）

### 1. 不新建仓库，不改写 git 历史

论文（已投 arXiv）多处写着预注册是 **"committed to the repository before the run"**。
那个「之前」**只能靠 git 提交时间戳证明**——`p4/PREREGISTRATION.md` 的提交时间早于
Qwen3 反事实梯子开跑，这是论文可证伪性的凭据之一。

新建仓库会把所有 commit 变成同一天，**这个凭据就消失了**。
`git filter-branch` / `rebase` 改写历史同理。

论文首页已印上本仓库 URL：
`https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring`
换仓库还会让已发布的链接失效。

**→ 在现有仓库里原地整理，保留全部历史。**

### 2. 不翻译数据文件

含中文的 330 个跟踪文件里，**118 个是 `.jsonl` 轨迹数据**——里面是实际发给模型的
中文提示词和模型的中文输出。

**翻译它们等于篡改实验记录。** 这篇论文通篇讲的就是测量保真度，附录 C 明确写着
「翻译过的提示词是第三个提示词」。同理：

- 任何 `.jsonl` / `.json` 里的 `content`、`prompt`、`arguments` 字段：**一个字都不能动**
- `paper/sections/C_prompts.tex` 里的中文提示词原文：**不能动**（英文对照已补全）
- 任何记录「模型说了什么」的地方：**不能动**

---

## 二、真正要做的（按优先级，总量很小）

论文只点名了 4 个文件，读者会顺着链接去找的就是它们。

### P0 — `p4/PREREGISTRATION.md` 译成英文（最重要）

7364 字节里有 **1433 个中文字**，基本是全中文。论文把它当作**可证伪性的证据**引用，
审稿人点进来读不懂，这条证据就等于没有。

要求：
- 译文另存为 `p4/PREREGISTRATION.md`（英文），中文原件改名 `p4/PREREGISTRATION.zh.md` 保留
- **不要修改任何预测内容或日期**。这是预注册，改一个字都破坏它的性质
- 开头加一行：`This is a translation of PREREGISTRATION.zh.md, committed <原提交日期>. The predictions and dates are unchanged; see git history for provenance.`

### P1 — `preflight_toolcall.py` 注释译成英文

285 个中文字。论文摘要写着 "We release a 98-line preflight check that catches every
silent failure reported here"——这是全文唯一一个「拿来即用」的产物，读者一定会打开。

**只译注释和 docstring，不动一行代码逻辑。** 改完跑一次确认仍能运行。

### P2 — `analysis/intent.py` 注释译成英文

170 个中文字。论文称它是 "a single classifier shared by all text and tables"，按名字引用。
同样**只译注释**——这个文件的判据决定了论文里每一个 emitted-call 计数，
逻辑改动会让所有数字失效。

### P3 — 顶层目录整理

现在顶层 40+ 个文件，读者第一眼看到 5 个 HANDOFF、4 个 RESULTS 变体、若干 log。

移动（`git mv`，不要删）：
```
HANDOFF_*.md  BOOT_CHECKLIST.md  OPEN_GPU_CHECKLIST.md      → internal/handoffs/
RESULTS_v1_backup.md  RESULTS_v1_refuted_20260829.md         → internal/superseded/
baseline_*.log  *_run*.log                                    → internal/logs/
TASK_REPO_CLEANUP.md（本文件）                                 → internal/
```

保留在顶层：`README.md` `README_zh.md` `RESULTS.md` `LICENSE` `CITATION.cff`
`preflight_toolcall.py` `requirements.txt`，以及 `paper/` `analysis/` `p1/`…`p6/` 等目录。

### P4 — README 加一张「东西在哪」的表

现在的 README 已经是英文，内容也好，但缺一个让读者按图索骥的映射。加一节：

| 论文里的说法 | 仓库位置 |
|---|---|
| the intent criterion | `analysis/intent.py` |
| replaying vLLM's hermes extractor | `analysis/failure_layer.py` |
| the same bytes re-parsed under four rules | `analysis/reparse_matrix.py` |
| a prediction committed before the run | `p4/PREREGISTRATION.md` |
| a 98-line preflight check | `preflight_toolcall.py` |
| data errata | `runs/final/ERRATA.md` |
| BFCL 2×2 | `p1/` |
| τ-bench | `p5/` |
| the rollout-path probe | `p3/rollout_probe.py` |

---

## 三、明确**不要**做的

- ❌ 新建仓库 / 改写历史
- ❌ 翻译 `.jsonl` / `.json` 数据
- ❌ 翻译 `paper/` 下的任何内容（论文已定稿投出，附录 C 的双语是刻意设计）
- ❌ 翻译 `internal/` 下的工作笔记（复盘、交接、监控日志）——它们是诚实的实验室记录，
  读者不需要，翻译只会引入错误
- ❌ 批量翻译其余 140 个 `.py`/`.sh` 的注释。**低价值高风险**：
  这些脚本产出了论文里的每一个数字，为了注释去动它们不划算。
  真要做，等 P3 实验全部结束、不再有脚本在跑的时候
- ❌ `git rm` 任何东西。一律 `git mv`

---

## 四、验收

改完请自查并在 PR/commit 里报告：

1. `git log --follow p4/PREREGISTRATION.zh.md` 仍能看到原始提交日期
2. `python preflight_toolcall.py --help` 仍可运行
3. `python -c "import analysis.intent"` 无报错
4. `git ls-files '*.jsonl' | xargs grep -l '[一-鿿]' | wc -l` 结果**仍是 118**（数据一个没动）
5. 论文 PDF 里的仓库 URL 仍然可达

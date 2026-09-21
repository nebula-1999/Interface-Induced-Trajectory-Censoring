# v3 修改日志

## 2026-09-21 Astra 收尾修订

当前 ICLR 正文 8 页、全文 30 页；公开长版 42 页。统计改动、计数单位、
RL 评估边界及匿名包范围详见 `ASTRA_FINAL_CHECK_20260921.md`。
新增的 27 项检验清单是回顾性审计，替代历史“约十二项”口径；原实验结果未改。
旧的 Wald 区间改为保守有限样本区间。以下历史记录不是当前统计规则的替代来源。

## 2026-09-19 最终核对

- 当前页数以 `iclr2027/README.md` 为准：投稿版正文到第 8 页，全文 28 页；
  公开长版全文 40 页。下方 v2/v3 页数表保留为历史记录。
- Seed/Astra 意见的逐项处理见 `ASTRA_REVIEW_AUDIT_20260919.md`。
- P3 完整 checkpoint 曲线、恢复分支计数、LoRA 更新和逐请求执行/observation
  核验已补入附录 G；原始证据与可复算脚本随仓库保存。
- 修复配置表、分类表越界及图中遗留的 capability / well-formed 过强措辞。
  Errata 仅调整换行，未删除任何披露；学习和面试文档不包含在这次收尾提交中。

以下为历次结构性修订记录，非当前页数或最新统计口径的替代来源。

编号对应你的修改说明书。本地已用 XeLaTeX 实测页数（macOS 上 `brew install texlive`，
CJK 字体从 Noto 换成系统自带 Songti SC / PingFang SC——**这一处需要在 Overleaf 改回去**）。

---

## P-1 Conclusion 重复四遍 —— 已于本次之前修复

你看到的 33–34 页四份 Conclusion、每份结尾带字面量 `sectionConclusion`，
是 commit `83a281a` 的一次查找替换打断 `\section{Conclusion}` 留下的坏块
（212 词 → 620 词）。

**已在 commit `6383139` 修掉**，从 `9c12a5d` 恢复。现在 `08_conclusion.tex` 里
`sectionConclusion` 出现 0 次，`\section{Conclusion}` 出现 1 次。全部小节扫过同类
损坏（重复段 + 缺反斜杠的 section），无第二处。

**另外抓到一个同类缺陷（原稿就有，你的说明书没提）**：§7 的
`\setcounter{enumi}{13}` 使 **item 14 被印两次**——「verifier 两处缝隙」和
「serving-stack 版本依赖」都编号 14。实际是 **17 条，不是 16 条**。已改为
`{14}`，现在编号 1–17 无重复，**一条未删**。

---

## P0 repaired-FC —— 2026-09-18 已落地

P3 的 150-step 7B LoRA 单变量对照已经完成。两臂只差 verl parser registry：broken
臂 8,689 条 tight emission 中接受并执行 10 次；repaired 臂接受 16,912 次、执行并返回
observation 16,844 次。机制恢复，但 held-out rescues 12→12，final 430/542→427/542，
没有可测学习收益。摘要、§5 RL 主结果、§6 cold-start、§7 item 8/10/11/12/14、§8、
附录 D/G/H 已按这个可信 null 统一；所有 “没有 repaired-FC control” 与 broken=0 的
过时表述均撤回。

---

## P1 压缩

| | v2 | v3 |
|---|---|---|
| 正文（实测，含标题页） | **33 页** | **18 页** |
| 全文 | 41 页 | **38 页** |
| §5 Results | 18 页 | **8 页** |

移入附录（全部保留，无删除）：
- §3.2 人工验证全过程 → **附录 F**（两轮标注、instrument failure、adjudication 规则、κ 的三次移动、correction factor 的三个值）
- offline re-parse matrix（Table 6）、failure-layer 分解（Table 9）→ **附录 G.2**
- 四家族完整 taxonomy（Table 4）+ 四柱图 Figure 3 → **附录 G.3**
- 主表（Table 17）、Llama 四控制表（Table 14）、both-parsed 条件表、feedback access 表、pressure 表、variance、Jaccard/组成漂移 → **附录 G**
- §5.7.1 的 (i)(ii)(iii)(iv) 四层障碍 → **附录 D.4**
- verifier 审计细节 → **附录 G**（正文留 8 句摘要，两处缝隙、审计结论、审计自身边界都在正文）
- BFCL 门控与复现记录（197 vs 196、480 请求）→ **附录 G**
- 训练曲线图 Figure 1、intent-parse-gap 图、training-causal 图 → **附录 G**

正文只剩 **1 张图**（五层漏斗）和 **11 张表**。

手法：表 caption 全部砍成「这是什么」，论证回正文；正文粗体从 121 处降到 50 处
（表格内的关键单元格保留）；同一结论重复陈述处只留一次；§7 从 `enumerate`
改成行内 `\emph{(n)}`，省掉列表间距。

---

## P2 摘要重写

按你给的六个要点重写，364 → 316 词。BFCL 2×2 领衔，删掉了
「**The mismatch reaches RL training.**」那句旧措辞，改为按 scale 分开报
（7B 45/115→0/0/0；1.5B 承认 over-determined）。收尾句保留。

**新增一句**：τ-bench 的 0→636 / 0→103（见下）。

---

## P3 措辞

- 3.1 删掉全部元层面自评：§8 的「claimed us as one of its instances, twice」、
  §3.2 的「the annotation instrument reproduced the very failure this paper is about」
  及其展开、§5.4 末的同类句。**事实全留，自我评价全去**。
- 3.2 Qwen3 结论句改成你给的版本（"rules out capability growth as a *sufficient* cause;
  it does not isolate the envelope, since the ladder changes family and envelope together"）。
- 3.3 粗体减量：见上。
- 3.4 hedge 去重：每个 claim 正文留一处，其余靠 §7。

---

## 新数据：τ-bench 全量（今天跑完，v2 时还没有）

115 题 retail，两臂全部配对，无丢弃。**这是 §7「remains open for interactive
suites」那一条，现在有结果了。**

| | documented | repaired |
|---|---|---|
| assistant 轮次 | 1389 | 1971 |
| 服务端解析出调用 | **0** | **636** |
| 发出但未被解析 | **789** | 6 |
| 工具执行 / observation | **0** | **636** |
| 进入过工具循环的题 | **0** | **103** |
| 解出的题 | 7 | 10 |

**任务成功差不显著，没有主张**：3 个不一致对全在 repaired 方向，
exact McNemar *p* = 0.25，且 documented 的解题集是 repaired 的真子集。
用户模拟器是本地 Llama-3.1-8B 而非 gpt-4o，**绝对分不可与排行榜比**，
只主张臂间差——这两句和数字写在同一段，不是脚注。

---

## 数字守恒校验

脚本比对 v2 与 v3 全部 tex 的数字 token：
- **v3 里凭空出现的实验数字：0**（差异只有 `2.2`/`2.3` 两个 LaTeX 列宽和一个正则边界）
- **v3 里彻底消失的数字：`2023`**（albayaydh 综述的「2023–2026」年份区间，非实验数据）
- τ-bench 那一批是今天新跑出来的真实结果，是这条规则的唯一例外，已单列在上。

`\ref` 全部可解析（0 处未定义），无重复 label，brace/环境平衡。

---

## 顺带修掉的编译错误

`main.tex` preamble 缺 pandoc 的 `Shaded` / `Highlighting` / `*Tok` 定义，
**附录 B 一直编译不过**（`! LaTeX Error: Environment Shaded undefined.`）。已补进 preamble。

---

## 页数核算（2026-09-18，XeLaTeX 实测）

当前 arXiv 单栏版正文（含标题、摘要、Limitations 和 Conclusion）在第 **13** 页结束；
参考文献为第 13--15 页，全文连附录共 **38** 页。P3 落地后 §Results 在第 11 页结束，
新表没有越界或跨页，编译无 overfull box、未定义引用或缺字。

这不是 9 页 ICLR 投稿版。此前“ICLR 可以把超页正文交给附录”的表述不准确：投稿版
必须按当年模板和正文页数限制执行。当前单栏页数也不能直接换算为 ICLR 双栏页数，
应当另建 submission 入口、套官方模板后实测，而不是继续用几何估算。

压缩时保留四条正文主证据：BFCL 2x2、Qwen ladder + matched counterfactual、Llama
single-variable control、P3 formal RL control。τ-bench、Instruct ladder、taxonomy 和
historical ReAct 在正文各留一段结论，完整表格与审计记录留附录；17 条 limitation
正文保留最约束结论的六条，附录 H 继续保留全部内容。这样压缩的是重复解释和表体，
不是不利披露或 ERRATA。

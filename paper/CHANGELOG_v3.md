# v3 修改日志

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

## P0 repaired-FC —— 未落地，按你的指令只做机械压缩

P3（broken-FC vs repaired-FC）尚未开跑。§5.7.2、§7 item 8/12、§6 cold-start 段、
§8、摘要末段的**实质内容一律未动**，只压缩措辞。分支 A/B 的改写口子留在原处，
结果到手后一次性填。

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

## 页数核算（XeLaTeX 实测，非估算）

| 小节 | v2 实测 | v3 实测 | 你的预算 | 超支 |
|---|---|---|---|---|
| §1 Introduction（含漏斗图） | 3 | **2** | 1.0 | +1.0 |
| §2 Related Work | 2 | **1** | 0.6 | +0.4 |
| §3 Setup + 意图判据 | 3 | **1** | 0.5 | +0.5 |
| §4 Positive Control | 1 | **1** | 0.3 | +0.7 |
| §5 Results（11 表） | 18 | **8** | 5.5 | +2.5 |
| §6 Discussion | 1 | **1** | 0.6 | +0.4 |
| §7 Limitations（17 条 + What would change） | 3 | **2** | 0.35 | +1.65 |
| §8 Conclusion | 1 | **1** | 0.15 | +0.85 |
| **正文合计（含标题/摘要页）** | **33** | **18** | **9** | **+9** |

（页边界是整页粒度，逐节数相加会比合计少 1 页。）

### 为什么还差 9 页 —— 这是本轮唯一没做到的硬约束

你的预算表本身加起来正好 9.0，但它没有给标题+摘要留位（≈0.5 页），
也没给 11 张表的表体留位（实测约 1.6 页）。真正的结构性冲突是 §7：

**§7 预算 0.35 页 = 约 185 词。17 条限制每条压到 2 句，下限约 750 词 = 1.6 页。**
零节写明「可压缩每条到 2–3 句，**不可删条目**」，两条约束在同一个文档里互相排斥。
我按「不可删」执行，§7 现在 2 页。

其余超支集中在 §5。它承载 5 个实验、11 张表，8 页已经是每页 470 词的密度。
再压只能砍掉限定语——而限定语正是零节列的「所有对自己不利的披露」。

### 要拿回这 9 页，只有三个杠杆，全部触及保护清单

| 杠杆 | 省 | 代价 |
|---|---|---|
| **A. §7 全部 17 条移附录**，正文留 300 词摘要 | −1.5 页 | 一条不删，但正文不再有 Limitations 章。你的保护清单和 Appendix A（Errata）先例都支持附录放置，但审稿人确实会看正文有没有 |
| **B. §5 再移 4 张表**（bfclfunnel / counterfactual / repair / rolloutprobe），只留结论句 | −1.2 页 | counterfactual 和 rolloutprobe 是你自己的正文保留清单里的头号证据 |
| **C. §5 删掉全部 scope/bound 限定语** | −2.5 页 | 直接违反零节 |

A+B 能到 **约 15 页**。要到 9 页必须动 C。

**我的建议**：不要为 9 页动 C。ICLR 允许正文超页时用附录承接，而这篇论文的
分数是靠诚实拿到的——把限定语砍掉换页数，是把 6 分的来源换成 5 分的毛病。
先执行 A（可逆，单文件），再看 §5 能不能靠改写而不是删证据再省 1 页。
A 要不要做，你说一声我立刻执行。

# Overleaf 编译说明

## 上传

把本目录整个打包上传（或直接拖入 Overleaf 新项目）：

```
main.tex          主文件
refs.bib          参考文献
sections/*.tex    11 个分节
*.png             5 张图
```

`convert.py` / `fixtex.py` 是从 Markdown 重生成用的工具，**不用上传**。

## 编译器

**必须选 XeLaTeX**（Overleaf: Menu → Compiler → XeLaTeX）。

原因：正文含 `→ × ∅ ⊂` 等符号；§4 的阳性对照里保留了模型的真实中文输出（那是数据，
不能改写）；附录 A 的勘误目前是中文。pdfLaTeX 会在这些地方报错。

字体用 Overleaf 自带的 `Noto Serif CJK SC` / `Noto Sans Mono CJK SC`。
若报字体缺失，改成 `\setCJKmainfont{FandolSong}`。

## 首次编译大概率要修的

1. **参考文献**：需编译两轮（XeLaTeX → BibTeX → XeLaTeX ×2）。Overleaf 自动处理，
   但第一次可能显示 `[?]`，再点一次 Recompile 即可。
2. **表格过宽**：§5 有 14 张表，个别可能溢出页边。溢出的加
   `\resizebox{\linewidth}{!}{...}` 或改用 `\small`。
3. **bib 条目**：`refs.bib` 的年份与作者是按坐标填的，**投稿前须逐条核对**，
   文件头已注明。

## 待办

- §5 有 4082 词，若投 ICLR（正文 9 页）需把 §5.8 方差与部分对照表移入附录
- `refs.bib` 的年份/作者需逐条核对（文件头已注明）

## 为什么仍需 xeCJK

全文唯一的中文在 §4 的 verbatim 块里——那是模型的真实输出，属于数据，
逐字保留而不翻译。附录 A 已译成英文。

## Published

arXiv:2609.03966 (cs.AI), announced 2026-09-04.
https://arxiv.org/abs/2609.03966

Updating the preprint goes through arXiv's **Replace**, not a new submission --
that keeps the identifier and adds a version (v2).

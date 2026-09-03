#!/bin/bash
# 打 arXiv 投稿包。
#
# 三件容易忘、忘了就出事的：
#   1. **必须带 main.bbl**。arXiv 不保证替你跑 bibtex；不带的话参考文献是空的。
#   2. **CJK 字体只能用 Fandol 且按文件名引用**（main.tex 里已如此）。写成
#      Noto / Songti 这类系统字体名在 arXiv 上必然编译失败——那边只有 TeX Live 自带的。
#   3. **不要带 .bib/.aux/.log**，也不要带没被 \input 的 tex（04_control.tex 已弃用）。
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-arxiv_submission.tar.gz}"
W=$(mktemp -d)
mkdir -p "$W/sections"
cp main.tex main.bbl ./*.png "$W/"
# 只拷真正被 \input 的分节
for f in $(grep -oE 'sections/[A-Za-z0-9_]+' main.tex | sort -u); do
  [ -f "$f.tex" ] && cp "$f.tex" "$W/sections/"
done
( cd "$W" && tar czf - . ) > "$OUT"
echo "[arxiv] 已生成 $OUT  ($(du -h "$OUT" | cut -f1))"
echo "[arxiv] 包内 tex：$(ls "$W/sections" | wc -l | tr -d ' ') 个分节 + main.tex"
rm -rf "$W"
echo
echo "投稿前自查："
echo "  · 作者块是否还是 Anonymous（arXiv 不能匿名投）"
echo "  · 建议分类：cs.LG（主）+ cs.SE / cs.CL（次）"

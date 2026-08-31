#!/usr/bin/env python3
"""修正 pandoc 产出：节标题、层级、hypertarget 冗余。

pandoc 把 "5. Results" 这样的首行读成有序列表项，并把 ### 降成 subsubsection，
所以需要一遍机械修正。表格保留 longtable（能跨页），只在 main.tex 补宏包。
"""
import os, re

SEC = {
    "01_intro": "Introduction", "02_related": "Related Work", "03_setup": "Setup",
    "04_control": "Positive Control: The Pipeline Works", "05_results": "Results",
    "06_discussion": "Discussion", "07_limitations": "Limitations",
    "B_repro": "Reproducibility",
}
D = "sections"
for stem, title in SEC.items():
    p = os.path.join(D, stem + ".tex")
    if not os.path.exists(p): continue
    s = open(p, encoding="utf-8").read()
    # 1) 去掉被误读成 enumerate 的节标题
    s = re.sub(r"\\begin\{enumerate\}.*?\\end\{enumerate\}\s*", "", s, count=1, flags=re.S)
    # 2) hypertarget 包裹展开：\hypertarget{x}{% \subsubsection{T}\label{x}} → \subsection{T}
    s = re.sub(r"\\hypertarget\{[^}]*\}\{%\s*\n\\subsubsection\{(.*?)\}\\label\{[^}]*\}\}",
               r"\\subsection{\1}", s, flags=re.S)
    s = re.sub(r"\\hypertarget\{[^}]*\}\{%\s*\n\\paragraph\{(.*?)\}\\label\{[^}]*\}\}",
               r"\\subsubsection{\1}", s, flags=re.S)
    s = re.sub(r"\\hypertarget\{[^}]*\}\{%\s*\n\\subsubsection\[[^\]]*\]\{(.*?)\}\\label\{[^}]*\}\}",
               r"\\subsection{\1}", s, flags=re.S)
    # 3) 小节编号已在标题文字里（"5.1 Four families…"），去掉重复编号交给 LaTeX
    s = re.sub(r"\\subsection\{\d+\.\d+ ", r"\\subsection{", s)
    s = re.sub(r"\\subsubsection\{\d+\.\d+\.\d+ ", r"\\subsubsection{", s)
    prefix = "" if stem == "B_repro" else f"\\section{{{title}}}\n\n"
    if stem == "B_repro":
        prefix = f"\\section{{{title}}}\n\n"
    open(p, "w", encoding="utf-8").write(prefix + s.lstrip())
    n_tab = s.count("\\begin{longtable}")
    print(f"  {stem+'.tex':<22} \\section 已加  子节 {s.count(chr(92)+'subsection')}  表 {n_tab}")

# main.tex 补 pandoc 表格所需宏包
m = "main.tex"; t = open(m, encoding="utf-8").read()
if "longtable" not in t:
    t = t.replace("\\usepackage{booktabs}",
                  "\\usepackage{booktabs}\n\\usepackage{longtable}\n\\usepackage{array}\n"
                  "\\usepackage{calc}\n\\providecommand{\\tightlist}{%\n"
                  "  \\setlength{\\itemsep}{0pt}\\setlength{\\parskip}{0pt}}")
    open(m, "w", encoding="utf-8").write(t)
    print("main.tex 已补 longtable/array/calc 与 \\tightlist")

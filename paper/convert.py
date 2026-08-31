#!/usr/bin/env python3
"""【已停用】早先的 md → tex 通道。

方向已经反转：**paper/sections/*.tex 是唯一事实来源**，
writeup/paper_draft.md 由 paper/regen_draft.py 从 tex 派生。

本脚本会用陈旧的 Markdown 覆盖 sections/*.md，进而诱使人重新
pandoc 成 .tex，把手工修过的 LaTeX（补回的 §5.6.1、重写的 §7、
全部降级过的措辞）一次抹掉。故此处直接拒绝执行。
"""
import sys
sys.exit(
    "convert.py 已停用：方向已反转为 tex → md。\n"
    "要改论文请直接改 paper/sections/*.tex，然后运行:\n"
    "    python paper/regen_draft.py\n"
)

# ---- 以下为原实现，仅作存档 ----
# #!/usr/bin/env python3
# """把 writeup/paper_draft.md 拆成 LaTeX 分节文件。
# 
# 不做全自动 Markdown→LaTeX（表格和数学会出错），只做机械的切分与
# 转义，剩下的人工过一遍。目的是让每次改动 Markdown 后能快速重生成骨架。
# """
# import os, re, sys
# SRC = os.path.join(os.path.dirname(__file__), "..", "writeup", "paper_draft.md")
# OUT = os.path.join(os.path.dirname(__file__), "sections")
# os.makedirs(OUT, exist_ok=True)
# md = open(SRC, encoding="utf-8").read()
# 
# # 按一级节切分
# parts = re.split(r"\n## ", md)
# name_map = {
#     "Abstract": "abstract", "1. Introduction": "01_intro", "2. Related work": "02_related",
#     "3. Setup": "03_setup", "4. Positive control": "04_control", "5. Results": "05_results",
#     "6. Discussion": "06_discussion", "7. Limitations": "07_limitations",
#     "8. Conclusion": "08_conclusion", "9. Reproducibility": "B_repro",
# }
# written = []
# for chunk in parts:
#     head = chunk.split("\n")[0].strip()
#     key = next((k for k in name_map if head.startswith(k)), None)
#     if not key: continue
#     fn = os.path.join(OUT, name_map[key] + ".md")
#     open(fn, "w", encoding="utf-8").write(chunk)
#     written.append((head, name_map[key], len(chunk.split())))
# print(f"已切出 {len(written)} 节到 sections/（.md 形态，待人工转 LaTeX）")
# for h, f, w in written:
#     print(f"  {f:<16} {w:>5} words   {h[:52]}")
# print("\n注意：表格与数学需人工转换；自动转换在 booktabs 表格和 \\% 转义上出错率高。")
# 
#!/usr/bin/env python3
"""从轨迹文件的 provenance 生成附录 E 的臂清单表（LaTeX）。

手抄 50 行配置必错，所以这张表由数据生成，不手写。
provenance 缺失的字段打 --- ，不猜。
"""
import json, glob, os

SHORT = {"Qwen2.5-Coder-1.5B-Instruct":"Qwen-1.5B","Qwen2.5-Coder-3B-Instruct":"Qwen-3B",
         "Qwen2.5-Coder-7B-Instruct":"Qwen-7B","Qwen2.5-Coder-14B-Instruct":"Qwen-14B",
         "Qwen2.5-Coder-32B-Instruct":"Qwen-32B","Meta-Llama-3.1-8B-Instruct":"Llama-3.1-8B",
         "Llama-3.2-1B-Instruct":"Llama-3.2-1B","Llama-3.2-3B-Instruct":"Llama-3.2-3B",
         "Mistral-7B-Instruct-v0.3":"Mistral-7B","deepseek-coder-1.3b-instruct":"DS-1.3B",
         "deepseek-coder-6.7b-instruct":"DS-6.7B"}
esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&")

rows = []
for f in sorted(glob.glob("runs/final/traj_*.jsonl")):
    R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if len(R) != 100:
        continue
    p = R[0].get("provenance") or R[0]
    g = lambda k, d="---": p.get(k, R[0].get(k, d)) if p.get(k, R[0].get(k, d)) is not None else d
    m = str(g("model")).split("/")[-1]
    rows.append((os.path.basename(f)[5:-6], SHORT.get(m, m), g("protocol"), g("strength"),
                 g("fc_schema"), g("adapter"), g("max_tokens"), g("temperature"), g("seed")))

out = [r"""\section{Arm inventory}
\label{app:arms}

Every full-length arm, generated from the trajectory files' own provenance rather than
transcribed by hand. All %d arms share one item set (\texttt{clean[:100]}; distinct-set
count verified = 1), so every cross-arm comparison in the paper is exactly paired. Fields
recorded as \texttt{---} were absent from that generation's provenance schema
(Appendix~A.4); this is disclosed rather than back-filled. Seven arms carry a known-wrong
\texttt{max\_tokens} record and are listed in Appendix~A.1; five are inadmissible for pass
rates and are listed in Appendix~A.3.

Not listed: \texttt{v8\_Llama8B\_recheck} (23 items, a targeted qualitative re-run) and
eight \texttt{*smoke*} files at n=3. Neither is a formal arm.

{\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{longtable}[]{@{}llllllrrr@{}}
\caption{All %d full-length arms and their recorded configuration.}\label{tab:arms}\\
\toprule
Arm & Model & Prot. & Strength & Schema & Adapter & tok & temp & seed \\
\midrule
\endfirsthead
\toprule
Arm & Model & Prot. & Strength & Schema & Adapter & tok & temp & seed \\
\midrule
\endhead""" % (len(rows), len(rows))]
for r in rows:
    out.append(" & ".join(esc(x) for x in r) + r" \\")
out.append(r"""\bottomrule
\end{longtable}
}""")
open("paper/sections/E_arms.tex", "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"附录 E 已生成：{len(rows)} 个臂")

#!/usr/bin/env python3
"""P2：把 Instruct 梯子与已有的 Coder 梯子并排，判定 §5.2 的归因。

判据一律来自 analysis/intent.py 的同一个函数——不为 P2 另写一套，
否则跨实验的数字不可比（这是全文的口径纪律）。

读法：
  · Coder 各尺寸 server-parsed 恒为 0，而 emitted 一路升 → 已有结论
  · 若 Instruct 各尺寸 server-parsed ≈ emitted → 错配锁定在
    「是否受过 tool-call token 训练」这一个变量，§5.2 可以从
    「Coder 家族内的现象」升级为「由训练缺失导致的接口错配」
  · 若 Instruct 也是 0 → 更强的发现：问题出在 hermes parser 与整个
    Qwen2.5 系模板约定之间，范围比论文现在写的更大

两种结果都要报。预期落空不是失败，是把 §5.2 的边界钉得更准。

用法: python p2/compare_ladders.py [--p2-dir runs/p2] [--coder-dir runs/final]
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from intent import TIGHT, REAL, STRONG_TAGS, WEAK_TAGS  # noqa: E402

SIZES = ["1.5B", "3B", "7B", "14B", "32B"]


def counts(path):
    """一条臂的四个计数，全部以首轮为口径——与 §5.2 完全一致。"""
    R = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    first = [(r.get("turns") or [{}])[0] for r in R]
    tight = strong = weak = parsed = 0
    for t in first:
        raw = t.get("raw_output") or ""
        if t.get("fc_tool_call") or t.get("parsed_tool_calls"):
            parsed += 1
        m = TIGHT.search(raw)
        if m and REAL.search(m.group(1)):
            tight += 1
        tag = t.get("_fc_intent")
        if tag in STRONG_TAGS:
            strong += 1
        if tag in WEAK_TAGS:
            weak += 1
    final = sum(1 for r in R if r.get("final_ok"))
    return dict(n=len(R), parsed=parsed, tight=tight, strong=strong, weak=weak, final=final)


def find(d, pats):
    for p in pats:
        g = sorted(glob.glob(os.path.join(d, p)))
        if g:
            return g[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-dir", default=os.path.join(HERE, "..", "runs", "p2"))
    ap.add_argument("--coder-dir", default=os.path.join(HERE, "..", "runs", "final"))
    a = ap.parse_args()

    rows = []
    for sz in SIZES:
        coder = find(a.coder_dir, [f"traj_v5_Qwen{sz}_fc_intent.jsonl"])
        instr = find(a.p2_dir, [f"traj_p2_Qwen2.5-{sz}-Instruct_fc.jsonl"])
        rows.append((sz,
                     counts(coder) if coder else None,
                     counts(instr) if instr else None))

    print("=" * 74)
    print("§5.2 阴性对照：同尺寸、同题集、同协议、同温度，只换血统")
    print("=" * 74)
    hdr = f"{'尺寸':>6s} │ {'Coder parsed':>12s} {'Coder tight':>11s} │ " \
          f"{'Instr parsed':>12s} {'Instr tight':>11s}"
    print(hdr); print("─" * len(hdr))
    have_any = False
    for sz, c, i in rows:
        f = lambda d, k: "—" if d is None else str(d[k])
        if i: have_any = True
        print(f"{sz:>6s} │ {f(c,'parsed'):>12s} {f(c,'tight'):>11s} │ "
              f"{f(i,'parsed'):>12s} {f(i,'tight'):>11s}")
    if not have_any:
        print("\nP2 还没有结果文件。等 run_p2.sh 跑完再运行本脚本。")
        return 0

    print()
    done = [(sz, c, i) for sz, c, i in rows if c and i]
    if not done:
        print("暂无可配对的尺寸。")
        return 0

    cz = all(c["parsed"] == 0 for _, c, _ in done)
    inz = all(i["parsed"] == 0 for _, _, i in done)
    gap = [(sz, i["tight"] - i["parsed"]) for sz, _, i in done]

    print("判定：")
    print(f"  Coder 各尺寸 server-parsed 恒为 0 ......... {'是' if cz else '否'}")
    print(f"  Instruct 各尺寸 server-parsed 恒为 0 ...... {'是' if inz else '否'}")
    print(f"  Instruct 的 emitted−parsed 缺口 .......... {dict(gap)}")
    print()
    if cz and not inz:
        print("  → 与预期一致。同一 chat template、同一题集、同一协议下，")
        print("    受过 tool-call token 训练的一支解析正常，未受训的一支恒为 0。")
        print("    §5.2 可由「Coder 家族内的现象」改写为「训练缺失导致的接口错配」。")
    elif cz and inz:
        print("  → 预期落空，且这是更强的发现：Instruct 同样被吞掉。")
        print("    问题不在 Coder 这一支，而在 hermes parser 与整个 Qwen2.5 系")
        print("    模板约定之间。论文范围要扩大，§5.1 的分层表需相应修订。")
    else:
        print("  → 出现了两种预设之外的模式，需逐尺寸看原始轨迹再下结论。")
    print()
    print("注：本表只是判据计数。写进论文前仍需过 validate_arms.py 的可用性闸门")
    print("    （行数 / rc / n_err / provenance），含缺失数据的臂不计算通过率。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

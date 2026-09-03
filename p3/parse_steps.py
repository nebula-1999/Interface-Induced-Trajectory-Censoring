#!/usr/bin/env python3
"""从 verl 驱动日志里抽出**每步**指标，用于判定 P3 的分支 A / 分支 B。

为什么需要它：`summarise_p3.py` 只给整轮一个汇总数，而 P3 要回答的是
「修好接口后，多轮学习**是否随训练恢复**」——那是一个趋势问题。只看一个
汇总数无法区分「一直在用工具但没进步」和「越训练越会用工具」。

判据（对应 HANDOFF_P3_20260903.md 第 5 节）：
  · tool_calls_time 恒为 0            → 该臂全程零工具执行
  · tool_calls_time > 0 且 score 上升  → 通道打开后学会了多轮修复（分支 A）
  · tool_calls_time > 0 但 score 不上 → 通道打开但没学会（分支 B）

**统计口径的说明**：前后段比较用的是 Welch t 检验，且只作探索性参考。
本文在 §5.6 已经因为多重比较吃过亏（p=0.093 不通过 Bonferroni），
这里不要重蹈覆辙——本脚本报告效应量（前后段均值差）而不只报 p 值。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

# verl 的日志形如：
#   (TaskRunner pid=123) step:7 - global_seqlen/min:89691 - ... - actor/entropy:0.19 - ...
# 数值可能是裸浮点，也可能是 np.float64(0.19) / np.int32(2)。
STEP_RE = re.compile(r"^.*?\bstep:(\d+)\s+-\s+(.*)$")
# 注意：这里必须**恰好两个**捕获组，findall 才能直接解包成 (key, raw)。
# 数值有两种写法：裸浮点 0.19，或 np 标量 np.float64(0.19) / np.int32(2)。
KV_RE = re.compile(
    r"([A-Za-z0-9_/]+):"
    r"(?:np\.[A-Za-z0-9_]+\((?P<wrapped>-?[\d.e+-]+)\)|(?P<bare>-?\d+(?:\.\d+)?(?:[eE]-?\d+)?))"
)

WANT = {
    "critic/score/mean": "score",
    "critic/rewards/mean": "reward",
    "num_turns/mean": "turns_mean",
    "num_turns/max": "turns_max",
    "response_length/mean": "resp_len",
    "timing_s/agent_loop/tool_calls/mean": "tool_calls_time",
    "timing_s/agent_loop/generate_sequences/mean": "gen_time",
    "timing_s/step": "step_time",
    "global_seqlen/mean": "seqlen",
}


def _num(m: re.Match) -> float | None:
    raw = m.group("wrapped") or m.group("bare")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            rec = {"step": step}
            for kv in KV_RE.finditer(m.group(2)):
                key = kv.group(1)
                if key in WANT:
                    n = _num(kv)
                    if n is not None:
                        rec[WANT[key]] = n
            rows.append(rec)
    # 同一个 step 可能出现多次（进度条刷新），保留字段最全的那条
    best: dict[int, dict] = {}
    for r in rows:
        prev = best.get(r["step"])
        if prev is None or len(r) > len(prev):
            best[r["step"]] = r
    return [best[k] for k in sorted(best)]


def _welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """返回 (均值差, Welch t)。样本不足时返回 (差, nan)。"""
    if not a or not b:
        return float("nan"), float("nan")
    if len(a) < 2 or len(b) < 2:
        # 样本不足，只报差值，不报 t（1 个样本算不出方差）
        return st.fmean(a) - st.fmean(b), float("nan")
    va, vb = st.variance(a), st.variance(b)
    se = (va / len(a) + vb / len(b)) ** 0.5
    if se == 0:
        return st.fmean(a) - st.fmean(b), float("nan")
    return st.fmean(a) - st.fmean(b), (st.fmean(a) - st.fmean(b)) / se


def report(rows: list[dict], arm: str, quarter: float = 0.25) -> dict:
    out: dict = {"arm": arm, "n_steps": len(rows)}
    if not rows:
        out["error"] = "没有解析到任何 step 行"
        return out

    k = max(1, int(len(rows) * quarter))
    head, tail = rows[:k], rows[-k:]

    for col in ("score", "turns_mean", "tool_calls_time", "resp_len"):
        h = [r[col] for r in head if col in r]
        t = [r[col] for r in tail if col in r]
        if not h or not t:
            continue
        diff, tv = _welch(t, h)          # 后段 − 前段：正=上升
        out[col] = {
            "first_quarter_mean": round(st.fmean(h), 4),
            "last_quarter_mean": round(st.fmean(t), 4),
            "delta": round(diff, 4),
            "welch_t": None if tv != tv else round(tv, 3),
        }

    tc = [r.get("tool_calls_time", 0.0) for r in rows]
    out["tool_calls_always_zero"] = all(abs(x) < 1e-9 for x in tc)
    out["steps_with_tool_activity"] = sum(1 for x in tc if abs(x) >= 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="verl 驱动日志（*_driver.log）")
    ap.add_argument("--arm", required=True, choices=["broken", "repaired"])
    ap.add_argument("--csv-out", type=Path, help="每步指标落盘路径")
    args = ap.parse_args()

    rows = parse(args.log)
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["step"] + sorted({c for r in rows for c in r if c != "step"})
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"每步指标已写入 {args.csv_out}（{len(rows)} 行）")

    res = report(rows, args.arm)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # 给人看的结论句
    if res.get("tool_calls_always_zero"):
        print("\n→ 该臂全程零工具执行（tool_calls_time 恒为 0）。")
    else:
        sc = res.get("score", {})
        print(f"\n→ 有工具活动的步数：{res['steps_with_tool_activity']}/{res['n_steps']}")
        if sc:
            print(f"→ score 前段 {sc['first_quarter_mean']} → 后段 {sc['last_quarter_mean']}"
                  f"（Δ={sc['delta']}, Welch t={sc['welch_t']}）")
            print("→ 判读：score 上升 = 分支 A；持平 = 分支 B。"
                  "t 值仅作探索性参考，正式结论请以效应量与曲线形状为准。")


if __name__ == "__main__":
    main()

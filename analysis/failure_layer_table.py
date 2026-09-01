#!/usr/bin/env python3
"""对所有 FC 臂逐条做失败层分解，输出一张可并排比较的表。

只跑这一档的子集会让两族的列不可比（Coder 的「未解析」和 Instruct 的「未解析」
根本不是同一个量），所以默认扫全部轨迹文件。

用法::

    python3 analysis/failure_layer_table.py                 # 扫 runs/ 下全部
    python3 analysis/failure_layer_table.py runs/final/*.jsonl
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.failure_layer import TIERS, TIER_ZH, classify  # noqa: E402
from analysis.intent import STRONG_TAGS  # noqa: E402  意图判据只有这一处来源

# 展示顺序：在 no_envelope 后面紧跟它的「其中有调用意图」子计数。
COLUMNS = ("server_parsed", "no_envelope", "_no_envelope_with_intent",
           "parser_loss", "strict_only", "bad_payload")


def guess_parser(name: str) -> str:
    """按臂名推断服务端 parser。轨迹没记这一项，只能猜——见表尾的说明。"""
    low = name.lower()
    if "plugin" in low:
        return "coder*"
    if "llama" in low:
        return "llama3"
    if "mistral" in low:
        return "mistral"
    if "ds" in low or "deepseek" in low:
        return "?"
    return "hermes"


def scan(path: Path):
    tiers = collections.Counter()
    n = 0
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("protocol") != "fc" or not rec.get("turns"):
            return None                      # 非 FC 臂，包装层的概念不适用
        turn = rec["turns"][0]               # 首轮口径，与 L1 一致
        if turn.get("parse_mode") == "request_error":
            continue                         # 请求失败不是模型行为
        n += 1
        tier = classify(turn.get("raw_output"),
                        server_parsed=turn.get("parse_mode") == "fc_tool_call")
        tiers[tier] += 1
        # no_envelope 里混着两种完全不同的情况：模型压根没想调工具（只是答了题），
        # 和模型发了合法的裸 JSON 调用但缺包装层（Coder 的那一种）。不拆开的话
        # 这一列会被读成「都在尝试」，与 34 那个数不可比。
        if tier == "no_envelope" and turn.get("_fc_intent") in STRONG_TAGS:
            tiers["_no_envelope_with_intent"] += 1
    return (n, tiers) if n else None


def main() -> None:
    args = sys.argv[1:]
    root = Path(__file__).resolve().parents[1]
    paths = ([Path(a) for a in args] if args
             else sorted(p for p in root.glob("runs/**/traj_*.jsonl")))

    rows = []
    for p in paths:
        try:
            res = scan(p)
        except Exception as exc:                      # 坏文件不该让整张表跑不出来
            print(f"! 跳过 {p.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if res:
            rows.append((p, *res))

    if not rows:
        print("没有可分析的 FC 臂"); return

    def label(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path)          # 显式传入 repo 外的文件时退回原路径

    labels = {p: label(p) for p, _, _ in rows}
    width = max(len(v) for v in labels.values())
    head = ["已解析", "无包装层", "└其中有意图", "真parser损失", "仅严格度", "载荷非法"]
    print(f"{'臂':<{width}} {'parser':>8} {'n':>4}  "
          + "  ".join(f"{h:>12}" for h in head))
    print("-" * (width + 15 + 14 * len(head)))
    totals = collections.Counter()
    for p, n, tiers in rows:
        cells = "  ".join(f"{tiers.get(t, 0):>12}" for t in COLUMNS)
        print(f"{labels[p]:<{width}} {guess_parser(p.name):>8} {n:>4}  {cells}")
        totals.update(tiers)
    print("-" * (width + 15 + 14 * len(head)))
    n_total = sum(totals.get(t, 0) for t in TIERS)
    print(f"{'合计':<{width}} {'':>8} {n_total:>4}  "
          + "  ".join(f"{totals.get(t, 0):>12}" for t in COLUMNS))
    print()
    for t in TIERS:
        print(f"  {t:<14} {TIER_ZH[t]}")
    print(f"  {'└其中有意图':<14} no_envelope 中 _fc_intent 属 analysis/intent.py 的 strong 档者，"
          f"即「发了裸调用但缺包装层」；其余是压根没想调工具")
    print()
    print("★ 读表注意：no_envelope 只有对 hermes 服务的臂才有含义。Llama 走")
    print("  llama3_json、Mistral 走 mistral、plugin 臂走社区 Qwen2.5-Coder parser，")
    print("  它们本来就不产 <tool_call> 标签，落在 no_envelope 是平凡真，不是发现。")
    print("  parser 列是**按臂名推断**的——轨迹里没记服务端 parser。这是应当补记的")
    print("  元数据：判据依赖它，而它现在只能靠文件名猜。")


if __name__ == "__main__":
    main()

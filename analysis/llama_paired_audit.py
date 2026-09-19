#!/usr/bin/env python3
"""Recompute every paired Llama control statistic from item-level JSONL.

This deliberately keeps the test object explicit.  Earlier manuscript drafts
placed final-pass p-values beside wrong-tool counts; this audit prevents that
cross-metric relabelling from recurring.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FILES = {
    "terse": "traj_v3_Llama8B_fc.jsonl",
    "rich": "traj_v4_Llama8B_fc_rich.jsonl",
    "thought": "traj_v4_Llama8B_fc_cot.jsonl",
    "official": "traj_v5_Llama8B_fc_official.jsonl",
    "strict": "traj_v6_Llama8B_fc_strict.jsonl",
}


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def load(path: Path) -> dict[int, dict[str, bool]]:
    rows: dict[int, dict[str, bool]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            first = row["turns"][0]
            wrong = bool(first.get("action")) and first.get("_fc_arg_keys", []) != ["code"]
            rows[int(row["clean_index"])] = {
                "wrong_tool": wrong,
                "turn1_pass": bool(row["first_ok"]),
                "final_pass": bool(row["final_ok"]),
            }
    if len(rows) != 100:
        raise ValueError(f"{path}: expected 100 unique items, found {len(rows)}")
    return rows


def compare(left_name: str, left: dict, right_name: str, right: dict) -> None:
    if left.keys() != right.keys():
        raise ValueError(f"pair mismatch: {left_name} versus {right_name}")
    print(f"{left_name} -> {right_name}")
    for metric in ("wrong_tool", "turn1_pass", "final_pass"):
        left_n = sum(row[metric] for row in left.values())
        right_n = sum(row[metric] for row in right.values())
        b = sum(left[i][metric] and not right[i][metric] for i in left)
        c = sum(not left[i][metric] and right[i][metric] for i in left)
        print(
            f"  {metric:11s} {left_n:3d}->{right_n:3d}  "
            f"discordant={b}/{c}  p={exact_mcnemar(b, c):.12g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / "final",
    )
    args = parser.parse_args()
    arms = {name: load(args.runs / filename) for name, filename in FILES.items()}
    for name in ("rich", "thought", "official", "strict"):
        compare("terse", arms["terse"], name, arms[name])
    compare("official", arms["official"], "strict", arms["strict"])


if __name__ == "__main__":
    main()

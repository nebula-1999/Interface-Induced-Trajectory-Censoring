#!/usr/bin/env python3
"""Audit the final P3 broken/repaired run from persisted artifacts.

The script is intentionally standard-library only.  It verifies evaluation
hashes and denominators, reconstructs the accepted final lineage after the
step-90 repair resume, reports per-checkpoint held-out outcomes, and explains
the parser-to-dispatch count gap from event-level records.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics


STEPS = (0, 30, 60, 90, 120, 150)
CHANNEL_SIZES = {"multi": 542, "repair": 454}
METRIC_KEYS = (
    "actor/grad_norm",
    "actor/lr",
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/advantages/min",
    "critic/advantages/max",
    "num_turns/mean",
    "timing_s/agent_loop/tool_calls/mean",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def final_eval_path(run_root: Path, arm: str, step: int) -> Path:
    if arm == "broken":
        base = run_root / "broken"
    elif step <= 90:
        base = run_root / "repaired"
    else:
        base = run_root / "repaired_resume_00090"
    return base / "eval" / f"step_{step:05d}.jsonl"


def row_outcome(row: dict) -> tuple[bool, bool, bool]:
    turns = row["turns"]
    first = bool(turns and turns[0].get("all_passed"))
    final = bool(turns and turns[-1].get("all_passed"))
    return first, final, final and not first


def output_signature(row: dict) -> tuple[str, ...]:
    return tuple(str(turn.get("code", "")) for turn in row["turns"])


def audit_eval(run_root: Path) -> tuple[dict, dict]:
    curves: dict[str, dict] = {}
    indexed: dict[tuple[str, int], dict] = {}
    file_manifest: dict[str, dict] = {}

    for arm in ("broken", "repaired"):
        curves[arm] = {}
        baseline: dict | None = None
        for step in STEPS:
            path = final_eval_path(run_root, arm, step)
            complete_path = path.with_suffix(".complete.json")
            complete = read_json(complete_path)
            digest = sha256(path)
            assert digest == complete["sha256"], f"hash mismatch: {path}"
            assert complete["step"] == step
            assert complete["metrics"]["code/native_weight_step"] == step
            assert complete["metrics"]["code/request_failures"] == 0

            rows = read_jsonl(path)
            keys = [(row["channel"], row["task_id"]) for row in rows]
            assert len(rows) == len(set(keys)) == 996, f"bad denominator: {path}"
            assert Counter(row["channel"] for row in rows) == Counter(CHANNEL_SIZES)
            by_key = {(row["channel"], row["task_id"]): row for row in rows}
            indexed[(arm, step)] = by_key
            if baseline is None:
                baseline = by_key

            channels = {}
            for channel, expected_n in CHANNEL_SIZES.items():
                selected = [row for row in rows if row["channel"] == channel]
                outcomes = [row_outcome(row) for row in selected]
                changed = sum(
                    output_signature(row) != output_signature(baseline[(channel, row["task_id"])])
                    for row in selected
                )
                gains = losses = 0
                for row, (_, final, _) in zip(selected, outcomes):
                    base_final = row_outcome(baseline[(channel, row["task_id"])])[1]
                    gains += int(final and not base_final)
                    losses += int(base_final and not final)
                channels[channel] = {
                    "n": expected_n,
                    "turn1_pass": sum(item[0] for item in outcomes),
                    "final_pass": sum(item[1] for item in outcomes),
                    "rescued": sum(item[2] for item in outcomes),
                    "mean_turns": statistics.fmean(len(row["turns"]) for row in selected),
                    "changed_from_step0": changed,
                    "changed_from_step0_rate": changed / expected_n,
                    "final_gains_from_step0": gains,
                    "final_losses_from_step0": losses,
                }
            curves[arm][str(step)] = channels
            file_manifest[str(path.relative_to(run_root))] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
                "rows": len(rows),
                "run_id": complete["run_id"],
                "native_weight_step": complete["metrics"]["code/native_weight_step"],
            }

    broken = indexed[("broken", 150)]
    repaired = indexed[("repaired", 150)]
    paired_endpoint = {}
    for channel in (*CHANNEL_SIZES, "all"):
        gains = losses = 0
        selected = [key for key in sorted(broken) if channel == "all" or key[0] == channel]
        for key in selected:
            broken_final = row_outcome(broken[key])[1]
            repaired_final = row_outcome(repaired[key])[1]
            gains += int(repaired_final and not broken_final)
            losses += int(broken_final and not repaired_final)
        paired_endpoint[channel] = {
            "n": len(selected),
            "repaired_gains_vs_broken": gains,
            "repaired_losses_vs_broken": losses,
            "discordant_total": gains + losses,
        }
    return {"curves": curves, "paired_step150": paired_endpoint}, file_manifest


def event_rows(
    directory: Path, lo: int, hi: int, run_root: Path
) -> tuple[list[dict], dict]:
    rows = []
    manifest = {}
    for path in sorted(directory.glob("events.*.jsonl")):
        manifest[str(path.relative_to(run_root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for row in read_jsonl(path):
            step = row.get("step")
            if step is None or row.get("validate") is True or not lo <= step <= hi:
                continue
            # Original and resumed workers can reuse local counters.
            row["audit_part"] = str(directory.relative_to(run_root))
            for field in ("text", "observation"):
                if field in row:
                    value = row[field]
                    assert isinstance(value, str)
                    assert len(value) == row[f"{field}_chars"]
                    assert hashlib.sha256(value.encode("utf-8")).hexdigest() == row[f"{field}_sha256"]
            rows.append(row)
    return rows, manifest


def audit_events(run_root: Path) -> tuple[dict, dict]:
    specifications = {
        "broken": [(run_root / "broken" / "events", 1, 150)],
        "repaired": [
            (run_root / "repaired" / "events", 1, 90),
            (run_root / "repaired_resume_00090" / "events", 91, 150),
        ],
    }
    result = {}
    manifest = {}
    for arm, parts in specifications.items():
        rows = []
        for directory, lo, hi in parts:
            part_rows, part_manifest = event_rows(directory, lo, hi, run_root)
            rows.extend(part_rows)
            manifest.update(part_manifest)
        kinds = Counter(row["kind"] for row in rows)
        extracts = [row for row in rows if row["kind"] == "extract"]
        accepted_calls = sum(int(row.get("accepted", 0)) for row in extracts)
        accepted_events = sum(int(row.get("accepted", 0)) > 0 for row in extracts)
        tight = sum(bool(row.get("tight")) for row in extracts)
        cross = Counter(
            (bool(row.get("tight")), int(row.get("accepted", 0)) > 0)
            for row in extracts
        )
        accepted_hist = Counter()
        for row in extracts:
            accepted_hist.update(row.get("accepted_names") or [])
        multi_call_events = sum(int(row.get("accepted", 0)) > 1 for row in extracts)
        trajectory_fields = ("audit_part", "step", "sample_index", "rollout_n")
        dispatch_fields = (*trajectory_fields, "request_id", "assistant_turn")

        def key(row, fields):
            assert all(field in row for field in fields), (row["kind"], fields)
            return tuple(row[field] for field in fields)

        dispatches = {
            kind: Counter(key(row, dispatch_fields) for row in rows if row["kind"] == kind)
            for kind in ("call_tool", "execute", "call_tool_result")
        }
        assert dispatches["call_tool"] == dispatches["execute"] == dispatches["call_tool_result"]
        assert all(count == 1 for count in dispatches["call_tool"].values())
        # Extraction records lack request_id/assistant_turn: link them only at
        # trajectory granularity, not by an invented per-generation identity.
        accepted_by_trajectory = Counter(
            key(row, trajectory_fields) for row in extracts if int(row.get("accepted", 0)) > 0
        )
        dispatched_by_trajectory = Counter(
            key(row, trajectory_fields) for row in rows if row["kind"] == "call_tool"
        )
        assert accepted_by_trajectory == dispatched_by_trajectory
        result[arm] = {
            "generation_events": len(extracts),
            "tight_events": tight,
            "accepted_calls": accepted_calls,
            "accepted_generation_events": accepted_events,
            "multi_call_generation_events": multi_call_events,
            "extra_calls_inside_multi_call_events": accepted_calls - accepted_events,
            "dispatch_events": kinds["call_tool"],
            "execute_events": kinds["execute"],
            "observation_events": kinds["call_tool_result"],
            "tight_and_accepted": cross[(True, True)],
            "tight_not_accepted": cross[(True, False)],
            "accepted_not_tight": cross[(False, True)],
            "neither_tight_nor_accepted": cross[(False, False)],
            "accepted_name_histogram": dict(accepted_hist),
            "dispatch_execution_observation_join": "unique request_id/assistant_turn within trajectory and run part",
            "accepted_dispatch_join": "counts matched per trajectory; extract lacks per-turn request identity",
            "verified_text_hashes": sum("text" in row for row in rows),
            "verified_observation_hashes": sum("observation" in row for row in rows),
        }
        assert kinds["call_tool"] == kinds["execute"] == kinds["call_tool_result"]
        assert accepted_events == kinds["call_tool"]
    return result, manifest


def metric_value(line: str, key: str) -> float | None:
    pattern = re.escape(key) + r":(?:np\.(?:float64|int32|int64)\()?([-+0-9.eE]+)"
    match = re.search(pattern, line)
    return float(match.group(1)) if match else None


def audit_training_logs(raw_root: Path) -> dict:
    specifications = {
        "broken": [(raw_root / "runs/p3-formal-20260916-a-pair-relaunch.log", 1, 150)],
        "repaired": [
            (raw_root / "runs/p3-formal-20260916-a-repaired.log", 1, 90),
            (raw_root / "runs/p3-formal-20260916-a-repaired-resume90.log", 91, 150),
        ],
    }
    output = {}
    for arm, parts in specifications.items():
        rows = {}
        for path, lo, hi in parts:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = re.search(r"\bstep:(\d+)\b", line)
                    if not match:
                        continue
                    step = int(match.group(1))
                    if not lo <= step <= hi:
                        continue
                    record = {key: metric_value(line, key) for key in METRIC_KEYS}
                    if record["actor/grad_norm"] is not None:
                        rows[step] = record
        assert sorted(rows) == list(range(1, 151)), f"missing training steps: {arm}"
        summary = {"step_count": len(rows), "steps": [1, 150]}
        for key in METRIC_KEYS:
            values = [rows[step][key] for step in sorted(rows) if rows[step][key] is not None]
            summary[key] = {
                "n": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            }
        output[arm] = summary
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Extracted evidence root containing runs/ and code-agent/",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw_root = args.root.resolve()
    run_root = raw_root / "runs/p3/p3-formal-20260916-a"

    evaluation, eval_manifest = audit_eval(run_root)
    events, event_manifest = audit_events(run_root)
    diagnostics = audit_training_logs(raw_root)
    report = {
        "schema_version": 1,
        "run_id": "p3-formal-20260916-a",
        "final_lineage": {
            "broken": "steps 1-150 from broken",
            "repaired": "steps 1-90 from repaired plus 91-150 from repaired_resume_00090",
            "excluded": "repaired original steps 91-116 after the interrupted branch",
        },
        "evaluation": evaluation,
        "events": events,
        "training_diagnostics": diagnostics,
        "file_manifest": {**eval_manifest, **event_manifest},
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

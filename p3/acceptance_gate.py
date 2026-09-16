"""Verify the real-GPU evidence required before a formal P3 launch.

This is an operational guard, not a security boundary.  It prevents a stale
environment variable or an uninspected smoke run from authorising ~20 GPU-hours.
The record names immutable run directories; verification re-reads and hashes the
artifacts and checks the mechanism-specific invariants every time.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


def _json(path: Path):
    return json.loads(path.read_text())


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_eval(root: Path, expected_steps):
    launch = _json(root / "launch.json")
    for step in expected_steps:
        stem = root / "eval" / f"step_{step:05d}"
        complete = _json(stem.with_suffix(".complete.json"))
        raw = stem.with_suffix(".jsonl")
        _require(complete["sha256"] == _sha(raw), f"eval hash mismatch: {raw}")
        metrics = complete["metrics"]
        _require(metrics.get("code/native_weight_step") == step,
                 f"wrong native weight step: {step}")
        _require(metrics.get("code/native_requests", 0) > 0,
                 f"no native requests: {step}")
        _require(metrics.get("code/request_failures") == 0,
                 f"evaluation request failure: {step}")
    return launch


def _verify_exit_and_summary(root: Path):
    _require(_json(root / "exit.json")["returncode"] == 0,
             f"non-zero smoke exit: {root}")
    summary = _json(root / "events" / "summary.json")
    _require(summary.get("valid") is True and not summary.get("dead_hooks"),
             f"invalid instrumentation: {root}")
    return summary


def _verify_resume_source(launch):
    source = Path(launch["resume_from"])
    recorded = launch.get("resume_checkpoint_files") or {}
    _require(recorded, "resume checkpoint hashes missing")
    for relative, expected in recorded.items():
        path = source / relative
        _require(path.is_file(), f"resume source missing: {path}")
        _require(path.stat().st_size == expected["bytes"],
                 f"resume source size changed: {path}")
        _require(_sha(path) == expected["sha256"],
                 f"resume source hash changed: {path}")


def _verify_observation_chain(root: Path):
    rows = []
    for path in sorted((root / "events").glob("events.*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    key = lambda row: (row.get("step"), row.get("sample_index"),
                       row.get("rollout_n"), row.get("validate"))
    extracts = defaultdict(list)
    results = defaultdict(list)
    for row in rows:
        if row.get("kind") == "extract":
            extracts[key(row)].append(row["t"])
        elif row.get("kind") == "call_tool_result":
            _require(row.get("observation_chars", 0) > 0,
                     "empty tool observation")
            _require(all(row.get(k) is not None for k in
                         ("step", "sample_index", "rollout_n")),
                     "observation lacks trajectory identity")
            results[key(row)].append(row["t"])
    _require(results, "no tool observations in repaired smoke")
    continued = sum(any(t_extract > t_result
                        for t_extract in extracts[k] for t_result in times)
                    for k, times in results.items())
    _require(continued == len(results),
             "not every observed trajectory continued to another generation")
    return len(results), continued


def verify(record_path, project_root):
    record_path = Path(record_path).resolve()
    record = _json(record_path)
    _require(record.get("schema_version") == 1, "unknown acceptance schema")
    _require(record.get("decision") == "accepted", "GPU acceptance not granted")
    broken = Path(record["broken_smoke"])
    resume = Path(record["resume_smoke"])
    repaired = Path(record["repaired_smoke"])

    broken_launch = _verify_eval(broken, [0, 1])
    broken_summary = _verify_exit_and_summary(broken)
    _require(broken_launch["arm"] == "broken" and broken_launch["mode"] == "smoke",
             "wrong broken smoke identity")
    _require(broken_summary["emitted_tight"] > 0
             and broken_summary["accepted_total"] == 0
             and broken_summary["code_tool_executes"] == 0,
             "broken smoke does not establish the closed channel")

    resume_launch = _verify_eval(resume, [1, 2])
    _verify_exit_and_summary(resume)
    _require(resume_launch["arm"] == "broken"
             and resume_launch["mode"] == "resume-smoke"
             and resume_launch["start_step"] == 1,
             "wrong resume smoke identity")
    _verify_resume_source(resume_launch)

    repaired_launch = _verify_eval(repaired, [0, 1])
    repaired_summary = _verify_exit_and_summary(repaired)
    _require(repaired_launch["arm"] == "repaired"
             and repaired_launch["mode"] == "smoke",
             "wrong repaired smoke identity")
    _require(repaired_summary["accepted_total"] > 0
             and repaired_summary["call_tool_events"] > 0
             and repaired_summary["code_tool_executes"] > 0,
             "repaired smoke did not open the channel")
    observations, continued = _verify_observation_chain(repaired)

    # The repaired smoke tested the current runtime semantics.  launch_arm.py is
    # allowed to differ because this gate itself is the subsequent change.
    project_root = Path(project_root).resolve()
    # check_smoke.py only had its stale module docstring corrected after the
    # run; this verifier independently re-checks all of its substantive claims.
    allowed_drift = {"p3/launch_arm.py", "p3/check_smoke.py"}
    for relative, expected in repaired_launch["source_sha256"].items():
        if relative not in allowed_drift:
            _require(_sha(project_root / relative) == expected,
                     f"accepted runtime source changed: {relative}")

    return {
        "acceptance_record_sha256": _sha(record_path),
        "broken_tight": broken_summary["emitted_tight"],
        "repaired_accepted": repaired_summary["accepted_total"],
        "repaired_executed": repaired_summary["code_tool_executes"],
        "observation_trajectories": observations,
        "continued_trajectories": continued,
        "resume_steps_attested": [1, 2],
    }

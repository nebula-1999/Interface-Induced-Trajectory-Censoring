#!/usr/bin/env python3
"""Strict completeness checks for the recovered P1 BFCL arms."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/root/autodl-tmp/p1")
RUN = ROOT / "bfcl_run"

ARMS = {
    "hermes": (
        "P1-Qwen2.5-Coder-7B-Hermes-FC",
        "p1-qwen25-coder7b-hermes",
    ),
    "repaired": (
        "P1-Qwen2.5-Coder-7B-Repaired-FC",
        "p1-qwen25-coder7b-repaired",
    ),
}


def jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise AssertionError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    manifest = json.loads((RUN / "test_case_ids_to_generate.json").read_text())
    expected = {case for cases in manifest.values() for case in cases}
    assert len(expected) == 200, f"manifest has {len(expected)} unique cases, expected 200"

    for slug, (registry, served) in ARMS.items():
        project = RUN / f"full_{slug}"
        result_rows = []
        for path in (project / "result" / registry).rglob("BFCL_v4_*_result.json"):
            result_rows.extend(jsonl(path))
        result_ids = [row.get("id") for row in result_rows]
        assert len(result_ids) == 200, f"{slug}: {len(result_ids)} results"
        assert len(set(result_ids)) == 200, f"{slug}: duplicate result IDs"
        assert set(result_ids) == expected, f"{slug}: result IDs differ from manifest"

        score_total = 0
        score_files = list((project / "score" / registry).rglob("BFCL_v4_*_score.json"))
        assert len(score_files) == 2, f"{slug}: found {len(score_files)} score files"
        for path in score_files:
            rows = jsonl(path)
            assert rows and "total_count" in rows[0], f"{slug}: no score header in {path}"
            score_total += int(rows[0]["total_count"])
        assert score_total == 200, f"{slug}: score total_count={score_total}"

        capture = jsonl(ROOT / "logs" / f"bfcl_{slug}.jsonl")
        capture_ids = {row.get("case_id") for row in capture if row.get("case_id")}
        assert capture_ids == expected, f"{slug}: proxy case IDs differ from manifest"
        errors = [
            row
            for row in capture
            if int(row.get("status") or 0) >= 400 or row.get("_parse_error")
        ]
        assert not errors, f"{slug}: {len(errors)} proxy/request errors"
        wrong_model = [row for row in capture if row.get("model") != served]
        assert not wrong_model, f"{slug}: {len(wrong_model)} rows used the wrong served model"

        parsed = sum(bool(row.get("parsed_tool_calls")) for row in capture)
        print(
            f"{slug}: results=200 score_total=200 cases=200 "
            f"requests={len(capture)} parsed_request_rows={parsed} errors=0"
        )

    print("P1 validation passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"P1 validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

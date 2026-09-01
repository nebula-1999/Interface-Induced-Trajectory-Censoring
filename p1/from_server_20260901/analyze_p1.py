#!/usr/bin/env python3
"""Analyze P1 proxy logs and optionally join official BFCL result/score files.

The emitted-call criterion is the task-agnostic analogue of ``analysis/intent.py``:
a JSON object must name one of the tools actually offered in that request and contain
an ``arguments`` object. This excludes schema echo and pseudo-calls while supporting
BFCL's many tool names (the paper's original criterion is ``run_tests``-specific).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def json_objects(raw):
    """Yield JSON objects embedded in prose/fences without a greedy regex."""
    decoder = json.JSONDecoder()
    for i, char in enumerate(raw or ""):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def normalized_call(obj):
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name, arguments = obj.get("name"), obj.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return (name, arguments) if isinstance(name, str) and isinstance(arguments, dict) else None


def tight(raw, offered_names):
    offered = {name for name in offered_names if name}
    return any(call and call[0] in offered for call in map(normalized_call, json_objects(raw)))


def strong(raw, names):
    text = raw or ""
    return (
        "<tool_call>" in text
        or "[TOOL_CALLS]" in text
        or any(name and f'"{name}"' in text and '"arguments"' in text for name in names)
    )


def weak(raw):
    text = raw or ""
    return '"arguments"' in text or ("{" in text and '"name"' in text)


def read_jsonl(patterns):
    for pattern in patterns:
        for filename in glob.glob(pattern):
            with open(filename, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield filename, json.loads(line)


def recursive_role_count(value, role):
    if isinstance(value, dict):
        return int(value.get("role") == role) + sum(
            recursive_role_count(v, role) for v in value.values()
        )
    if isinstance(value, list):
        return sum(recursive_role_count(v, role) for v in value)
    return 0


def load_bfcl_results(roots):
    """Return model -> case -> number of actual execution-result messages."""
    out = defaultdict(dict)
    if not roots:
        return out
    for root_value in roots:
        root = Path(root_value)
        for path in root.rglob("BFCL_v4_*_result.json"):
            model = path.relative_to(root).parts[0]
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    out[model][row["id"]] = recursive_role_count(
                        row.get("inference_log", []), "tool"
                    )
    return out


def load_bfcl_scores(roots, results):
    """Return model -> case -> BFCL validity from BFCL's sparse score JSONL.

    BFCL writes details for failed cases but normally omits correct cases.  The
    latter are reconstructed as result IDs minus explicit invalid IDs, then
    checked against the aggregate header's correct_count.
    """
    out = defaultdict(dict)
    if not roots:
        return out
    for root_value in roots:
        root = Path(root_value)
        for path in root.rglob("BFCL_v4_*_score.json"):
            model = path.relative_to(root).parts[0]
            rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
            if not rows:
                continue
            header, details = rows[0], rows[1:]
            category = path.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
            candidates = {
                case_id for case_id in results.get(model, {})
                if case_id.startswith(category + "_")
            }
            invalid = {row["id"] for row in details if row.get("valid") is False}
            valid = candidates - invalid
            if len(candidates) != int(header["total_count"]):
                raise ValueError(f"{path}: result/score total_count mismatch")
            if len(valid) != int(header["correct_count"]):
                raise ValueError(f"{path}: inferred correct_count mismatch")
            out[model].update({case_id: case_id in valid for case_id in candidates})
    return out


def summarize_proxy(patterns):
    by = defaultdict(lambda: dict(requests=[], cases=defaultdict(list)))
    for filename, row in read_jsonl(patterns):
        tag = row.get("tag") or Path(filename).name
        by[tag]["requests"].append(row)
        if row.get("case_id"):
            by[tag]["cases"][row["case_id"]].append(row)
    return by


def first_request(rows):
    return min(rows, key=lambda r: (
        r.get("request_index") is None,
        r.get("request_index") if r.get("request_index") is not None else 10**9,
        r.get("ts", 0),
    ))


def print_request_table(by):
    print("=" * 112)
    print("P1 HTTP funnel (first request per BFCL case; emitted format measured only when parse failed)")
    print("=" * 112)
    print(f"{'arm':37s} {'cases':>6s} {'HTTP err':>8s} {'first parsed':>12s} "
          f"{'unparsed':>9s} {'tight':>7s} {'strong':>8s} {'weak':>6s} {'any parsed':>11s}")
    for tag, data in sorted(by.items()):
        cases = data["cases"]
        rows = [first_request(v) for v in cases.values()] if cases else data["requests"]
        ok = [r for r in rows if r.get("status", 200) < 400 and r.get("n_tools_offered")]
        parsed = sum(bool(r.get("parsed_tool_calls")) for r in ok)
        unparsed = [r for r in ok if not r.get("parsed_tool_calls")]
        tight_n = sum(tight(r.get("content", ""), r.get("tool_names_offered", [])) for r in unparsed)
        strong_n = sum(strong(r.get("content", ""), r.get("tool_names_offered", [])) for r in unparsed)
        weak_n = sum(weak(r.get("content", "")) for r in unparsed)
        any_parsed = sum(any(x.get("parsed_tool_calls") for x in rs) for rs in cases.values())
        http_err = sum(r.get("status", 200) >= 400 for r in data["requests"])
        print(f"{tag[:37]:37s} {len(rows):6d} {http_err:8d} {parsed:12d} "
              f"{len(unparsed):9d} {tight_n:7d} {strong_n:8d} {weak_n:6d} {any_parsed:11d}")


def print_joined_table(by, results, scores):
    if not results and not scores:
        return
    print("\n" + "=" * 112)
    print("P1 five-layer case funnel (multi_turn_base only; BFCL execution/validity joined by case id)")
    print("=" * 112)
    print(f"{'arm':37s} {'n':>5s} {'unparsed tight':>15s} {'first parsed':>13s} "
          f"{'any parsed':>11s} {'executed':>9s} {'success':>8s} {'rescued':>8s}")
    for tag, data in sorted(by.items()):
        model = tag.replace("/", "_")
        cases = {k: v for k, v in data["cases"].items() if k.startswith("multi_turn_base_")}
        if not cases:
            continue
        first = {k: first_request(v) for k, v in cases.items()}
        first_parsed = {k for k, row in first.items() if row.get("parsed_tool_calls")}
        first_tight = {
            k for k, row in first.items()
            if not row.get("parsed_tool_calls")
            and tight(row.get("content", ""), row.get("tool_names_offered", []))
        }
        any_parsed = {k for k, rows in cases.items() if any(r.get("parsed_tool_calls") for r in rows)}
        executed = {k for k, n in results.get(model, {}).items() if n > 0 and k in cases}
        success = {k for k, valid in scores.get(model, {}).items() if valid and k in cases}
        rescued = (any_parsed - first_parsed) & success
        print(f"{tag[:37]:37s} {len(cases):5d} {len(first_tight):15d} "
              f"{len(first_parsed):13d} {len(any_parsed):11d} {len(executed):9d} "
              f"{len(success):8d} {len(rescued):8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=["p1/logs/*.jsonl"])
    ap.add_argument("--result-root", action="append")
    ap.add_argument("--score-root", action="append")
    args = ap.parse_args()
    by = summarize_proxy(args.logs)
    if not by:
        print("No proxy logs found.")
        return 1
    print_request_table(by)
    results = load_bfcl_results(args.result_root)
    scores = load_bfcl_scores(args.score_root, results)
    print_joined_table(by, results, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

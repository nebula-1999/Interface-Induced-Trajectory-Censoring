#!/usr/bin/env python3
"""The **single** intent criterion used by all prose, tables, and figures.

Three earlier counts (manual 18 / tight criterion 21 / trajectory labels 23)
came from temporary scripts written at different stages. This module defines
three evidence levels; every report must state which level it uses:

  tight   Strict criterion: {"name":"run_tests", ... "arguments" ...
          "code":"<a string literal containing real Python>"}. It excludes
          two contaminants: (1) repetition of the injected schema (with
          parameters/description but no arguments), and (2) illustrative
          pseudocode where "code" is a variable name rather than a string.
  strong  Broader strong evidence: trajectory _fc_intent is labelled
          json_named_call or xml_tool_call.
  weak    Weak heuristic: only a JSON structure (json_arguments or
          json_code_obj), which may be an ordinary answer.
"""
import json, os, re

TIGHT = re.compile(
    r'"name"\s*:\s*"run_tests".{0,200}?"arguments"\s*:\s*\{.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}',
    re.S)
REAL = re.compile(r'\\n|def |class |return |import |lambda ')
STRONG_TAGS = {"json_named_call", "xml_tool_call"}
WEAK_TAGS = {"json_arguments", "json_code_obj"}


def is_tight_call(raw: str) -> bool:
    m = TIGHT.search(raw or "")
    return bool(m and REAL.search(m.group(1)))


def classify(path):
    """Return parsed/tight/strong/weak counts for an arm, all on the **first turn**."""
    R = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    first = [(r["turns"] or [{}])[0] for r in R]
    return {
        "n": len(R),
        "parsed": sum(1 for t in first if t.get("action")),
        "tight": sum(1 for t in first if is_tight_call(t.get("raw_output") or "")),
        "strong": sum(1 for t in first if t.get("_fc_intent") in STRONG_TAGS),
        "weak": sum(1 for t in first if t.get("_fc_intent") in WEAK_TAGS),
        "final_ok": sum(r["final_ok"] for r in R),
        "n_err": sum(1 for r in R for t in r["turns"]
                     if (t.get("raw_output") or "").startswith("__ERROR__")),
    }


if __name__ == "__main__":
    D = os.path.join(os.path.dirname(__file__), "..", "runs", "final")
    print(f"{'scale':<8}{'n':>5}{'parsed':>8}{'tight':>8}{'strong':>8}{'weak':>8}{'final pass':>10}")
    print("-" * 60)
    for s in ["1.5B", "3B", "7B", "14B", "32B"]:
        c = classify(os.path.join(D, f"traj_v5_Qwen{s}_fc_intent.jsonl"))
        print(f"{s:<8}{c['n']:>5}{c['parsed']:>8}{c['tight']:>8}{c['strong']:>8}"
              f"{c['weak']:>8}{c['final_ok']:>10}")

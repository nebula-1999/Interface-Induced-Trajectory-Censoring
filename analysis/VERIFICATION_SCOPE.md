# Anonymous supplement: verification scope

From the extracted package root:

```
python3 analysis/verify_supplement.py
python3 analysis/submission_statistics.py --output /tmp/recomputed_statistics.json
```

The first command uses only Python's standard library. See imports in the statistics
script for its additional dependencies.

| Verification target | Supported? |
| --- | --- |
| Fifteen Llama tests from reduced per-item records | Yes |
| Post-hoc native FC: nine paired contrasts and timeout inventory | Yes; all 1,626 reduced item records in native_fc/outcomes.json |
| Native FC: package versions, tool schema, frozen task IDs and source hashes | Yes; anonymized native_fc/configurations.json |
| Native FC: exclusion of two historically flagged items and final-submission rescue alternative | Yes; descriptive sensitivities, without altering the 542-item primary analysis |
| Native FC: reclassify raw generations or re-execute code | No; full raw conversations/tests/checkpoints are not in this reduction |
| Other historical p-values from stated discordant counts | Yes; not a raw-trajectory audit |
| 7B six-checkpoint outcomes and endpoint discordances | Yes; 11,952 outcomes |
| 7B recorded funnel and accepted resume segments | Yes; excluded records retained |
| Reclassify all training tight labels from raw text | No |
| BFCL fixed-output JSON extraction, structural checks and official AST scoring | Yes; 100 documented cases plus repaired positive control |
| Inspect BFCL factorial first responses and tau-bench full trajectories | Yes; 800 first responses and 230 trajectories |
| Inspect RL raw emissions/observations | Yes; deterministic 10-trajectory sample per arm, not the full population |
| Verify original text hashes or checkpoint weights | No |

Lineage selects broken steps 1-150, original repaired 1-90, resumed repaired 91-150.
Original repaired steps 91-116 are excluded. Validation/out-of-range events are
retained with flags; the verifier independently recalculates inclusion by ranges.
Formal step-90 evaluation is the original, not the resumed retest.

Outcomes retain arm, step, channel, pseudonymous task ID, first/final success,
rescue and turn count. The exporter checked original completion hashes, native
weight-step metadata and zero request failures. These source-level checks need
the full archive and cannot independently be repeated from this reduction.

Compact events retain source segment/file/line, kind, step, trajectory indices, available
request/turn IDs and recorded classification/count fields. Text, observations,
arguments and identifying run paths are omitted. Dispatch/execution/observation
can be uniquely joined. Extraction lacks per-turn request identity: its accepted
counts match dispatch only at trajectory granularity; no missing links are invented.

The verifier recomputes results from records and compares with expected.json.
Agreement verifies reduction and arithmetic, not omitted raw labels. Hashes
cannot substitute for file contents.

The "No" for reclassification above refers to the full training population, not
the supplied twenty-trajectory raw-text sample. See mechanism_evidence/README.md
for selection, source boundaries, and the offline BFCL replay command.

Full-archive-only commands include analysis/p3_final_evidence.py, checkpoint
audits, raw emission classification and historical figure regeneration. These
are not default runnable commands in the anonymous supplement.

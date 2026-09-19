# P3 final evidence audit

Audit date: 2026-09-19 (Asia/Shanghai)

## Verdict

The formal 7B broken-FC versus repaired-FC comparison is now locally auditable
from raw evaluation rows, full rollout events, final-lineage logs, manifests,
and server-side checkpoint tensor comparisons. No new training was run.

The intervention restores the training-time tool channel by more than three
orders of magnitude, but the accepted 150-step, single-seed run shows no
detected held-out multi-turn improvement. Retained checkpoints show actual LoRA
updates, evaluation records attest the expected native weight step, and the
accepted totals exclude the abandoned resume branch. Weight-step metadata is
an attestation by the evaluator, not an independent hash of its in-memory weights.

## Retrieved evidence

- Remote run: `/root/autodl-tmp/runs/p3/p3-formal-20260916-a`
- Local archive: `p3/results/formal_20260916_a/p3_evidence_20260919.tar.gz`
- Archive SHA-256:
  `bfb0aa8447292726d09eac1d140cbb031635d12e100fd2a18a42ea7208e5c1cc`
- Extracted files: 173 files, 154 MB
- Machine-readable audit: `p3/results/formal_20260916_a/audit.json`
- Audit SHA-256:
  `24d4e3df499df9a6e1b819beb199440df50e96b9ce2be7c4de19a6cd5e55617e`
- Checkpoint audit:
  `p3/results/formal_20260916_a/checkpoint_lora_audit.json`

The 15 GB checkpoint weight files remain on the persistent AutoDL data disk and
were not copied into Git. The tensor audit was run in place before shutdown.

## Accepted lineage

- Broken: steps 1-150 from `broken`.
- Repaired: original steps 1-90 from `repaired`, then steps 91-150 from
  `repaired_resume_00090`.
- Excluded: the interrupted original repaired steps 91-116.
- Formal repaired step 90 uses the original evaluation. The resumed test-retest
  evaluation is retained separately and changes final pass from 428 to 426.

All 12 accepted checkpoint evaluations contain 996 unique `(channel, task_id)`
rows, zero request failures, matching content hashes, and the expected native
weight step. The 996 rows comprise 542 multi-turn items and 454 seeded-repair
items.

## Complete multi-turn curve

Each cell is `turn-1 / final / rescued after turn 1`, out of 542.

| Step | Broken FC | Repaired FC |
|---:|---:|---:|
| 0 | 418 / 430 / 12 | 418 / 430 / 12 |
| 30 | 418 / 430 / 12 | 418 / 429 / 11 |
| 60 | 420 / 433 / 13 | 417 / 431 / 14 |
| 90 | 418 / 432 / 14 | 415 / 428 / 13 |
| 120 | 419 / 433 / 14 | 415 / 427 / 12 |
| 150 | 418 / 430 / 12 | 415 / 427 / 12 |

At step 150, repaired has zero gains and three losses relative to broken on the
multi-turn channel (exact paired McNemar p=0.25). Relative to repaired step 0,
the final checkpoint has one gain and four losses (p=0.375). Neither result is
evidence that repair is harmful.

The generated code sequence changes for 93/542 broken items and 110/542
repaired items between step 0 and step 150. Aggregate stability therefore does
not mean that the policy's emitted programs stayed fixed.

## Training and checkpoint update evidence

Both accepted final-lineage logs contain exactly steps 1-150 with no missing
step. The learning rate is 1e-6 throughout.

| Diagnostic | Broken | Repaired |
|---|---:|---:|
| actor gradient norm | 0.0120-0.0251 | 0.00641-0.01599 |
| training reward mean | 0.150-0.469 | 0.168-0.435 |
| mean turns per step | 2.000-2.063 | 2.891-4.484 |
| mean tool-call time per step | 0-0.0205 s | 0.260-1.839 s |

The server-side checkpoint audit compares all 392 LoRA tensors (80,740,352
parameters):

- Broken step 120 to 150: 45,471,572 values changed (56.3%), LoRA delta
  L2=0.0787, and the aggregate tensor hash changed.
- Repaired step 90 to resumed step 150: 45,296,056 values changed (56.1%),
  LoRA delta L2=0.1222, and the aggregate tensor hash changed.

This directly rules out a dead update path over the retained endpoint
intervals. The checkpoint comparison is reproducible on the server with
`analysis/p3_checkpoint_lora_audit.py`.

## Event-level funnel

| Arm | Generation events | Tight | Parsed calls | Accepted generations | Executed | Observed |
|---|---:|---:|---:|---:|---:|---:|
| Broken | 19,203 | 8,689 | 10 | 10 | 10 | 10 |
| Repaired | 33,367 | 17,608 | 16,912 | 16,844 | 16,844 | 16,844 |

The repaired parsed/executed difference is fully accounted for: 40 assistant
generations contain multiple `run_tests` calls, producing 68 calls beyond the
first. Accepted-generation and dispatch counts match within every trajectory.
All 16,844 dispatches join uniquely to an execution and observation by request ID
and assistant turn. Extraction records lack those two identifiers, so the
extraction-to-dispatch link is a per-trajectory count check, not a one-to-one
generation join. The deployed one-parallel-call cap explains the surplus.

The tight classifier and parser acceptance are different tests, not nested
funnel stages. In repaired FC, 16,323 generations satisfy both, 1,285 are tight
but unaccepted, 521 are accepted but not tight, and 15,238 satisfy neither.
Therefore `17,608 -> 16,912` must not be described as 696 calls lost by the
parser.

## Reproduction

```bash
python3 analysis/p3_final_evidence.py \
  p3/results/formal_20260916_a/raw \
  --output p3/results/formal_20260916_a/audit.json
```

The script verifies evaluation hashes, denominators, native weight steps,
request failures, final-lineage selection, full text and observation hashes,
request-level dispatch/execution/observation joins, event counts, complete
checkpoint curves, output changes, and training-log diagnostic ranges.

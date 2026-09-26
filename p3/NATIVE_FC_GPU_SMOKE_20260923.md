# Native FC GPU smoke — 2026-09-23

Status: loading and bounded integration checks PASS. No full evaluation or training started.

Hardware: NVIDIA A800 80GB PCIe; approximately 37 GB disk free before testing.
All arms used the unchanged common evaluator, six identical frozen tasks (first
three and last three), original template/tools, greedy decoding, 1024-token output
cap and four-generation cap. Source hash for native_fc_eval.py:
`dd31b34547df7740431b851ad87c4a33990c8ece66912ac950f2bc1de60d3b46`.

| Arm | Tasks | Final pass | Tool executions | Length stops | Evaluation seconds |
|---|---:|---:|---:|---:|---:|
| base | 6 | 5 | 0 | 0 | 33.15 |
| broken150 | 6 | 5 | 0 | 0 | 57.85 |
| repaired150, final lineage | 6 | 5 | 0 | 0 | 61.18 |

Times exclude model loading. All outputs have COMPLETE.json, no INVALID marker,
identical task IDs/source hashes, and raw-text hashes verified after download.
Every terminal test returned status `ok` (not every solution passed).
Both adapters' in-memory ordered-float32 tensor hashes exactly match the verified
checkpoint receipts; scaling is checked as 1.0 by the evaluator. Base is explicitly
adapter-free. Model config/tokenizer identity is checked, but full base shards have
not received an independent weight digest audit.

These six-task behavioral results are not evidence of equivalence or of no native
FC learning. All three produced plain code without calls; no prompts were changed
to force favorable scientific outcomes. The three ordinary smokes consequently do
not exercise real-model-initiated multi-turn interaction.

## Separate synthetic positive control

`native_fc_gpu_feedback_smoke.py` injects one known valid run_tests call for a toy
function, executes it through the sandbox, checks tool-role rendering, and asks the
real base 7B GPU model to continue from that feedback. PASS. This verifies the
otherwise-unvisited execution/feedback/GPU-continuation path; it does NOT measure
model initiation and is excluded from benchmark outcomes. Its 128-token diagnostic
continuation budget does not change the 1024-token scientific evaluator.

Local evidence: `p3/results/native_fc_20260923/smoke_base_1902/`,
`smoke_broken_1905/`, `smoke_repaired_resumed/`, and
`gpu_feedback_fixture.json`. The interrupted assistant turn did not interrupt
broken inference: completion was checked before proceeding, not rerun.

Eight local controller/preflight tests pass using unittest discovery. An initial
module-style test invocation failed to resolve local imports; discovery corrected
the invocation without modifying runtime code.

## Remaining

No full run is authorized by this smoke. Six single-generation examples are too
small to estimate average multi-turn cost or scientific performance. A naive
single-turn extrapolation across 3 x 542 tasks is about 3.82 hours excluding load;
four-turn behavior and longer outputs can increase that substantially. Use a
bounded pilot before agreeing a full-run GPU budget. Do not reuse the earlier
unvalidated 1–3 hour estimate as a guarantee. No original checkpoint was deleted,
no training started, and the GPU instance was not shut down.

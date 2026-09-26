# Native FC bounded pilot — verified 2026-09-23

All three 24-task arms completed; no timeout or INVALID marker. Driver wall time
22:55:06–23:06:49 (11m43s, including loads). No full run launched.

| Arm | Final success | Tasks with tool execution | Tool executions | Rescues | Generation/test seconds |
|---|---:|---:|---:|---:|---:|
| base | 13/24 | 1 | 1 | 0 | 142.42 |
| broken150 | 12/24 | 2 | 5 | 0 | 293.80 |
| repaired150 | 13/24 | 1 | 1 | 0 | 237.92 |

All arms use identical frozen first/last twelve task IDs and source hashes; this
is a deterministic convenience pilot, not a representative random sample. The
earlier six-task smoke overlaps this set and must not be added as extra samples.
No generation hit the length cap. Every terminal and interactive test returned
status `ok` (execution completed, not necessarily tests passed). Every recorded
execution has an observation and matching tool-message ID. All raw-text hashes
verify locally. Both GPU-loaded LoRA digests match their final-lineage receipts.

base/repaired each initiated once, on HumanEval/3. broken executed four calls on
HumanEval/10 and one on Mbpp/799. Thus genuine model-initiated feedback paths were
exercised, in addition to the separate synthetic smoke fixture.

base and repaired have the same per-task final pass vector; broken differs only on
HumanEval/3 (failure versus their success). This small pilot does not establish
equivalence, absence of learning, or harm. More tool executions are not by
themselves evidence of better tool use. Existing training totals (10 vs 16,844)
remain unchanged and are not comparable to these evaluation invocation counts.

At observed throughput, linear extrapolation for all three arms over 542 tasks
is 4.23 GPU hours excluding loads. This is not a guarantee: task length and
multi-turn mix can change. A planning allowance of roughly 5–6 hours is more
prudent for the sequential backend. Full evaluation remains a separate decision;
do not change prompt/interface in response to the pilot outcome.

Downloaded evidence: `p3/results/native_fc_20260923/pilot_24_20260923/`.
GPU confirmed idle after completion; the instance was not shut down. No training,
checkpoint deletion, or publication was performed.

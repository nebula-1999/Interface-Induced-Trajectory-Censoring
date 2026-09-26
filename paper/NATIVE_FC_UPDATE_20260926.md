# Native FC evaluation: manuscript update, 2026-09-26

## What was completed

Full post-hoc evaluation of the same 542 tasks under a common working native FC
interface for base, broken150 and final-lineage repaired150. No new training or
seed was added. The three complete raw archives were retrieved after the run.
Input/task hashes, loaded adapter digests, raw emission hashes and observation
links pass the audit. Pilot items are overlapping and never added to the denominator.

| Weight | Success | Initiating tasks | Executions | Rescues | Terminal timeouts |
|---|---:|---:|---:|---:|---:|
| base | 239/542 | 46 | 91 | 2 | 4 |
| broken150 | 248/542 | 50 | 93 | 1 | 5 |
| repaired150 | 248/542 | 47 | 90 | 1 | 6 |

Repaired minus broken: 16 gains, 16 losses, exact McNemar p=1;
paired risk difference 0 percentage points, conservative pointwise 95% interval
[-3.47, 3.47]. The interval covers item-pair uncertainty only, not seed/retest noise.
All terminal timeouts remain unsuccessful in the full denominator. They have not
been independently diagnosed as algorithmic nontermination; there are no interactive
test timeouts. No output hit the token cap.

## Reviewer response

We added the requested evaluation of both trained checkpoints under a single
working native FC interface, together with the untrained base. Only accepted
run_tests calls obtain interactive tool-role feedback; terminal scoring is independent
and supplies no additional feedback. This removes the previous requirement that the
evaluation program initiate all testing. It does not eliminate task/test distribution
shift or exactly reproduce the training AgentLoop. Equal success totals do not imply
equivalent policies: the two endpoint passing sets differ on 32 tasks.

The result supports the restricted statement that neither the prior programmatic
feedback protocol nor this common native FC protocol detects a success gain from
parser repair in this single-seed, 150-step setting. It does not establish that
interface repair cannot improve learning or that native FC behavior never changes.

## Changes

- Both abstracts and local abstract-field text: add native FC, marked post-hoc.
- Both result sections: concise native-FC endpoint paragraph; full protocol/table
  and timeout inventory in Appendix G, label app:nativefc.
- Limitations: replace missing-measurement framing with remaining task/controller,
  seed and uncertainty limitations; do not pool the two rescue definitions.
- Conclusions: restrict no-detected-benefit claim to the two measured protocols.
- Appendix I: retain historical 27-test audit, disclose nine new exploratory tests
  and their separate adjustment. None is nominally significant.
- Reproducibility: supply 1,626 reduced item records plus statistics and CPU script.
  Full native-FC generations/test inputs/checkpoints are not claimed to be included.

## Recompute

`python3 analysis/native_fc_results.py --archive p3/results/native_fc_20260923/full_20260925 --out submission_packages/native_fc`

In the anonymous bundle use `--outcomes native_fc/outcomes.json` instead of
`--archive`, with a fresh output directory. Reduced records reproduce statistical
results and timeout identities; they do not permit raw-text reclassification.

No GitHub push, arXiv replacement or OpenReview upload is performed by this update.

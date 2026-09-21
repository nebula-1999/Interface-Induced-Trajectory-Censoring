# Final scope and statistical consistency audit — 2026-09-21

No GPU job, additional seed, or new inference experiment was run.

## 1. RL interpretation

Both manuscripts now introduce the shared parser-independent evaluator before
the result: generated code is executed automatically and test feedback returned.
The conclusion concerns recovered training experience and no detected held-out
gain under that evaluation, not learned voluntary tool initiation. Gradient
availability wording was replaced with experience access. Rescues are 12 at the
two endpoints, not at every checkpoint. Historical 1.5B details remain in the
appendix and no longer lead the public manuscript's Introduction.

## 2. Count units and association boundary

Tables distinguish regex-positive generation events, accepted calls, executed
tool invocations, and evaluation items. There are 16,912 accepted calls in 16,844
accepted generation events. Forty multi-call generations contribute 68 extra calls.
Logged max_parallel_calls=1 and the versioned verl tool-agent loop explain the
first-call selection rule: the loop slices the returned call list and does not
dispatch, merge or queue the rest. Source reference:
https://github.com/volcengine/verl/blob/v0.9.0/verl/experimental/agent_loop/tool_agent_loop.py

Dispatch/execution/observation have per-request joins. Extraction only has
trajectory-level count agreement; no stronger per-call join is asserted. The
installed library file itself was not retrieved for a new byte-identity check.

## 3. Statistics

- New executable: analysis/submission_statistics.py; outputs in the adjacent JSON.
- RL risk difference: -0.5535 percentage points; conservative finite-sample 95%
  interval [-1.7858, 0.7176] percentage points. Two 97.5% Clopper-Pearson marginal
  intervals for discordance probabilities are combined with a union bound.
- The interval assumes independent item pairs, excludes training-seed and
  inference-retest variation, and is not adjusted over the whole test family.
- A retrospective family explicitly enumerates 27 comparisons. It includes all
  15 audited Llama metrics plus 12 unique reported comparisons. Its threshold is
  0.00185185. Historical motivating runs remain exploratory. This is not a
  preregistered policy and replaces the earlier approximate family of twelve.
- Llama strict-to-ReAct (p~0.0019) and the both-parsed Llama contrast (p~0.0021)
  no longer pass the correction. The strict wrong-tool/final-pass effects and
  the random-300 Qwen protocol contrast still pass. No outcome counts changed.
- Four unit tests check boundary cases, arm-swap symmetry, the reported endpoint,
  and exact small-multinomial coverage on a grid.

The 15 Llama tests are recomputed from item records; the other tests use explicit
persisted discordance counts with source labels. This is not a new raw-item audit
of every historical comparison. The Qwen100 repair 16/7 counts were checked
against the stored paired trajectories.

## 4. Measurement and consistency

Tight is defined by the implemented regex and lexical markers, not valid JSON or
executable Python. 80*0.9 is a sensitivity estimate assuming precision transfer;
it does not correct false negatives. Qwen3 tight counts concern only unparsed
outputs. Llama official-template strict turn-1 is 33 to 46. Feedback-conditioned
tables are descriptive, not causal localization. Human-validation denominator,
stratification and presentation-confound caveats now agree between F and H.
The Qwen both-parsed nominal p-value is consistently rounded from the exact
11/4 test (0.11847) to 0.118, not 0.119.

## 5. Preservation

Data Errata is unchanged in this revision. All 17 current limitations and the
section on what would change the conclusions remain. Preregistration documents,
failed/invalid runs, interruption/retest disclosures and full audit history remain
retained. Unrelated learning/interview documents were not edited.

## 6. Submission boundary

Public author block remains Wenbo Wang and Di Sang. Anonymous PDF has no author
metadata. Both PDF abstracts and both form-text files now use the user's latest
research claims; no OpenReview or arXiv web form was modified.

The local anonymous zip contains the paper sources/PDF, statistics code and
outputs, reduced Llama outcome inputs, P3 aggregate evidence/configuration, and
preregistration documents. It excludes raw rollout archives, heavy checkpoints,
public-author sources, git history and launch logs. Its README explicitly states
that it is not a complete training/raw-event reproduction bundle. The package
passes an explicit identity-marker scan; this does not guarantee impossibility of
re-identification from the already-public research topic or commit identifiers.

Still requires the submitting author's action: upload the correct candidate,
synchronize the actual portal abstract, and download/review the portal-produced
final PDF. No online submission or public repository push was performed here.

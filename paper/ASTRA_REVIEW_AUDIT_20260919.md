# Astra review audit (2026-09-19)

This file separates verified manuscript defects from requests that still need
persisted evidence. It is not a rebuttal draft.

## Verified and corrected

1. **BFCL 2x2 terminology.** For the parse-rate matrix `[[0, 0], [0, 196]]`,
   ordinary marginal main effects are not zero. What is zero is each
   one-component simple effect starting from the documented configuration.
   The paper now says that neither one-component replacement recovers a call,
   while joint replacement recovers 196/200 and demonstrates complementarity.
   It also states that official BFCL scores were computed for the two
   end-to-end corner configurations, not all four 2x2 cells.

2. **Llama McNemar labels.** The manuscript had attached final-pass p-values to
   wrong-tool counts. Recalculation from the 100 paired JSONL rows gives:

   | comparison | metric | counts | discordant | exact two-sided p |
   |---|---|---:|---:|---:|
   | terse -> rich | wrong tool | 23 -> 22 | 6 / 5 | 1.0 |
   | terse -> thought | wrong tool | 23 -> 59 | 1 / 37 | 2.8376e-10 |
   | terse -> official | wrong tool | 23 -> 22 | 1 / 0 | 1.0 |
   | official -> strict | wrong tool | 22 -> 0 | 22 / 0 | 4.7684e-7 |
   | official -> strict | turn-1 pass | 33 -> 46 | 1 / 14 | 9.7656e-4 |
   | official -> strict | final pass | 44 -> 61 | 1 / 18 | 7.6294e-5 |

   The executable audit is `analysis/llama_paired_audit.py`.

3. **Turn-1 localization.** Equal aggregate turn-1 counts do not identify the
   failure layer when the evaluation adapter changes parser, template, and
   few-shot together. The paper now treats the 53-vs-53 result as descriptive
   and relies on fixed-output replay and event tracing for localization.

4. **Classifier semantics.** `tight` is now described as a heuristic,
   tight-classified call-like emission rather than a validity oracle. The
   4,000-character cap creates false negatives, but 0.900 precision proves
   false positives also exist; the raw total is therefore not called a strict
   lower bound.

5. **Null-learning wording.** The claim is now “no held-out improvement was
   detected in one seed over 150 steps,” not that a working channel is
   generally insufficient. Longer runs would narrow the observation's scope,
   not overturn an already observed endpoint.

6. **Qwen3 control.** It is now a preregistered, cross-family supporting ladder,
   not a strict scale counterfactual or proof about capability growth.

7. **Error-bearing sampling seed.** The complete-denominator endpoint remains
   reported with the request counted as failure, while the clean-arm analysis
   is labeled separately. Two clean seeds have a mathematically defined but
   unstable SD; the manuscript no longer says it cannot be computed.

8. **Human validation denominators and dependence.** The text now distinguishes
   the 98-item sample and gold set, the 97 complete A1/A2 pairs, and the 62
   identical-byte pairs. It says the adjudicator was blind, but does not call
   the whole gold-set construction classifier-independent because classifier
   strata and disagreement selection affected the workflow.

9. **P3 evaluation protocol and resume lineage.** The manuscript now specifies
   the shared parser-independent programmatic-feedback evaluation (native
   rollout RPC, current LoRA step attestation, temperature 0, four turns,
   1,024 tokens, same 542 EvalPlus items). It also states that formal event
   totals include original steps 1-90 plus resumed steps 91-150 and exclude the
   abandoned 26-update step-91-116 branch.

10. **Paired endpoint effect.** The formal RL comparison now reports the
    paired final-pass difference ($-0.55$ percentage points), an approximate
    paired Wald 95% interval ($[-1.18,0.07]$), all three discordances, and the
    exact McNemar $p=0.25$. The interval describes evaluation-item uncertainty
    for this completed run; it does not represent training-seed variation.

## P3 evidence packaging resolved on 2026-09-19

1. **Checkpoint evaluations retrieved and verified.** The formal broken and
   repaired step 0/30/60/90/120/150 JSONL and completion records are now local.
   Every file matches its recorded SHA-256, contains 996 unique channel/task
   rows, has zero request failures, and attests its native weight step. Appendix
   G now reports the complete 542-item multi-turn curve.

2. **Training and parameter updates verified.** The accepted final-lineage logs
   contain all 150 steps in both arms, non-zero gradient norms at every step,
   reward and advantage variation, and a fixed learning rate of 1e-6. Direct
   checkpoint comparison finds non-zero changes in 45.47M broken-arm and 45.30M
   repaired-arm LoRA parameters over the retained endpoint intervals.

3. **Machine-verifiable lineage manifest produced.** The executable
   `analysis/p3_final_evidence.py` selects broken steps 1-150 and repaired
   original steps 1-90 plus resumed steps 91-150, explicitly excluding the
   abandoned original 91-116 branch. Its `audit.json` output records hashes and
   sizes for every accepted evaluation and event file.

4. **Funnel gaps resolved by event-level joins.** The 16,912 parsed calls occur
   in 16,844 accepted generation events. Forty generations contain multiple
   calls and contribute exactly 68 calls beyond the first. Accepted-generation
   and dispatch counts match per trajectory; dispatch, execution and returned
   observation join uniquely by request ID and assistant turn. Extraction lacks
   those identifiers, so that first link is not a per-generation join. Tight
   classification and parser acceptance are not nested: 16,323 events satisfy
   both, 1,285 are tight-only, and 521 are accepted-only.

## Priority judgment

No additional training seed was needed to repair a correctness defect: the
first run is now auditable end to end. A second paired seed would test variation
across training trajectories and is the next expensive scientific extension,
not a prerequisite for interpreting the completed single-seed intervention.

## Reconciliation with the Seed review

The Seed review is useful as a presentation and exposure audit, but its overall
assessment is more optimistic than the evidence currently supports. We adopted
the items that can be verified from persisted artifacts and rejected conclusions
that depend on an incorrect estimand or an unsupported inference.

### Adopted and verified

- Replaced “decisive” with “preregistered single-variable” for the formal RL
  comparison and kept the one-seed scope in the main text.
- Explained the 540/542 denominator split and how the two known-defective items
  enter the formal paired comparison.
- Separated the Llama single-variable strict effect (44 to 61, +17) from the
  bundled interface contribution relative to terse FC (49 to 61, +12).
- Marked the matched supporting ladder's upper endpoint as 8B; the 32B value is
  from the separate mismatched ladder.
- Specified the shared parser-independent held-out evaluator. Equal step-0
  aggregate counts are reported as an observation, not used to infer the design.
- Reported validation-sample storage-cap hits by scale: 0/6, 0/13, 0/15, 0/24,
  and 10/40; all observed hits occur at 32B.
- Added explicit references to every main-text figure and table, repaired the
  Appendix D step-count description, and added verified citations for vLLM,
  GRPO, LoRA, KodCode, Harness-Bench, and Agent-Reactive Bugs.
- Clarified that applying the 0.900 tight-stratum precision to the 32B headline
  assumes transfer across checkpoint sizes.

### Not adopted as stated

- The BFCL matrix does not have “two zero main effects” under the usual marginal
  definition. It has two zero simple effects at the documented baseline and a
  large joint replacement effect.
- Equal first-turn totals do not localize the failure layer when the adapter also
  changes template and few-shot context.
- A 4,000-character storage cap creates false negatives, but an imperfect
  classifier also creates false positives; the uncalibrated emitted count is not
  a strict lower bound.
- A second paired seed is useful for training-trajectory generalization, but it
  cannot repair missing checkpoint artifacts, incorrect lineage accounting, or
  unverified parameter updates. It remains lower priority than the four evidence
  packaging items above.
- Converting all figures to vector form is desirable polish, not a correctness
  blocker. Treating an em-dash edit as a major issue is likewise disproportionate.

### Literature check

The suggested overlapping works are real and now cited. Harness-Bench studies
model--harness configuration effects across workflows; this paper differs by
isolating a serialization--parser contract with component interventions and
tracing it into RL experience. Agent-Reactive Bugs develops an empirical taxonomy
from issue reports; this paper contributes controlled paired evidence. These
distinctions are now stated explicitly rather than implied.

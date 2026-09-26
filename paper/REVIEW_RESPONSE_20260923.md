# Evidence closeout, 23 September 2026

## Completed without GPU or new generation

1. Baseline provenance: versioned vLLM 0.27.1 Qwen Models guidance is family-level,
   not an individually validated Coder recommendation. Appendix D now states this,
   including retrospective access date and the prior issue. No deployment-prevalence
   inference is made. The short results section points readers to this boundary.
2. New fixed-output BFCL analysis: deterministic answer-blind JSON recovery on all
   100 original single-turn outputs yields 97 recovered / offered-name / structurally
   schema-compliant calls and 92 official AST successes. Repaired positive control
   reproduces 96. Existing dedicated parser has no `<tools>` envelope to consume in
   the original 100 outputs. This does not overwrite any original 2x2 cell.
3. Local anonymous package adds 800 factorial first responses, 200 single-turn
   scoring records (100 per corner), all 230 tau-bench trajectories and a deterministic
   step-1 sample of ten RL trajectories per arm. Verifier checks counts and supplied
   raw text hashes. Files are locally assembled, not uploaded or publicly pushed.
4. Removed `actual cause`, the claim that extra tokens can only widen the gap,
   and causal claims about aggregate-score stabilisation. Also removed a residual
   unsupported claim that changing the BFCL template preserves payload bytes.
5. Updated reproducibility scope in the paper and bundle. BFCL input fields are
   inference logs, not full raw HTTP captures. RL sample permits content inspection,
   not execution replay (training test inputs absent); extraction-to-dispatch links
   remain trajectory-level. No false claim of full-archive reproducibility.

## Still not done, by design

Validation completed: both PDF builds pass without overfull boxes or unresolved
references; changed appendix/configuration pages were rendered and visually
inspected. The anonymous version remains within nine main-text pages (32 total);
the public version is 44 pages. A fresh extraction of the 77-file anonymous zip
passed all compact-event/outcome checks, raw-subset checks, four replay unit tests,
and the independent 92/100 and 96/100 official-checker recomputation.

- Shared working native-FC evaluation of both RL endpoints and step 0: requires
  checkpoint availability/loading checks, a pinned common FC protocol, and a
  separate GPU budget decision. No performance outcome is inferred here.
- Full multi-turn counterfactual scoring from frozen outputs: invalid because
  recovered actions change feedback and subsequent generations; not attempted.
- New seed, retraining, new family, full-scale regeneration: not started.
- Permission/license review before public release remains separate from creating
  a local submission candidate. No private archive was pushed to GitHub.

## Recompute

`python3 analysis/test_bfcl_fixed_replay.py`

`python3 analysis/bfcl_fixed_replay.py --bundle submission_packages/mechanism_evidence/bfcl`

`python3 analysis/verify_mechanism_evidence.py`

`python3 analysis/build_anonymous_submission.py`

BFCL scoring uses unchanged pinned official source files with a minimal registry
shim matching the original handler's `underscore_to_dot=True`. Structural schema
checks cover the observed request-schema vocabulary, not format annotations.

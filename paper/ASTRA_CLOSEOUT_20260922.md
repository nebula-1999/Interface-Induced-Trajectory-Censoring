# Final classification and supplementary-material audit

No new inference, training or GPU access. No experimental count was replaced.

## Changes

- Recomputed BFCL first-request indicators from the original clean-run logs:
  documented 166/169/187, repaired 0/1/2. Documented tight-and-weak overlap is
  166; strong-and-weak overlap is 168. These are overlapping predicates, not
  mutually exclusive categories. Table columns and benchmark-specific definitions corrected.
- KodCode regex does not prove valid JSON/Python. External BFCL and tau indicators
  use different definitions. Human-validation precision is not transferred to them.
- Added 11,952 reduced 7B outcomes and compact events with lineage exclusions.
  Independent verifier recomputes curves, endpoint discordances, accepted calls,
  generation events, execution/observation totals and available joins.
- Preserved excluded original repaired steps 91-116. No per-generation
  emission-to-dispatch join is invented; extraction lacks the required IDs.
- Reproducibility statement and Appendix B distinguish package-supported checks
  from full-archive-only procedures. Raw text and checkpoints remain omitted.
- Llama +12/+19 contrasts explicitly descriptive under the 27-test correction.
- Clean-arm and end-to-end error handling distinguished, without reinstating p=0.648.
- Historical full-checkpoint rescue range corrected to 5-10 from existing sequences.
- Abstract endpoint claim softened in both TeX versions and both portal text files.
- Historical ReAct table caption and body bound together; conclusion no longer
  implies an unavailable emission-to-execution join.

## Verification

Follow-up scope correction: the body now limits the shared classifier to KodCode
and names the BFCL JSON-call criterion explicitly. The replay returns at the
missing-envelope check, so the tau-bench statement now says that none of the 789
turns reaches payload validation. Removed the public-long-version inference that
this establishes zero malformed payloads or a cross-domain payload-validity difference.
No classifier behavior or experimental numbers changed.

Final wording follow-up: both Discussion sections describe preflight as a screening
check for configurations that fail to return an expected call, not a guarantee of
catching every studied failure or identifying its mechanism. No new experiment.

- Source audit checked original evaluation hashes, denominators and event text hashes
  before deriving reduced records.
- Standalone verifier passes without the original archive: 11,952 outcomes,
  broken 10 executions, repaired 16,844 executions, 16,912 accepted calls.
- Unpacked anonymous-package statistical output matches the stored output exactly.
- Both paper versions compiled; targeted tables rendered and visually inspected.
- No online submission or GitHub push performed. Unrelated writeup edits preserved.

## Local regeneration

```
python3 analysis/build_p3_supplement.py
python3 analysis/verify_supplement.py submission_packages/p3_compact
python3 analysis/build_anonymous_submission.py
```

The exporter needs the separately retained local full archive. The verifier in the
resulting package does not. See analysis/VERIFICATION_SCOPE.md for exact boundaries.

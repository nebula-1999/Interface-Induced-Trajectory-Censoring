# Raw-content mechanism evidence

This local anonymous candidate adds content inspection to the statistical supplement.
It does not reproduce training or claim complete raw-archive disclosure.

## BFCL fixed-output analysis

Run from the package root:

```
python3 analysis/test_bfcl_fixed_replay.py
python3 analysis/verify_mechanism_evidence.py
python3 analysis/bfcl_fixed_replay.py --bundle mechanism_evidence/bfcl
```

No GPU, API or non-standard Python package is required. The bundle contains the
unchanged AST checker and its dependencies from Gorilla commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, with Apache-2.0 license.
The registry shim sets only `underscore_to_dot=True`, matching the experiment's
OpenAI handler; it avoids loading unrelated inference-provider SDKs.

The new extractor scans maximal JSON values in byte-preserved response text,
without fixing syntax, renaming tools, changing values or consulting gold answers.
All recovered calls are sent to the checker: an extra wrong call is not silently
discarded in favor of a correct one. Structural signature checks cover types,
required fields, declared argument names, array items and enums, not format
annotations. These are offline call candidates, not proof of execution intent in
every context, and this is not a production-safe fallback parser.

Expected documented-arm results, denominator 100: recovered 97, allowed tool names
97, structural signature checks 97, official AST successes 92. Original server
parses: zero. Positive control: unchanged repaired-arm outputs score 96/100.
None of the 100 original responses contains the dedicated parser's required
`<tools>` start tag. Therefore this result does not attribute recovery to that
existing plugin or change the measured factorial cells.

Records include recorded inference request fields, raw documented-arm content,
structured repaired outputs, schemas, ground truth and verdicts. They are not full
HTTP request captures. First responses for all 800 factorial cases are supplied
separately; the replay scoring is only for 100 single-turn cases. No counterfactual
multi-turn success is computed from frozen conversations.

## Tau-bench

Both complete 115-task arms are retained, sorted by task ID, with original task
metadata, rewards and messages. Customer identities are synthetic benchmark
personas, not study participants. Recorded calls and tool messages can be audited;
environment re-execution requires the external benchmark and its configuration.
No original HTTP headers, host names, credentials or wall-clock timestamps are
exported. Local simulator results are not leaderboard-comparable absolute scores.

## RL sample

For each arm, take the ten lowest `(sample_index, rollout_n)` keys at training
step 1, without outcome selection. Keep all extraction, dispatch, execution and
observation events for those keys. Emission/observation SHA-256 values are checked
against the actual retained text before export. Request identifiers are hashed;
process IDs, run IDs, paths and timestamps are omitted.

This is an inspectable raw-text sample, not a statistically representative estimate
or a full environment replay. Original task prompts/tests are not included.
Extraction has trajectory identity but no per-turn request ID. Do not infer exact
extraction-to-dispatch links from adjacent records. Dispatch/execution/observation
retain shared request identifiers where available. No model code is executed by
the supplement verification scripts.

## Baseline provenance

The versioned vLLM 0.27.1 documentation's Qwen Models section lists
`Qwen/Qwen2.5-*` and the `hermes` flag. This is family-level guidance, not a
separate Coder-specific validation. Documentation inspected on 2026-09-23:
https://docs.vllm.ai/en/v0.27.1/features/tool_calling/#qwen-models

The known Coder incompatibility predates these experiments (issue opened
2026-01-23): https://github.com/vllm-project/vllm/issues/32926
These sources support plausibility of the configuration choice, not a prevalence
estimate of deployment mistakes. The documentation access is retrospective,
not a newly discovered contemporaneous snapshot of the experimental machine.

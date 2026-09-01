# P1: BFCL v4 five-layer replication

P1 tests whether the paper's interface-censoring result survives on a standard tool-use
benchmark. It uses BFCL's official v4 data, multi-turn executor, and evaluator at pinned
Gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.

The stock BFCL local-Qwen path is **not** used: it calls `/v1/completions` and parses raw text
inside BFCL, bypassing the serving parser under study. `bfcl_registration.py` instead routes
BFCL's official OpenAI function-calling handler through `toolcall_proxy.py` and then vLLM's
real `/v1/chat/completions` parser. Only two additions are made: explicit
`tool_choice=auto` and trace headers joining HTTP requests to BFCL case IDs.

## Paired design

- Model: Qwen2.5-Coder-7B-Instruct, vLLM 0.27.1.
- Fixed sampling seed `20260901`, inference seed `0`, `temperature=0`, and identical IDs in both arms.
- 100 `simple_python` cases for standard single-turn AST accuracy.
- 100 `multi_turn_base` cases for parse, actual execution, and final task validity.
- Arm 1: documented `hermes` parser.
- Arm 2: the already-vendored community Qwen2.5-Coder parser and paired template.

Each arm must pass a two-case end-to-end smoke test. A failed HTTP request, missing case-ID
header, missing result/score row, wrong vLLM version, or missing plugin makes the run fail
closed and creates `P1_INVALID`; it never silently continues to the full arm.

## Deploy

```bash
OLD_GPU_HOST=autodl-code bash p1/deploy_p1.sh
```

If the old instance rebooted, first update its SSH alias/port, or set `OLD_GPU_HOST` to a
working alias. P1 never connects to or modifies the P2 host.

Outputs are under `/root/autodl-tmp/p1/bfcl_run/`; `P1_SUMMARY.txt` contains the joined
funnel, and `provenance.txt` pins code/plugin hashes and runtime versions. The script does
not shut the instance down automatically.

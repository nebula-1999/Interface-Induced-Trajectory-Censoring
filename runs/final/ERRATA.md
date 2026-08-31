# Errata (read alongside the trajectory files)

> 中文版见 [`ERRATA_zh.md`](ERRATA_zh.md).

## 1. Seven ReAct arms record the wrong `max_tokens` provenance

The files below record `2048`; the **actual generation limit was 1024**. Cause: the `gen()`
helper in `probe_react_full.py` carried a default of `max_tokens=1024` for a long time, while
`gen_fc()` used the global `MAX_TOKENS=2048`. The provenance block written to disk recorded
`MAX_TOKENS` for both paths, so the ReAct side disagreed with what was actually sent.
Fixed 2026-08-31 11:20 (`gen`'s default changed to `MAX_TOKENS`).

```
traj_v3_Llama8B_react.jsonl        recorded 2048 / actual 1024
traj_v3_Qwen7B_react.jsonl         recorded 2048 / actual 1024
traj_v3_Mistral7B_react.jsonl      recorded 2048 / actual 1024
traj_v11_DS1.3b_react.jsonl        recorded 2048 / actual 1024
traj_v11_DS6.7b_react.jsonl        recorded 2048 / actual 1024
traj_v11_Llama32_1B_react.jsonl    recorded 2048 / actual 1024
traj_v11_Llama32_3B_react.jsonl    recorded 2048 / actual 1024
```

**Direction of the bias.** The contemporaneous FC arms were genuinely at 2048, so in every
affected ReAct-vs-FC comparison **the FC side had the longer generation budget**. ReAct still
won those comparisons, which makes the reported protocol gaps a **conservative estimate**:
correcting the defect can only widen them, not shrink them.

**Replacement data.**

- the four v11 arms are superseded by `traj_v13_*_react.jsonl` (genuinely 2048)
- the three v3 arms are superseded by `traj_v14_*_react.jsonl` (genuinely 2048)

Cite the v13 / v14 versions throughout. The v3 / v11 ReAct arms are retained only as a
historical record.

## 2. The Llama-3.1-8B FC initiation rate needed a manual correction

`traj_v3_Llama8B_fc.jsonl` records L1 (initiation) as 97/100, but **23 of those calls invoked
the task's own function** rather than `run_tests`. That batch predates tool-name validation,
and `has_action` did not distinguish which tool was called. **The corrected figure is
L1 = 74/100.**

`traj_v8_Llama8B_recheck.jsonl` re-runs exactly those 23 items under the same configuration:
23/23 reproduce as `wrong_tool`, with tool names such as `can_form_word` and
`check_password_strength` — the task functions themselves.

From the same batch, `traj_v3_Qwen7B_fc_nosuffix.jsonl` has L1 = 0 and needs no correction;
`traj_v3_Mistral7B_fc.jsonl` has L1 = 2, and those two calls were not individually re-checked
for tool name.

## 3. Arms inadmissible for pass-rate comparison

```
traj_v5_Mistral7B_fc_official.jsonl        n_err=42  (rc=2)
traj_v9_Mistral7B_fc_official_strict.jsonl n_err=39  (rc=2)
traj_v9_Mistral7B_fc_strict.jsonl          n_err=3   (rc=2)
traj_v3_Mistral7B_fc.jsonl                 n_err=2   (rc=2)
traj_v11_Llama32_3B_fc_strict.jsonl        rc=2
```

For these arms the **error-rate census is valid** — that is precisely the phenomenon under
study — but **pass rates are not comparable**, because data is missing rather than failing.
Any *p*-value computed across them is withdrawn; specifically, the previously reported
p = 0.648 is retracted.

## 4. Provenance schema drifted across generations

Fields were added incrementally over four schema generations: the earliest arms lack `seed`,
`temperature` and `fc_schema`. The fields that matter for the claims — `model`, `protocol`,
`adapter`, `max_tokens`, `clean_index` — are present in every generation.

The `script_sha256` field was never successfully added (the patch was interrupted twice).
Script consistency is instead guaranteed externally, by comparing against
`v13_pinned_hashes.txt` with the validator.

## 5. Verified correct

- **All 49 full-length (n=100) formal arms share one and the same 100-item set** (`clean[:100]`;
  number of distinct sets = 1), so the pairing is valid.
  - Separately, `traj_v8_Llama8B_recheck.jsonl` is a 23-item qualitative re-run (by design it
    covers only those 23), and there are 8 `*smoke*` files at n=3. Neither counts as a formal arm.
  - An earlier version of this file said "37" — that was a snapshot taken while the counting
    experiment was still running. Corrected.
- Every FC arm was genuinely at `max_tokens=2048`.
- A9's sampling genuinely took effect: at temperature 0.6, two seeds differ on 95/100
  first-turn programs and flip the final outcome on 16/100.

## 6. Third-party components

The Qwen2.5-Coder `<tools>` parser comes from
[hanXen/vllm-qwen2.5-coder-tool-parser](https://github.com/hanXen/vllm-qwen2.5-coder-tool-parser)
(Apache 2.0), downloaded 2026-08-31 00:45, at what was then the head of `main`:

```
commit 1b921501f30cbfe347dccb1db7de3c82a1d55131  (1b92150, 2026-04-29)
        "fix: buffer partial <tools> prefix in streaming to prevent tag leak"
```

Local file SHA256 (first 20 hex digits):

```
c16bb1f88936a2d96c7c  qwen2_5_coder_tool_parser.py               (unmodified)
736bd175adbf90942c1d  tool_chat_template_qwen2_5_coder.jinja     (MODIFIED: few-shot get_weather -> run_tests)
a95b2a9b91e65b3d452f  tool_chat_template_qwen2_5_coder.jinja.orig (original)
```

Reason for the modification: the template's few-shot example calls `get_weather`, which is not
the single legal tool in this experiment (`run_tests`). Left unchanged it induces calls to a
non-existent tool, simultaneously inflating L1 and contaminating the empty-argument statistics.

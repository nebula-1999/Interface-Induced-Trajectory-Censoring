3. Setup

**Task.** 100 problems drawn from a decontaminated KodCode subset (n-gram containment
against EvalPlus). **All 49 full-length arms in this paper use the identical 100 items**
(verified: unique item-set count = 1), so every cross-arm comparison is paired.

**Protocols.**
*ReAct* — a text protocol with `Thought:` / `Action: run_tests` / `Action Input: ```python …``` `
/ `Observation:`; the tool name is fixed in the template.
*Function calling (FC)* — OpenAI-style, `tool_choice: auto`, one tool `run_tests` taking a
single `code` string. Schema variants: **terse** (bare `{"type":"string"}`), **rich**
(parameter description, counter-example, code sample), **strict** (`strict: true`,
`additionalProperties: false`).

**Serving.** vLLM 0.27.1, per-family parser and chat template as documented.
`temperature=0` is greedy but **not bit-wise deterministic** under continuous batching:
identical prompts can differ across batch compositions. The main table is a single sample
per item; §5.8 quantifies sampling variability separately at `temperature=0.6` rather than
assuming greedy decoding removes it.
`max_tokens = 2048` for **both** protocols (see `ERRATA.md` §1 — an earlier asymmetry
favoured FC; re-running under the true shared limit changed final pass by ≤1 item).

**Measured quantities.** For each item's first turn we record: whether the server parsed a
tool call; the raw output; the tool name called; the raw `arguments`; `finish_reason`; and
a protocol-agnostic intent classification (§3.1). Execution, per-turn results, and final
pass are recorded per turn.

### 3.1 Intent criterion

Reporting only server-parsed calls is the practice under examination, so we need an
independent measure of what the model emitted. A single classifier (`analysis/intent.py`)
is shared by all text, tables and figures:

- **tight** — a JSON object naming `run_tests` with an `arguments` object whose `code` is a
  **string literal containing real Python**. Excludes two contaminants we observed:
  (i) echoing the injected schema (`parameters`/`description` present, no `arguments`);
  (ii) illustrative pseudo-code (`{"code": test_code}` — an identifier, not a string).
- **strong** — trajectory tag `json_named_call` or `xml_tool_call`.
- **weak** — JSON-shaped fragments only; at 1.5B these are almost entirely false positives
  and we report them separately rather than merging them.

### 3.2 Human validation of the intent criterion

The scale trend in §5.2 rests entirely on an automated classifier, so we validated it
against human judgement. 98 outputs were sampled **stratified by classifier verdict**
(40 `tight`, 28 `strong`-but-not-`tight`, 30 neither) — deliberately over-sampling the
decision boundary so both false positives and false negatives are estimable. Sampling is
seeded and the annotator saw only the raw output, never the classifier's label.

We report **two rounds**, because the first one is itself a finding.

| | agreement | Cohen's κ | precision | recall | FP | FN |
|---|---|---|---|---|---|---|
| Round 1 (raw output only) | 86.7% | 0.713 | 70.0% | 96.6% | 12 | 1 |
| After adjudication | **96.9%** | **0.936** | **95.0%** | 97.4% | 2 | 1 |

All 12 round-1 false positives shared one cause: **the tool call sits mid-document, after a
fenced ```python block** (match positions 1209–2585 in outputs of median length ~2000). The
annotator read the code block, concluded "this is a direct answer, not a call", and stopped.
Re-shown the matched span alone, 10 of 13 disputed items were reversed.

We report this rather than only the adjudicated figure because **it reproduces the paper's
mechanism on a human reader**: the evidence was present in every case, and what determined
whether it was seen was the order in which the text was scanned. The serving parser is
blocked by format; the annotator was blocked by reading order. Both produce the same
conclusion — "the model did not call the tool" — from the same bytes.

Three items survived adjudication as genuine disagreements. They mark the criterion's grey
zone rather than annotation noise, and we do not resolve them.

**Correction applied.** Weighting the per-stratum agreement rates by pool sizes gives a
correction factor of **0.957** on `tight` counts. §5.2's headline (80/100 at 32B) becomes
**≈77** after correction; the monotone trend and the contrast against a flat server-side
zero are unaffected. We report uncorrected counts in tables with this factor stated, rather
than silently rescaling.

**Single annotator.** One person labelled both rounds, so κ measures human–classifier
agreement, not inter-annotator reliability. A second independent annotator would be needed
for the latter and was not available.

### 3.3 Validity gating

An arm is admissible for pass-rate comparison only if it has exactly 100 items, exit code 0,
**zero request errors**, and provenance (model / protocol / temperature / `max_tokens` /
adapter) matching the intended configuration. Arms with request errors retain a valid
**error-rate census** but are marked N/A for pass rates and *no p-value is computed for
them*. This rule is enforced in code (`validate_arms.py`, `final_table.py`), not by
convention: an earlier draft computed *p*=0.648 on such an arm, and that number is
retracted.

---

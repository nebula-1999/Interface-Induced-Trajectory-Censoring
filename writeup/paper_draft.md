# Interface-Induced Trajectory Censoring

**Tool-call interfaces silently decouple observed agent behaviour from model capability,
and contaminate reinforcement learning**

*Draft v1 — 2026-08-31. All numbers traceable to `runs/final/`; read `ERRATA.md` before citing.*

---

## Abstract

Evaluations of LLM agents routinely report tool-call rates taken from the serving stack.
We show this number can be zero while the model emits well-formed calls, and that the
resulting misattribution contaminates both benchmarks and reinforcement learning.

Configuring four model families for function calling on vLLM **following official
documentation**, we find each fails at a different layer, and — critically — the most consequential ones
fail **silently**:
DeepSeek-Coder's chat template never injects tools; Qwen2.5-Coder emits JSON the
recommended `hermes` parser cannot see; Llama-3.1-8B hallucinates *the task function
itself* as a callable tool in 23% of items; Mistral-7B repeats its `[TOOL_CALLS]` marker
and triggers HTTP 400. Mistral's failure does surface as HTTP 400; the other three do not. Apart from a missing
`--enable-auto-tool-choice`, every failure we found either returns HTTP 200 with an empty
`tool_calls` array or is indistinguishable from a model that declines to call tools.

For Qwen2.5-Coder the undercount **grows monotonically with scale**: across a 21× range
the server parses 0/100 calls at every size, while well-formed calls the model actually
emits rise from 0 to 80/100 at 32B. Llama's failure is eliminated by a single
`strict: true` flag (23→0, *p*=0.0001) — a remedy that three prior single-variable
controls failed to identify.

The same mismatch reaches **RL training**: under function calling a 10-step GRPO run
executes **zero** tool calls with `num_turns` pinned at its minimum, while a ReAct run
matched in model, data, hyper-parameters, seed and prompt strength (though not in step
count — 10 vs 3) executes 2.05 calls per rollout. The training dashboard shows nothing wrong: `critic/rewards/mean` climbs from 0.233 to
0.281 **while the tool-execution count stays exactly zero**. Because the reward admits
direct answers, no trajectory ever rewards tool use — **the tool-using branch is
unreachable by gradient**.
This explains our own training result, where RL improved first-draft quality
(+2.6–2.8 pp; *p*=0.06–0.08, **not significant at α=.05**, direction consistent across three runs) while multi-turn rescues stayed flat at 6–9/540.

Installing a dedicated adapter restores the mechanism (parse 0→84, multi-turn 0→37,
rescues 0→9), yet a protocol gap remains: with interfaces fully repaired, ReAct still
beats function calling on both families that admit a comparison (80 vs 61, *p*=0.0019;
74 vs 62, *p*=0.0169); the third produces no error-free FC run at all. Conditioning on
items the server actually parsed, that gap survives on one family and not the other. **Interface problems mask protocol problems; only after fixing the former
does the latter become visible.**

---

## 1. Introduction

We set out to train a multi-turn code agent with RL and ended up unable to explain our own
training curve. Over 150 GRPO steps on Qwen2.5-Coder-1.5B, pass@1 on a held-out EvalPlus
split rose 2.6–2.8 points, consistently across three independent runs
(*p* = 0.064 / 0.078 / 0.077 — **none significant at α = .05**; the evidence is the
consistent direction and magnitude across two algorithms and two seeds, not any single test). But the gain
decomposed strangely: **91–94% of newly-passing items passed on the first turn**, and the
number of items rescued by turn 2 or later stayed flat at **6–9 out of 540** from step 0
to step 150. RL had improved first-draft quality and left multi-turn debugging untouched.

The natural reading is that a 1.5B model cannot learn multi-turn repair. The actual reason
is that **the tool was never successfully called**. Not rarely — never. And nothing in the
stack said so: the server returned HTTP 200, `tool_calls` was an empty array, and the
training loop recorded a well-formed single-turn trajectory.

### 1.1 What is measured when we measure an agent

An agent evaluation does not measure a model. It measures a composition:

```
model  ×  protocol  ×  serialization  ×  parser  ×  execution stack
```

Every stage can fail, and the failure of most stages is **indistinguishable at the output
from model incapability**. When a parser does not recognise the format a model emits, the
serving layer reports zero tool calls — exactly what a model that refuses to use tools
would produce. We call this **interface-induced trajectory censoring**: the agent's
trajectory is truncated at an interface boundary, and the truncation is systematically
misattributed to the policy.

The interface acts in **two opposite directions**, and our results contain one clean
instance of each:

- **Masking.** The model emits a valid call, the parser does not recognise it, the action
  never reaches the environment (Qwen2.5-Coder, §5.2).
- **Suppression.** The model emits an *invalid* call, and a schema constraint removes it
  from the sampleable support before it becomes an action (Llama-3.1-8B, §5.3).

The correct general statement is therefore neither "the failure is in the model" nor "the
failure is in the interface":

> **The serving interface is simultaneously part of the measurement function and part of
> the agent's effective action space.**

Under RL this is not a metaphor. The interface sits literally between policy and
environment, `θ → Y → I(Y) → a → o → r`; when `I(Y) = ∅` holds for every tool-intended
`Y`, the experience distribution contains no tool-mediated trajectory at all.

*Censoring* is borrowed **by analogy**, and we state the analogy's limit. In survival
analysis censoring is a threshold on a continuous variable; a parser is instead a
deterministic filter on output *format*. What carries over — and what motivates the term —
is the structure of the inference error: the observation is not noisy or missing at random,
it is **systematically unobservable on one side of a boundary**, and the boundary
correlates with the very quantity being measured. We show below that it
correlates *positively* with model scale: the more capable the model, the more of its
behaviour is censored.

### 1.2 What is and is not new here

The specific observation that Qwen2.5-Coder does not emit `hermes`-format tool calls was
reported in vLLM issue #32926 (2026-01-23) together with a `<tools>` parser proposal; that
issue was closed as *not planned* and labelled *stale*, so the trap is still live today.
§2 gives full attribution. What this paper adds is measurement, decomposition and
consequence:

1. **Quantification.** How much capability the censoring removes, across a 21× scale range,
   under a criterion strict enough to exclude schema echo and pseudo-code.
2. **Layer disentanglement.** Separating *model intent* / *emitted format* / *server
   parse* / *actual execution* / *multi-turn rescue* into five independently measured
   quantities. Reporting only the third is the standard practice we argue against.
3. **A cross-family taxonomy.** Four families failing at four different layers —
   template, parser, schema, token — with different remedies and different detectability.
4. **RL contamination, measured.** Direct instrumentation showing the censored interface
   yields zero tool executions in training rollouts, and an argument that this renders the
   tool-using branch unreachable by policy gradient.
5. **A repair loop.** How much is recovered by installing a correct adapter, and what
   protocol gap survives the repair.

We deliberately foreground (2) and (4). A reader who takes this as a catalogue of
misconfigurations has read it as a systems note; the claim we defend is that **the
composition above is what benchmarks actually measure, and that attributing its output to
the model is a systematic error with a measurable magnitude.**

---

## 2. Related work

**Evaluation validity under nuisance factors.** A line of work shows that reported LLM
capability is sensitive to factors that ought to be irrelevant. Sclar et al. (*FormatSpread*)
quantify how prompt-formatting choices alone move benchmark scores by margins comparable to
model differences; Mizrahi et al. (*State of What Art?*) show single-prompt evaluation is
unreliable and argue for multi-prompt protocols; Dehghani et al. (*The Benchmark Lottery*)
document how benchmark and configuration choices determine which method appears to win.
Our contribution is adjacent but distinct: those works vary an input the evaluator controls
and observe score movement. **We identify a component of the measurement apparatus — the
serving stack's tool-call parser — that can zero out a capability signal entirely, without
any error, and whose distortion grows with model scale.**

**Constrained decoding and its costs.** Tam et al. (*Let Me Speak Freely?*) report that
format restrictions and structured-output constraints can degrade reasoning. This bears
directly on §5.3, where enabling `strict: true` eliminates a 23% role-confusion failure and
*improves* pass rate (49 → 61). We do not claim constraints are free: our result is that on
this task the constraint's benefit — admitting 23 previously-discarded items into execution
— outweighs any reasoning cost, and we report the CoT control (§5.3) where added reasoning
made matters worse. Whether constraints hurt elsewhere in the same stack is untested here.
Guided-decoding machinery (Willard & Louf, *Outlines*) underlies both `strict` and vLLM's
`required` mode.

**Tool-use benchmarks.** BFCL, τ-bench (Yao et al.), ToolLLM/ToolBench (Qin et al.),
API-Bank (Li et al.) and Toolformer (Schick et al.) evaluate or train tool use. To our
knowledge these report tool-call rates as parsed by the serving layer. §5.2 shows that
number can be zero while 80/100 well-formed calls are emitted, which — if the pattern
replicates on their stacks — would affect any absolute tool-use rate they report.

**Protocols and agent loops.** ReAct (Yao et al.) is the text protocol we compare against.
vLLM (Kwon et al.) is the serving stack; its tool-calling documentation specifies distinct
per-family configuration, which §5.1 shows is not optional. verl (Sheng et al.,
*HybridFlow*) provides the RL training stack; importantly its AgentLoop performs **its own**
tool-call parsing (`verl/experimental/agent_loop/tool_parser.py`, adapted from vLLM v0.9.1),
so a vLLM-side parser plugin does not affect training rollouts — a distinction that cost us
one wasted experiment design (§5.6).

**Prior report of the Qwen case.** vLLM issue #32926 (2026-01-23) documents that
Qwen2.5-Coder was not trained on tool-call tokens, emits ```json blocks or bare JSON without
format instructions, and reaches 98–100% compliance with a `<tools>` few-shot template plus
a dedicated parser. The issue was closed as *not planned* and labelled *stale*. §5.2
reproduces this baseline exactly — zero `<tool_call>`/`<tools>` tags at every scale — and
extends it to scale quantification, other families, and training. A related verl issue
(#4124) reports the same role-confusion signature we characterise in §5.3: the model calls
the task function (`solve_equation`) and the agent loop raises `KeyError`.

**RL with tool feedback.** GRPO (Shao et al.) and RLOO (Ahmadian et al.) are the estimators
we train with; EvalPlus (Liu et al.) and KodCode supply held-out evaluation and training
problems respectively. We are not aware of prior work measuring interface-layer censoring
of tool trajectories inside an RL rollout, nor connecting it to which capability the policy
gradient can reach.

## 3. Setup

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

## 4. Positive control: the pipeline works

Before attributing a zero to a model we verify the measuring instrument. On the *same live
server*, same model, same tools, same system prompt, changing only `tool_choice`:

```
tool_choice = auto      → finish_reason = stop        tool_calls = 0
                          content: "好的，下面是一个实现 count_distinct 的示例代码：```python…"

tool_choice = required  → finish_reason = tool_calls  tool_calls = 1
                          name = run_tests
                          arguments = {"code": "def count_distinct(nums):\n    return len(set(nums))"}
```

Under `required`, `hermes` parses correctly — server flag, parser and chat template are all
functional. The zero under `auto` is therefore model behaviour, not misconfiguration.

**Bound on this control.** `required` uses constrained decoding, so it establishes only
that *the pipeline is capable*; it does not establish that the model spontaneously knows the
`<tool_call>` format. The two must be stated separately.

---

## 5. Results

### 5.1 Four families, four failure layers — and the consequential ones are silent

| Family | Layer | Symptom | Detectable in advance? | Remedy |
|---|---|---|---|---|
| DeepSeek-Coder | template | chat template never injects tools | ✅ template inspection | none |
| Qwen2.5-Coder | parser | emits ```json, not `<tool_call>` | ❌ template check **false-positives** | dedicated adapter |
| Llama-3.1-8B | schema | calls the *task function* as a tool (23%) | ❌ requires argument inspection | `strict: true` |
| Mistral-7B-v0.3 | token | repeats `[TOOL_CALLS]` → HTTP 400 | ⚠️ errors, but only sometimes | **none found** |

Mistral is the one family whose failure announces itself (HTTP 400 on a repeated
`[TOOL_CALLS]` marker). The other three — and a missing `--enable-auto-tool-choice` aside —
return HTTP 200 with `tool_calls: []`, which is byte-identical to a model that declined to
call the tool. **The claim is not that all failures are silent; it is that the silent ones
dominate and are the hardest to attribute.**

**A false-positive capability check.** The natural pre-flight test — does
`apply_chat_template(tools=…)` render the schema into the prompt? — correctly rejects
DeepSeek-Coder but **passes Qwen2.5-Coder**, whose `tokenizer_config.json` inherits Hermes
scaffolding from Qwen2.5-Instruct while the checkpoint was never trained on it.
**Protocol support is a per-checkpoint property and is not inferable from the chat
template**: templates are inherited, training is not.

### 5.2 Censoring grows monotonically with scale

Qwen2.5-Coder, `tool_choice: auto`, `hermes` (the documented recommendation for Qwen2.5),
n=100 per size:

The three intent columns are **nested, not mutually exclusive**: `tight ⊂ strong`
(tight adds the requirement that `arguments.code` be a string literal containing real
Python), and `weak` counts items with JSON structure but *no* `run_tests` name — disjoint
from both. At 32B, `strong = 100` and `tight = 80` means every item contained a named call
and 80 of them carried a usable payload.

| Size | Server-parsed | **tight** | strong (⊇ tight) | weak (disjoint) | Final pass |
|---|---|---|---|---|---|
| 1.5B | 0 | 0 | 0 | 27* | 31 |
| 3B | 0 | 4 | 5 | 28 | 37 |
| 7B | 0 | 21 | 22 | 1 | 53 |
| 14B | 0 | 36 | 42 | 3 | 53 |
| **32B** | **0** | **80** | 100 | 0 | 68 |

Across a 21× range the server reports a flat zero while well-formed emitted calls rise to
80/100. **The more capable the model, the more of its behaviour is censored.** Under the
loose criterion every one of 32B's 100 items contains a `{"name":"run_tests",…}` structure.

\* The 1.5B row uses the **optional** tool instruction. Under the **mandatory** prompt
used for training (§5.6) the same checkpoint gives weak = 66 and unparsable = 46: coercion
produces more JSON-shaped debris, not more valid calls. The two numbers describe different
prompts, not a discrepancy.

Raw-tag counts confirm the mechanism and reproduce hanXen's baseline exactly: across all
sizes, `<tool_call>` and `<tools>` appear **zero** times; ```json and `"name":` appear
throughout.

**Offline re-parse matrix.** To exclude "your extractor is simply better than hermes", we
re-parsed the *same stored bytes* under four rules (`analysis/reparse_matrix.py`, no GPU):

| Arm (server-parsed = 0) | server | hermes replay | `<tools>` replay | bare JSON | tight |
|---|---|---|---|---|---|
| Qwen-1.5B | 0 | 0 | 0 | 0 | 0 |
| Qwen-3B | 0 | 0 | 0 | 5 | 4 |
| Qwen-7B | 0 | 0 | 0 | 22 | 21 |
| Qwen-14B | 0 | 0 | 0 | 42 | 36 |
| Qwen-32B | 0 | **0** | 0 | **100** | **80** |

The `hermes` column is zero at every scale: the model never produces that format. The
bare-JSON and tight columns are not: the calls exist. Their difference from the server
column is precisely what the format mismatch removes.

† For arms where the server *did* parse, offline content-only re-parsing is **undefined,
not zero**: a successful parse moves the call into the structured `tool_calls` field and
leaves `content` empty. Those rows are reported as N/A rather than 0.

### 5.3 A 23% failure that survived three controls and fell to the fourth

Llama-3.1-8B under FC issues a tool call on 97/100 items, but on 23 of them the call names
**the function the task asks it to implement** — `can_form_word`, `check_password_strength`,
`floyd_warshall` — with that function's own parameters as arguments
(`{"tiles": "aabbcc", "word": "abc"}`). A re-run restricted to those 23 items under matched
configuration reproduced 23/23 as wrong-tool calls.

*This also corrects a measurement of our own*: because the probe did not originally verify
the tool name, these 23 were counted as initiations. **Llama's true `run_tests` initiation
rate is 74/100, not 97/100** (`ERRATA.md` §2).

| Alternative explanation | Control | Empty-argument rate | Verdict |
|---|---|---|---|
| schema lacks a parameter description | rich vs terse | 23 → 22 (*p*=1.000) | rejected |
| ReAct merely adds a reasoning step | FC + Thought scaffold | 23 → **59** (*p*<0.001, **worse**) | rejected |
| server not configured per documentation | + official chat template | 23 → 22 (*p*=0.125) | rejected |
| **schema constraint not enabled** | **+ `strict: true`** | **23 → 0 (*p*=0.0001)** | **accepted** |

#### The constraint penetrates to task performance

Adding the official template alone changes nothing; adding `strict` on top of it changes
everything. Isolating the single variable:

| Arm | Calls issued | Wrong-tool | Executed | Turn-1 | Final |
|---|---|---|---|---|---|
| terse (parser only) | 97 | 23 | 74 | 34 | 49 |
| + official template | 95 | 22 | 73 | 33 | 44 |
| + official template **+ `strict`** | 98 | **0** | **97** | **46** | **61** |
| ReAct (reference) | 98 | 0 | 97 | **61** | 80 |

The official template moves nothing (22 wrong-tool, 73 executed, 33/44 — marginally worse
than terse). `strict` alone drives the chain
**constraint → valid execution (73→97) → task performance (33→46 turn-1, 44→61 final)**.
This is not a formatting fix: 23 items that previously produced no executable code now
execute and become repairable. All 98 calls under `strict` carry code containing
`def`/`class`/`import`; none was converted into a well-named call with a malformed payload.

**A three-rung ladder, both rungs significant.** Paired McNemar on the identical 100 items:

| Comparison | b/c | *p* | Attributable to |
|---|---|---|---|
| terse-FC → strict-FC (49 → 61) | 15/3 | **0.0075** | **interface repair, +12** |
| strict-FC → ReAct (61 → 80) | 27/8 | **0.0019** | **protocol, +19** |
| terse-FC → ReAct (49 → 80) | 38/7 | 3.1e-06 | both, +31 |

Roughly **two fifths of what looked like a protocol effect was an interface effect**, and
each rung is separately significant. This delivers the quantification claim on a
fully-configured family without relying on the 32B measurement.

**Where the residual gap lives.** With `strict` enabled, FC and ReAct have *identical*
execution mechanics — 98 calls, 0 wrong-tool, 97 executed on both sides — yet turn-1 pass
differs 46 vs 61. Once the interface is equalised, **what remains is first-draft code
quality under the protocol, not tool-use mechanics.**

*(One `strict` call is issued and syntactically valid but yields no test result: item 53
imports `seaborn`, unavailable in the sandbox, so the harness collected zero tests. It is
counted as a failure, not silently dropped.)*

The rich schema explicitly warned *"this tool is not the function you are asked to
implement; do not pass the task function's parameters"* and supplied an example — the
confusion persisted at 22/100. The CoT control is the more interesting negative: asking the
model to reason about the task's signature and edge cases **before** constructing the call
raised confusion from 23 to 59, because the parameters it had just reasoned about were the
ones it then supplied. **On an interface with an ambiguous role boundary, more reasoning
amplifies misuse rather than correcting it.**

The defensible statement is narrower than either "model defect" or "configuration bug":
**the model exhibits role confusion under unconstrained function calling, and the interface
constraint determines whether that confusion can be expressed as an environment action.**
We cannot show from these experiments that the underlying tendency persists once
constrained — only that it stops reaching the action space. vLLM's `auto` mode is
unconstrained by default.

**This section is itself an instance of the paper's claim.** We approached with explicit
suspicion, ran three single-variable paired controls, and still wrote a configuration
problem into a draft as a model deficiency.

### 5.4 Main table, unified configuration

All arms: true `max_tokens=2048`, identical 100 items, `temperature=0`.

| Model | ReAct turn-1 | ReAct final | FC turn-1 | FC final | b/c | *p* | FC arm |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 61 | **80** | 46 | 61 | 27/8 | **0.0019** | official template + strict |
| Qwen2.5-Coder-7B | 57 | **74** | 53 | 62 | 17/5 | **0.0169** | dedicated adapter |
| Mistral-7B-v0.3 | 26 | 33 | N/A | N/A | N/A | N/A | — |

**Mistral has no admissible FC arm.** All four configurations produce request errors —
parser only: 2; official template: **42**; parser + strict: 3; official template + strict:
**39**. The correct statement is not "the two protocols do not differ on Mistral" but
**Mistral-7B cannot produce an error-free 100-item run under function calling**, including
under vLLM's own recommended configuration, which *raises* the error rate from 2% to 42%.
Its failure is at the token level — a repeated `[TOOL_CALLS]` marker that the `mistral`
parser rejects — and `strict: true`, which fixes Llama's schema-level failure, does not
help (42 → 39).

### 5.5 Repair loop: how much comes back

Qwen2.5-Coder-7B, identical items, `tool_choice: auto` throughout, changing only the
adapter combination:

| Arm | Parsed | Mean turns | ≥2 turns | Turn-1 | Final | Rescued |
|---|---|---|---|---|---|---|
| ReAct | 95 | 1.92 | 40 | 57 | **74** | 17 |
| FC + hermes | **0** | 0.96 | **0** | 53 | 53 | **0** |
| FC + dedicated adapter | **84** | 1.82 | **37** | 53 | **62** | **9** |

The mechanism returns unambiguously: parse 0→84, items reaching a second turn 0→37,
multi-turn rescues 0→9. **Multi-turn repair is restored from literal zero.**

**Internal validity check.** The two FC arms have **identical turn-1 pass (53 vs 53)**. The
adapter can only affect what happens *after* the first turn, and the data show exactly that
— evidence that we are measuring the interface and not the model.

**Why Llama's turn-1 does move (34→46) while Qwen's does not.** This is not an
inconsistency; it is the taxonomy making a prediction. Qwen's failure sits at the **parser
layer**, downstream of generation: the first-turn completion is produced and then
discarded, so repairing the parser cannot change it — hence 53 vs 53. Llama's failure sits
at the **schema layer**, inside generation: the model spends its entire first turn calling
the wrong function, so the first-turn output is itself destroyed and repairing the
constraint restores it — hence 34 → 46. **Different failure layers leave different
downstream signatures, and the direction of the turn-1 effect identifies the layer.**

The pass-rate gain is smaller and **not significant** (53→62, *p*=0.093).

#### Does the residual gap survive conditioning on a successful parse?

The adapter recovers 84/100 parses while ReAct reaches 95/100, so part of the remaining
74-vs-62 gap may simply be the 11 items FC never got to attempt. Restricting to items
**both** arms parsed successfully:

| Family | Items both parsed | ReAct | FC | b/c | *p* |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B | 83 | 63 | 56 | 11/4 | **0.119 (n.s.)** |
| Llama-3.1-8B | 96 | 78 | 60 | 25/7 | **0.0021** |

**The answer differs by family, and we report it as such.** On Llama — where `strict`
equalises execution at 97/97 — the protocol gap is unambiguous. On Qwen the gap does *not*
survive conditioning: most of 74 vs 62 is attributable to the residual 11-item parse
deficit, not to the protocol. **The residual protocol effect is established on one family,
not two**; §5.4's headline comparison should be read with this decomposition attached.

#### Access to feedback vs. use of feedback

Where the gap does exist, it is not about reaching the second turn (Qwen):

| Arm | Reached turn ≥2 | Rescued | Repair success given feedback |
|---|---|---|---|
| ReAct | 40 | 17 | **42%** |
| FC + adapter | 37 | 9 | **24%** |
| FC + hermes | **0** | **0** | undefined |

Access is nearly equal (40 vs 37); what differs is what the model does with the
observation once it arrives. **The interface determines whether feedback arrives; the
protocol appears to affect whether it is used** — though on Qwen this difference is within
the non-significant band above, so we state it as an observation, not a result.

**Attribution bound.** The dedicated solution changes **parser, chat template and few-shot
simultaneously**; it can only be called an *adapter combination*. A template × parser 2×2
ablation was not run. We additionally rewrote the template's few-shot examples, which
originally invoked `get_weather` — a tool absent from our schema, which would have induced
calls to a non-existent tool and inflated both initiation and empty-argument counts.
Original and modified hashes are recorded.

**Adapters are not a universal switch.** The same adapter recovers 84/100 at 7B and **1/100
at 3B**: the 3B checkpoint ignores the `<tools>` few-shot entirely (zero `<tools>` tags in
100 items, 99 direct code) while still solving the task at a comparable rate (final 53).
Adapter effectiveness is non-monotonic in scale, reinforcing §5.1's per-checkpoint claim.

### 5.6 Censoring reaches RL training

Same model, data, hyper-parameters and seed; **prompt strength matched** — the FC arm uses a
mandatory instruction (*"you must call `run_tests` to verify your code; do not answer
directly"*):

| Arm | Steps | `num_turns/mean` | Tool time | Tool calls / rollout |
|---|---|---|---|---|
| FC (`tool_agent` + hermes) | 10 | **2.000** (constant) | **0.00 s** | **0.000** |
| ReAct (`react_agent`) | 3 | 5.883 | 12.21 s | **2.052** |

Instrumentation is placed at the two agent-side call sites (`CodeTool.execute`,
`ReActAgentLoop._run_tests`). It **must not** be placed in the sandbox entry point: the
reward function evaluates final code through the same function, which would merge reward
evaluation into the tool-call count. The FC counter file is empty; the ReAct file has 788
lines, all tagged `react`.

Three independent signals agree: `num_turns` pinned at its minimum of 2, tool time exactly
0.00 s, dedicated count 0.

**Wording bound.** What is measured is **parser-accepted and executed calls = 0**, *not*
"the model never attempted to call". FC rollouts were not saved, so direct answers,
malformed calls and dropped intent cannot be separated from the training logs. A probe
replication under the identical mandatory prompt on the same 1.5B checkpoint (n=100) gives:
**0 well-formed calls, 54 direct answers, 46 unparsable outputs, 66 items containing JSON
fragments but not one naming `run_tests`.** Notably the mandatory instruction *degraded*
the model — final pass 31 → 15 versus the optional prompt — leaving it able neither to call
the tool nor to answer cleanly.

**Why gradient cannot fix this.** Valid calls are zero **from step 1**, and the reward
admits direct answers (FC `critic/rewards/mean` rose 0.233→0.281). The policy gradient
therefore contains **no signal pushing toward tool use**: the tool branch has a
zero-sample return, while direct answering is continuously reinforced. The model cannot
discover the tool path by exploration, because every attempt is discarded at the protocol
layer and no positively-rewarded tool trajectory ever exists to be reinforced.
**A mandatory prompt plus GRPO does not cross this cold start.**

> Interface-induced censoring does not merely weaken a learning signal.
> It renders a policy branch **unreachable by gradient**.

**Causal chain.**

```
Qwen2.5-Coder-1.5B emits nothing hermes can parse
  (probe replication under the training prompt: 0 parsed, 0 tight, 66 JSON fragments)
    → parser-accepted calls in training rollouts = 0 (num_turns pinned at 2)
      → Observation never returns; trajectories degenerate to single-turn
        → RL optimises only a single-turn signal
          → multi-turn rescues flat at 6–9/540; 91–94% of new passes are turn-1
```

The last two links are the training result of §1; the middle two are measured here.

#### Why no repaired-FC training arm exists

The natural control is *repaired* FC rather than ReAct. We attempted to construct it and
failed at three successive layers; each attempt is a measurement, so we report them.

**(i) A vLLM parser plugin does not reach the training loop.** verl's AgentLoop performs its
own tool-call parsing (`verl/experimental/agent_loop/tool_parser.py`, registered as
`hermes`, adapted from vLLM v0.9.1). `multi_turn.format=hermes` selects *that* parser, not
the server's. Attaching a plugin to the vLLM rollout server therefore changes nothing in
training — a distinction worth stating because the two look identical from the outside.

**(ii) A verl-side parser recovers calls, but none of them are the tool.** We implemented a
`qwen2_5_coder` parser inside verl's registry accepting `<tools>`, ```json fences and bare
JSON. On the exact training condition (1.5B, mandatory prompt, n=100) it lifts recoverable
calls from **0/100 (hermes) to 52/100**. Of those 52, **zero name `run_tests`**:

| Category | n | % |
|---|---|---|
| wrong name, `arguments` carry complete code | **0** | 0% |
| wrong name, `arguments` are the task function's parameters | **36** | 69% |
| wrong name, executable code present elsewhere in the body | 16 | 31% |

Every call targets the function the task asks the model to *write* —
`can_form_word({"tiles","word"})`, `check_password_strength({"password"})` — the same
signature as Llama's failure in §5.3, now at a 5× smaller model. **Name normalisation
cannot repair this: no call carries code to normalise.**

**(iii) A role-disambiguation few-shot makes it worse, not better.** Since the failure is
semantic rather than syntactic, we targeted it directly: a system prompt containing the
wrong call and the right call side by side, with an explicit warning that the task function
is not a tool. Same model, same 100 items, same server; only the system prompt changes.

| | recoverable calls | **naming `run_tests`** | final pass |
|---|---|---|---|
| mandatory prompt (baseline) | 52 | **0** | 15 |
| + role-disambiguation few-shot | **64** | **0** | **13** |

The intervention made the model *more fluent at emitting the wrong call* and slightly worse
at the task. This is the third independent instance of §5.7's pattern.

**(iv) Constrained decoding is unavailable in this stack.** The remedy that works at
evaluation time — forcing the tool choice, or schema-constrained decoding — is not exposed
by verl 0.9.0's vLLM rollout path (its only `strict` key belongs to the profiler).

We therefore state the limitation precisely: **no executable repaired-FC condition was
available through the existing verl 0.9.0 vLLM rollout configuration.** This is a property
of the training stack, not a proof that FC is irreparable in principle; forced tool choice
via a rollout backend that exposes guided decoding remains untested. Accordingly, **ReAct
serves as a positive-control interaction channel, not as a parser-repair control**, and the
comparison changes protocol and interface together. We flag this rather than let the
comparison be read as a clean single-variable intervention.

**Scope.** This holds for Qwen2.5-Coder + `hermes`. It does not generalise to "FC training
never receives tool feedback". Whether the §5.5 adapter also restores the training signal
was not tested. The demonstration is 10 and 3 steps — a **mechanism** result, not an
outcome comparison.

### 5.7 Pressure on a broken interface makes things worse

Two independent controls point the same way. Adding a reasoning scaffold to FC raised
role confusion from 23 to **59** (§5.3): asked to analyse the task function's signature
before acting, the model supplied exactly those parameters to the tool. Replacing the
optional tool instruction with a mandatory one (*"you must call `run_tests`; do not answer
directly"*) on Qwen2.5-Coder-1.5B moved parser-accepted calls not at all — still 0/100 —
while final pass fell from **31 to 15** and unparsable outputs rose from ~0 to 46/100.

Neither intervention touches the interface, and both make the observable outcome worse.
Coercion cannot substitute for a working channel: the model is pushed away from the
answer it could give and toward a call that will be discarded anyway.

**Practical reading.** When an agent appears reluctant to use tools, strengthening the
instruction is the cheapest thing to try and the most likely to mislead — it can degrade
task performance while leaving the tool-call count at zero, which looks like a model that
is both unwilling *and* incapable.

### 5.8 Sampling variance, and an inverted noise structure

Llama-3.1-8B, `temperature=0.6`, n=100 × 3 seeds:

| | seed 1 | seed 2 | seed 3 | admissible |
|---|---|---|---|---|
| ReAct turn-1 | 53 | 56 | 43 | 3/3 |
| ReAct final | 72 | 72 | 72 | 3/3 |
| FC turn-1 | 44 | 44 | 45 | — |
| FC final | 62 | 66 | ~~57~~ | **2/3** |

FC seed 3 contains one request error — the model emitted parallel tool calls and vLLM
returned *"This model only supports single tool-calls at once"* (documented: Llama 3 does
not support parallel calls) — so it is inadmissible by §3.2. With two admissible FC arms we
report the interval [62, 66] and **no standard deviation**. Worst case: ReAct's lowest (72)
versus FC's highest (66) = **+6**; the direction is consistent across all admissible
repetitions.

**Inverted noise structure.** ReAct's turn-1 varies widely (43–56, sd 6.8) while its final
converges; FC's turn-1 is nearly constant (44–45, sd 0.58) while its final disperses. The
mechanism is compensation in the rescue count — 19 / 16 / **29**, largest where turn-1 was
worst — and in all three seeds the count of "passed at turn 1 but failed finally" is **0**:
the repair loop is monotone.

**A necessary caveat.** The three identical finals (72) are a **coincidence of totals, not
identical item sets**: pairwise Jaccard is 0.71–0.80 (three-way intersection 57, union 87),
i.e. 15–24 items differ between seeds. **sd = 0.00 must not be read as "ReAct is perfectly
stable."** The accurate statement is that **multi-turn repair absorbs turn-1 sampling
noise, stabilising the total while its composition keeps moving** — a property that matters
more for deployment than for leaderboard position.

---

## 6. Discussion

**Report the composition, not the model.** Any claim of the form "model *M* does not use
tools" is, as measured, a claim about `M × protocol × serialization × parser × stack`.
We recommend that agent evaluations report at minimum: (i) server-parsed calls,
(ii) emitted-but-unparsed calls under a stated criterion, (iii) request errors, and
(iv) the exact serving configuration. Reporting only (i) is the practice this paper argues
against, and (ii) is where the 21× scale trend lives.

**Pre-flight, don't post-hoc.** A missing `--enable-auto-tool-choice` is the only failure
that announces itself. We therefore treat a `tools`-bearing request returning HTTP 200 —
*and* a server-parsed call under `tool_choice: required` — as a required pre-flight for any
FC experiment. Template inspection alone is insufficient (§5.1). We ship this check as a
30-line script (`preflight_toolcall.py`): it issues one canonical request, asserts
`tool_calls` is non-empty with the expected `name` and parseable `arguments`, then repeats
under `tool_choice: required` as a positive control. It runs in seconds and would have
caught every silent failure in this paper.

**Interface repair precedes protocol comparison.** On Llama the ReAct-vs-FC gap was 31
points before the `strict` fix and 19 after — **two fifths of an apparent protocol effect
was an interface effect**, and both components are separately significant (*p*=0.0075 and
*p*=0.0019). On Qwen the decomposition goes further: conditioning on items the server
parsed, the remaining gap is **not** significant (*p*=0.119), i.e. essentially all of the
observed difference is attributable to the interface. Comparisons of agent protocols that
do not first exhaust the serving configuration matrix are not measuring protocols — and
even after repair, the residual must be re-tested conditionally before it is called a
protocol effect.

**A cold-start hazard for agentic RL.** §5.6's mechanism generalises beyond this stack:
whenever (a) tool trajectories are censored at rate ~1 and (b) the reward admits a
tool-free path, the tool-using branch has no sampled return and cannot be reinforced.
Prompt-level coercion does not help, and we observed it actively hurting (final pass 31→15).
Practitioners should verify a **non-zero tool-execution count in the first training step**
before interpreting any multi-turn result.

---

## 7. Limitations

1. **Sample.** One task family (decontaminated KodCode), 100 items per arm, and the items
   are `clean[:100]` — **not a random sample**. Pairing is exact but generalisation to the
   corpus is unestablished.
2. **Single-sample main table.** §5.4 is `temperature=0`, one sample. Variance is estimated
   only for Llama-3.1-8B, and only 2 of 3 FC arms are admissible.
3. **Multiple comparisons.** Roughly a dozen McNemar tests, **uncorrected**. The main
   effects (~1e-3) survive Bonferroni; marginal results (*p*=0.093, 0.125) do not.
4. **Adapter attribution.** §5.5 changes parser, template and few-shot together; the
   2×2 ablation isolating the parser was not run.
5. **Scale.** Up to 32B; no frontier models. Single A800.
6. **Training evidence is a mechanism demonstration** (10 and 3 steps), not a 150-step
   outcome comparison. We claim gradient-unreachability, not a quantified final gap.
7. **The residual protocol effect rests on one family.** Conditioned on successful parsing
   it holds on Llama-3.1-8B (*p*=0.0021) but not on Qwen2.5-Coder-7B (*p*=0.119), and
   Mistral admits no comparison. Claims about "ReAct beats function calling" should be
   read as family-specific.
8. **Instrumentation errata.** Seven ReAct arms recorded `max_tokens=2048` while actually
   running at 1024 (an asymmetry that favoured FC; re-running changed final pass by ≤1
   item); Llama's initiation rate required manual correction from 97 to 74; five arms are
   error-bearing and admissible only for error-rate census. All are enumerated in
   `ERRATA.md` and none is silently absorbed into a reported number.

---

## 8. Conclusion

Tool-call interfaces censor agent trajectories, and the censoring is silent, family-specific,
scale-increasing, and — in reinforcement learning — capable of making an entire policy
branch unreachable by gradient. Repairing the interface restores the mechanism but recovers
only part of the outcome gap, and a genuine protocol difference remains underneath.

The most uncomfortable finding is methodological. We ran three single-variable paired
controls on a 23% failure and concluded it was a model deficiency; a fourth control, taken
from a sentence in the serving documentation, reduced it to zero. **The mechanism this
paper describes claimed us as one of its instances**, which is the strongest argument we can
offer that reporting server-parsed tool-call rates as model capability is not a safe default.

---

## 9. Reproducibility

```bash
# analysis and figures (no GPU)
pip install -r requirements.txt
python analysis/intent.py          # the single intent criterion, all four counts
python runs/final/final_table.py   # main / variance / error-rate tables (N/A enforced)
python figures/make_figs.py

# probe (needs a vLLM server)
python probe_react_full.py --model <path> --port 8000 --n 100 \
  --protocol {react,fc} --strength {optional,mandatory} \
  --fc-schema {terse,rich,strict} --parser-adapter cross_family \
  --temperature 0.0 --seed 0 --out traj.jsonl
python validate_arms.py            # per-arm admissibility: lines / rc / n_err / provenance / script hash
```

Model snapshots must be pinned: `tokenizer_config.json` inheritance is the mechanism
behind §5.1's false-positive capability check, so a family's behaviour can change with a
repository revision. Every model path, its HF revision, and the serving flags used are
recorded per arm in the trajectory provenance and in `runs/final/by_config/README.md`.

Trajectories for all 49 full-length arms, the training logs, the errata, and the
per-configuration index are in `runs/final/`. Third-party parser: hanXen
(`1b92150`, Apache 2.0), few-shot examples rewritten from `get_weather` to `run_tests`;
original retained with both hashes recorded.

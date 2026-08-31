5. Results

### 5.1 Four families, four failure layers — and the consequential ones are silent

| Family | Layer | Symptom | Detectable in advance? | Remedy |
|---|---|---|---|---|
| DeepSeek-Coder | template | chat template never injects tools | ✅ template inspection | none |
| Qwen2.5-Coder | parser | emits a fenced JSON block, not `<tool_call>` | ❌ template check **false-positives** | dedicated adapter |
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

### 5.9 Multi-turn repair stabilises the total while its composition keeps moving

Two independent observations, from different axes, show the same structure.

**Across sampling seeds** (§5.8): ReAct's final pass is 72 / 72 / 72 at `temperature=0.6`,
yet the sets of solved items have pairwise Jaccard 0.71–0.80 — 15–24 items differ between
seeds. Turn-1 pass varies widely (43–56, sd 6.8) and the rescue count compensates inversely
(19 / 16 / 29).

**Across training steps** (§5.10): between step 15 and step 30 of the ReAct run, 85 of 542
first-turn completions changed verbatim and 6 items flipped outcome — 3 gained, 3 lost —
leaving turn-1 and final pass numerically identical (340 / 346 at both checkpoints). The
policy is demonstrably moving; the aggregate is not.

The common structure is that **the repair loop absorbs perturbation upstream of it**.
Whatever the first draft does — vary with sampling, drift with training — items that fail
and are repairable get repaired, and the total lands in the same place. This has two
consequences worth separating:

*For measurement.* An aggregate pass rate is a **poor instrument for detecting change in a
multi-turn agent**: it is stabilised by the very mechanism under study. Item-level set
comparison (Jaccard, flip counts) detects movement that the total conceals. We report both
throughout, and note that had we reported only totals we would have concluded — twice, on
two different axes — that nothing was happening.

*For deployment.* Stability of the total under sampling noise is a property practitioners
actually want, and it is contributed by the multi-turn loop rather than by the base policy.
It is invisible in single-turn evaluation.

**Bound.** Both observations are on one model family and one task. The identical totals are
in part coincidence — three seeds landing on exactly 72, two checkpoints landing on exactly
340/346 — and we do not claim the aggregate is invariant, only that it is far less sensitive
than its components.

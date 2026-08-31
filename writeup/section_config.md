# Settled sections (independent of the A3/A4 results)

> 中文版见 [`section_config_zh.md`](section_config_zh.md).

## 1. Each model family requires its own mutually incompatible configuration, and any missing piece masquerades as "insufficient capability"

Evaluating function calling through vLLM's OpenAI-compatible endpoint, each of four families
demands a different server-side configuration. The decisive point: **a missing setting does not
raise an error.** The server returns 200 as usual, `tool_calls` is an empty array, and
everything downstream sees only "the model did not call a tool."

| Family | Officially required configuration | Symptom when missing | Source |
|---|---|---|---|
| Llama 3.1 / 3.2 | `--tool-call-parser llama3_json` **+ `--chat-template tool_chat_template_llama3.1_json.jinja`** | malformed arguments (official wording: "may generate incorrectly formatted arguments") | vLLM Tool Calling docs |
| Mistral (HF format) | `--tokenizer-mode hf --config-format hf --load-format hf --tool-call-parser mistral` **+ `tool_chat_template_mistral_parallel.jinja`**; plus the constraint that `tool_call_id` must be exactly 9 characters | 400 / failure when passing results back across turns | ibid. |
| Qwen2.5-Instruct | `--tool-call-parser hermes` (the template already carries the Hermes style) | — | ibid. |
| **Qwen2.5-Coder** | **no working configuration**: this variant was not trained on tool-call tokens and has no native tool-call format | **silently returns 0 calls** | vLLM issue #32926 |
| DeepSeek-Coder | none: the chat template never injects tools at all | detectable (the template contains no tools) | measured in this work |

A missing `--enable-auto-tool-choice` is the only setting that fails loudly (`tool_choice:auto`
returns 400); everything else degrades silently. **Every FC experiment therefore needs a
preflight**: send one request carrying `tools`, assert HTTP 200, then assert that the server
actually parsed a call out of the response.

## 2. "The chat template references tools" does not mean "the model was trained for tool calling"

The common way to decide whether a checkpoint supports FC is to check whether
`apply_chat_template(tools=...)` renders the schema into the prompt. This work started there
too (`supports_native_fc()`). It correctly classifies DeepSeek-Coder as unsupported (the
template does not inject), but gives a **false positive on Qwen2.5-Coder**:

- Qwen2.5-Coder's `tokenizer_config.json` inherits the Hermes tool scaffold from
  Qwen2.5-Instruct, so the template check returns True
- but the variant was never trained on tool-call tokens, so the model does not emit
  `<tool_call>` markers
- the hermes parser finds no markers, so `tool_calls: []`
- recorded as a **0% initiation rate**

**Protocol support is a per-checkpoint property, and it cannot be inferred from the chat
template.** Templates are inherited; training is not.

## 3. A strict server-side parser silently underestimates model capability

Qwen2.5-Coder-7B, hermes parser, `tool_choice: auto`, n=100:

| Criterion | Count |
|---|---|
| `tool_calls` parsed by the server | **0/100** |
| Well-formed under the **strict** criterion | **21/100** |
| Loose strong evidence (contains `"name":"run_tests"` or a `<tools>` marker) | 22/100 |
| Weak heuristic (JSON structure only; may be an ordinary answer) | 1/100 |

A typical instance:

```json
{"name": "run_tests", "arguments": {"code": "def flatten_list(...)..."}}
```

Correct function name, correct `arguments` structure, correct `code` field — the sole
non-compliance is that it is not wrapped in `<tool_call>` markers.

The underestimate is **21 percentage points** under the strict criterion, **with no error
signal of any kind**.

> **A note on consistency of measurement.** Every intent statistic in this report comes from a
> single classifier function in `analysis/intent.py` (three tiers: tight / strong / weak),
> shared by the prose, the tables and the figures. Earlier drafts carried figures such as
> 18 / 23 / 24 that came from ad-hoc scripts written at different times; **this function is
> authoritative and supersedes all of them.**

**Methodological claim**: when evaluating an agent's protocol compliance, "the model attempted
a call" and "the server parsed a call" must be reported as two independent metrics. Reporting
only the latter records an engineering configuration problem as a model capability defect.
This work added a protocol-agnostic intent detector (`_detect_tool_intent`) for that purpose.

## 4. A taxonomy of three failure modes

| Type | Representative | Detectable? | Presentation |
|---|---|---|---|
| template does not inject | DeepSeek-Coder | yes — a template check suffices | no tool awareness |
| template injects but the model was not trained | Qwen2.5-Coder | no — the template check is a false positive | invents its own JSON format, swallowed by the parser |
| trained but wrong arguments | Llama-3.1-8B | no — requires inspecting argument contents | initiates calls (97%) but 23% pass the task function's own parameters as tool arguments |
| trained but non-compliant output | Mistral-7B-v0.3 | partly — some errors, some silent | repeats `[TOOL_CALLS]` triggering 400; 17% well-formed but unmarked; hallucinates tool names outside the schema |

## 5. Llama's schema confusion is not sloppy prompt writing

The most immediate objection: the `code` parameter originally had no `description`, so the
model fills it wrongly because it was not told clearly. A paired control (same model, same 100
items, same server configuration, changing only the tool schema):

| Schema | Initiated | Empty args | Final pass |
|---|---|---|---|
| terse (`code: {"type": "string"}`, no description) | 97 | **23** | 49 |
| rich (full parameter descriptions + counter-example warning + code sample) | 97 | **22** | 49 |

McNemar 10 vs 10, **p = 1.000**.

The rich version's tool description says explicitly: "this tool is not the function you are
being asked to implement; do not pass the task function's parameters into it — the single
parameter `code` is the complete source text you wrote," and the parameter description supplies
`"def add(a, b):\n    return a + b"` as an example. Under that explicit warning, 22/100 still
fill in the task function's parameters (`['s']`, `['tiles','word']`,
`['columnName','filePath']`), and they are the same parameter names that the terse version got
wrong.

### 5b. Nor is it a server-configuration problem

vLLM's documentation recommends, for Llama 3.1, `--tool-call-parser llama3_json` **together
with** `--chat-template examples/tool_chat_template_llama3.1_json.jinja`. The arms above supply
only the parser. Re-running with the official template added (paired, same 100 items):

| Arm | Initiated | Empty args | turn-1 pass | Final pass |
|---|---|---|---|---|
| ReAct | 98 | **0** | 61 | **80** |
| FC underconfigured (parser only) | 97 | 23 | 34 | 49 |
| FC full official configuration (+ chat template) | 95 | **22** | 33 | 44 |

Official vs underconfigured: McNemar b=1, c=6, **p = 0.125, no difference**; the wrongly-filled
parameter names are the same set as before. ReAct vs official-configuration FC: b=42, c=6,
**p < 0.0001**.

### 5c. Three negations

| Alternative explanation | Control | Empty-argument rate | Verdict |
|---|---|---|---|
| the schema's parameters lack descriptions | rich vs terse schema | 23 -> 22 (p=1.000) | ruled out |
| ReAct merely adds a thinking step | FC + Thought scaffold | 23 -> 59 (p<0.001, worse) | ruled out |
| the server does not follow the official recommendation | + official chat template | 23 -> 22 (p=0.125) | ruled out |

### 5d. A fourth test overturned the conclusion of the first three

From the vLLM documentation: to impose schema-level constraints under `tool_choice="auto"`,
`VLLM_ENFORCE_STRICT_TOOL_CALLING=true` (the default) is required **and at least one tool must
carry `strict: true`**; otherwise vLLM extracts calls from plain text, and "arguments may
occasionally be malformed or **violate the function's parameter schema**."

None of the arms above carried `strict: true`. With it added (official template + strict,
paired, same 100 items):

| Arm | Initiated | Empty args | turn-1 pass | Final pass |
|---|---|---|---|---|
| ReAct | 98 | 0 | 61 | **80** |
| FC terse (parser only) | 97 | 23 | 34 | 49 |
| FC official template | 95 | 22 | 33 | 44 |
| **FC official template + `strict:true`** | 98 | **0** | 46 | **61** |

McNemar, strict vs official template: b=18, c=1, **p = 0.0001**. Function names are
`run_tests` on 98/98, `_fc_arg_err` is empty, and zero calls carry wrong parameter names.

**Corrected conclusion**: those 22% are not the model's role confusion. **They are a
configuration artefact of the missing `strict: true`** — precisely the situation the vLLM
documentation describes in so many words. None of the three negations in 5a–5c touched the
actual cause, and their conclusion that "this is a model problem" **is withdrawn**.

**But the protocol gap does not disappear**: ReAct 80 vs fully-configured FC 61, McNemar b=27,
c=8, **p = 0.0019**. The gap narrows from 31 points to 19, and remains significant.

### 5e. This section is itself an instance of the thesis

We came in with an explicit suspicion, ran three single-variable paired controls (schema
descriptions, reasoning scaffold, official chat template), and still wrote up a configuration
problem as a "model capability defect" until the fourth control caught it. **Tool-interface
mismatch masquerading as a model lacking agentic ability — this work is among that mechanism's
victims.** Any evaluation claiming "model X cannot use tools" should first exhaust the
server-side configuration matrix.

## 6. ReAct's advantage does not come from the Thought scaffold — adding it makes FC worse

The second obvious objection: ReAct's system prompt contains a `Thought:` step and FC's does
not, so the comparison measures "protocol + chain of thought" against "protocol alone," with
two variables tied together.

The control: give FC a system prompt demanding a Thought first ("first write out your reasoning
in a Thought: what the problem asks for, what the edge cases are, what algorithm you intend to
use. Once you have thought it through, call the run_tests tool..."), changing nothing else.
Llama-3.1-8B, paired on the same 100 items:

| Arm | Initiated | Empty args | turn-1 pass | Final pass | Mean turns | Median turn-1 code length |
|---|---|---|---|---|---|---|
| ReAct | 98 | **0** | 61 | **80** | 1.83 | 509 |
| FC terse | 97 | 23 | 34 | 49 | 1.59 | 272 |
| FC + Thought | 95 | **59** | 12 | **24** | 0.84 | 182 |

McNemar: FC+Thought vs FC terse, b=4, c=29, **p < 0.001, significantly worse**.

If ReAct's advantage came from the chain of thought, moving the chain of thought onto FC should
narrow the gap. Instead the gap widens from 31 points (80 vs 49) **to 56 points** (80 vs 24),
and schema confusion rises from 23% to 59%.

**Mechanism**: the Thought instruction makes the model first analyse the task function's
signature, parameters and edge cases; carrying that context into the construction of a tool
call, it drops the very parameters it just reasoned about straight into the tool's arguments.
**Against an interface whose role definition is ambiguous, strengthening reasoning amplifies
misuse rather than correcting it.**

**Boundary**: this shows only that this particular CoT wording harms FC; it does not generalise
to "any reasoning scaffold is harmful." It is nonetheless sufficient to rule out the
alternative explanation that "ReAct's advantage is just an extra thinking step," which is what
this control arm exists to do.

## 7. Positive control: 0% is not a pipeline fault

The first thing to suspect about "the server parsed 0 calls" is a misconfiguration. A positive
control was therefore run **on the same live server** — same model, same tools, same system
prompt, changing only `tool_choice`:

```
tool_choice=auto      -> finish=stop        tool_calls=0
                         content: "好的，下面是一个实现 count_distinct 的示例代码：```python..."

tool_choice=required  -> finish=tool_calls  tool_calls=1
                         name=run_tests
                         args={"code": "def count_distinct(nums):\n    return len(set(nums))"}
```

(The assistant content is reproduced verbatim; our task prompts are in Chinese, and model
output in transcripts is not translated.)

Under `required`, hermes parsing works perfectly (function name, `arguments` structure and
`code` field all correct), proving that the server flag, the tool parser and the chat template
are all functional. The 0% under `auto` is therefore model behaviour, not a configuration
fault.

**But this control's force has a boundary**: `required` uses constrained decoding to **force**
output into tool-call format, so it proves only "the pipeline works," not "the model
spontaneously knows to use `<tool_call>` markers." The two must be stated separately.

The behaviour distribution under `auto` (Qwen2.5-Coder, n=100 per size):

| Behaviour | 1.5B | 3B |
|---|---|---|
| server parsed a call | 0% | 0% |
| invented its own JSON call format (parser rejects) | 27% | 33% |
| abandoned the call and wrote code directly | 74% | 68% |

The two categories sum to roughly 100% (one overlap), meaning that under `auto` the model has
only two states: **attempting a call in a self-invented format**, or **abandoning the tool
entirely and answering directly**. There is no third. The latter can still be extracted and
executed by `cross_family` (final pass 31%/37%), but the tool result never flows back and
multi-turn repair does not exist at all (mean turns 0.95).

## 8. Closing the loop: how much of the swallowed capability does a dedicated adapter return?

Qwen2.5-Coder cannot be parsed by hermes (§§2, 3, 7). The community proposed a `<tools>`-marker
solution for this (vLLM issue #32926 / hanXen); that proposal was closed as *not planned*
(stale label), meaning the trap is still live today. This section uses it as a positive
comparison.

**Note the attribution boundary.** The solution replaces **parser, chat template and few-shot
examples** all at once, so it can only be called an "adapter combination" — the improvement
cannot be attributed to the parser alone. A strict decomposition needs a template x parser 2x2
ablation, which was not run. In addition, the original template's few-shot examples call
`get_weather`, which is not this experiment's single legal tool (`run_tests`) and would induce
calls to a non-existent tool; they were rewritten to `run_tests`, the original is retained as
`.orig`, and SHA256 for both versions is recorded. The probe also gained function-name
validation (anything other than `run_tests` is recorded as `unknown_tool`, excluded from L1 and
not executed).

Qwen2.5-Coder-7B, paired on the same 100 items, `tool_choice: auto` held fixed throughout:

| Arm | Parsed | Mean turns | Items reaching >=2 turns | turn-1 pass | Final pass | Multi-turn rescues |
|---|---|---|---|---|---|---|
| ReAct | 95 | 1.92 | 40 | 57 | **74** | 17 |
| FC + hermes | **0** | 0.96 | **0** | 53 | 53 | **0** |
| FC + dedicated adapter | **84** | 1.82 | **37** | 53 | **62** | **9** |

Request errors: 0. `unknown_tool`: 0.

**At the mechanism level there is no ambiguity**: parsed 0->84, items reaching a second turn
0->37, multi-turn rescues 0->9. **Multi-turn repair is restored from literally zero.**

**Internal-validity check**: the two FC arms have **identical turn-1 pass (53 vs 53)**. The
adapter can only affect what happens after the first turn and should not affect the quality of
the model's first program — and the data says exactly that. This corroborates that what this
section measures is the interface, not model capability.

**But the pass-rate gain is not significant**: 53 -> 62 (+9 points), McNemar b=16, c=7,
**p = 0.093**.

**And ReAct still leads**: 74 vs 62, b=17, c=5, **p = 0.0169**; consistent with the Llama side
(ReAct 80 vs official-configuration + strict FC 61, p = 0.0019).

### Three layers of conclusion

1. **What the interface mismatch swallowed**: on Qwen, 0->84 parsed, 0->37 multi-turn, 0->9
   rescued; on Llama, the 23 argument-contaminated calls zeroed by `strict: true` (§5d)
2. **The protocol gap that remains after the interface is repaired**: consistent across both
   families, ReAct > fully-configured FC
3. **The benefit of repairing the interface is real but bounded**: every mechanism is restored;
   only part of the pass rate comes back

**The interface problem masked the protocol problem; only by repairing the interface does the
latter become visible.**

## 9. Training side: interface mismatch means RL receives no tool feedback at all

The first eight sections are all at the evaluation layer. This one measures the same mechanism
directly **inside the rollout phase of RL training**.

Model under training: Qwen2.5-Coder-1.5B, verl 0.9.0, batch 16 x rollout.n 8 = 128 trajectories
per step; the two arms share every hyperparameter, data item and seed apart from the protocol.
**System-prompt strength is aligned on both sides** (FC uses the mandatory wording: "**you must
first call the run_tests tool to verify your code**; do not answer directly").

| Arm | Steps | `num_turns/mean` | Tool time | Tool calls | Per rollout |
|---|---|---|---|---|---|
| FC (tool_agent + hermes) | 10 | **2.000** (unchanged first to last) | **0.00 s** | **0** | **0.000** |
| ReAct (react_agent) | 3 | 5.883 (5.34->6.35) | 12.21 s | 788 | **2.052** |

Both arms rc=0, no crashes. The unequal step counts do not affect the conclusion — **ten steps
did not produce a single call**.

**Credibility of the count**: verl 0.9.0's step metrics contain **no** tool-call count (only
`timing_s/agent_loop/tool_calls`, which is **elapsed seconds**; reading it as a count is a hard
error). Instrumentation was therefore added at the two agent-side call sites —
`CodeTool.execute` (FC path) and `ReActAgentLoop._run_tests` (ReAct path) — each writing one
labelled line per call. **The counter must not go in `sandbox.run_tests`**: the reward function
evaluates the final program through that same function, which would mix reward evaluation into
the tool-call count. Result: the FC-side counter file has 0 lines; the ReAct side has 788, all
labelled `react`, so the provenance is uncontaminated.

Three mutually independent pieces of evidence agree: `num_turns` pinned at its minimum of 2,
tool time exactly 0.00 s, and a dedicated counter at 0.

### The boundary of the wording

What this section measures is that **tool calls accepted and executed by the parser number
zero** — **not** that "the model never attempted a call." FC3 did not save the raw rollout
text, so the following three cases cannot be distinguished:

1. the model output code directly (no attempt);
2. the model output a malformed `<tool_call>` (attempted, non-compliant format);
3. the model had call intent, but the parser discarded it.

The model under training is Qwen2.5-Coder-**1.5B**, and the §3 probe measured at that scale:
well-formed under the strict criterion **0/100**, loose strong evidence **0/100**, weak
heuristic (JSON structure only) **27/100**. Cases 2 and 3 very likely exist, but at that scale
**well-formed calls were zero to begin with**.

Correct phrasing: *FC produced 0 parser-accepted / executed tool calls despite mandatory
instructions and correctly injected tool schemas.*
Incorrect phrasing: *The model never attempted to call tools.*

### This strengthens rather than weakens the main conclusion: a protocol/parsing cold start

Valid calls are **zero from step 1 onward**, while the reward function permits scoring by
emitting code directly (this FC arm's `critic/rewards/mean` went 0.233 -> 0.281 — it really was
rising). So the policy gradient contains **no signal at all pushing the policy toward tool
use**: the tool path has a constant zero-return sample, while the direct-answer path keeps
collecting positive reward.

**A mandatory prompt plus GRPO is not enough to cross this cold start**: the model cannot
discover the tool path by exploration, because every attempt is discarded at the protocol layer
and no positively-rewarded tool trajectory has ever existed to reinforce. This is the concrete
mechanism by which interface mismatch acts during training, and it is far worse than "some
signal is missing" — **it renders a policy branch unreachable during optimisation.**

### Full causal chain

```
Qwen2.5-Coder-1.5B (the model being trained) does not emit hermes-parseable calls
  (probe replication in this section, same mandatory prompt: parsed 0/100, well-formed
   under the strict criterion 0/100, 66/100 contain a JSON fragment and none writes
   "name":"run_tests")
  -> tool calls accepted by the parser during training rollout = 0 (this section;
     num_turns constant at 2)
    -> Observation never flows back, trajectories degenerate to a single turn
      -> RL can only optimise on single-turn signal
        -> in evaluation, multi-turn rescues stay flat at 6-9/540 throughout, and
           91-94% of newly-passing items pass on turn 1
```

**A note on which numbers belong where**: the first link uses figures for **the 1.5B model
actually being trained**. The curve in §3 rising to 80/100 is a **cross-scale phenomenon**
(32B) and must not be substituted into this training's causal chain — at 1.5B, well-formed
calls under the strict criterion are zero.

The last two links come from this work's 150-step x 3-run training results (EvalPlus,
GRPO x 2 seeds + RLOO); the middle two are measured directly in this section. **"Interface
mismatch contaminates RL training" thereby moves from inference to measurement.**

**Scope**: the conclusion holds for the specific combination Qwen2.5-Coder + hermes parser. It
does not generalise to "all FC training receives no tool feedback." For this model, switching
to §8's dedicated adapter raises the parse rate from 0% to 84%; whether the training side
recovers likewise was not measured.

## 10. Main table (unified configuration) and Mistral's special status

Three core pairings, all genuinely at `max_tokens=2048`, same 100 items, `temperature=0`:

| Model | ReAct turn-1 | ReAct final | FC turn-1 | FC final | b/c | p | FC arm |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 61 | **80** | 46 | 61 | 27/8 | **0.0019** | official template + strict |
| Qwen2.5-Coder-7B | 57 | **74** | 53 | 62 | 17/5 | **0.0169** | dedicated adapter |
| Mistral-7B-v0.3 | 26 | 33 | **N/A** | **N/A** | **N/A** | **N/A** | see below |

**ReAct figures are taken from the v14 arms** (genuinely 2048). The earlier v3 versions
actually ran at 1024 (see `ERRATA.md`); re-running gives 80 / 74 / 33 against the 1024
versions' 80 / 74 / 32 — **the empirical effect of the asymmetric limit is zero**, because the
median ReAct response on these programming problems is about 1000 tokens.

### Mistral has no FC arm usable for pass-rate comparison

All four configurations were tried, and **every one produces request errors**:

| Configuration | n_err |
|---|---|
| `--tool-call-parser mistral` only | 2 |
| official template (`tool_chat_template_mistral_parallel.jinja` + the three hf flags) | **42** |
| parser + `strict: true` | 3 |
| official template + `strict: true` | **39** |

Arms containing missing data get **no pass rate and no p-value** (an early version of this work
computed p=0.648 on such an arm anyway; that number is withdrawn). The correct statement is
therefore not "the two protocols do not differ on Mistral," but:

> **Mistral-7B-v0.3 cannot produce a single error-free 100-item run under function calling.**
> All four configurations fail, including vLLM's officially recommended one. The failure is not
> "a low score" but **an inability to finish at the interface layer**. Every error is the same
> kind: a repeated `[TOOL_CALLS]` marker triggering 400 (and the officially recommended
> parallel template raises the error rate from 2% to 42%).

## 11. Sampling variance

Every conclusion above rests on single-sample runs at `temperature=0`. Taking the
most-completely-configured pair (Llama-3.1-8B, ReAct vs official template + strict) and
repeating three times at `temperature=0.6`:

| Arm | seed 1 | seed 2 | seed 3 | valid arms |
|---|---|---|---|---|
| ReAct turn-1 | 53 | 56 | 43 | 3/3 |
| ReAct final | 72 | 72 | 72 | 3/3 |
| FC turn-1 | 44 | 44 | 45 | — |
| FC final | 62 | 66 | ~~57~~ | **2/3** |

FC seed 3 contains one request error (the model emitted parallel tool calls and vLLM reported
*"This model only supports single tool-calls at once"* — the official documentation states
Llama 3 does not support parallel calls), marked invalid by the rule in the previous section.
With only 2 valid FC arms remaining, **a standard deviation cannot be reported**; only the
interval [62, 66].

**Worst case**: ReAct's lowest 72 vs FC's highest 66 = **+6 points**, and the direction is
consistent across every valid replicate.

### The two protocols have opposite noise structures

| | turn-1 pass | Final pass |
|---|---|---|
| ReAct | 43–56 (sd 6.8) | 72 / 72 / 72 |
| FC | 44–45 (sd 0.58) | 62 / 66 (/ 57 invalid) |

**ReAct's turn-1 fluctuates widely yet its final converges; FC's turn-1 is extremely stable yet
its final diverges.** The mechanism is in the rescue counts: ReAct's three seeds rescue
19 / 16 / **29** respectively, compensating inversely with turn-1 quality — the run with the
worst first turn (43) rescues the most. And across all three seeds, items that passed on turn 1
but failed at the end number **0**: the multi-turn loop is monotone.

**A necessary qualification**: the three identical final values of 72 are **a coincidence of
totals, not the same set of items**. Pairwise Jaccard between the three seeds' passing sets is
only 0.71–0.80 (intersection 57, union 87), differing on 15–24 items. A standard deviation of
0.00 **must not be stated as "ReAct's results are perfectly stable."** The accurate statement
is: **multi-turn repair absorbs turn-1 sampling noise, holding the total steady while the
composition still moves.**

The implication is larger for engineering deployment than for scores — the value of multi-turn
is not only a higher number, but **a lower sensitivity of the result to sampling**.

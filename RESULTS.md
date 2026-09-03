# The silent fraction grows with model scale — and reaches the policy gradient

**Tool-call interfaces censor agent trajectories before anything downstream sees them**

> 中文版见 [`RESULTS_zh.md`](RESULTS_zh.md).
>
> This document is the top-level summary. Full evidence, control design and the exact
> boundaries of every claim are in [`writeup/section_config.md`](writeup/section_config.md)
> (11 sections); data errata in [`runs/final/ERRATA.md`](runs/final/ERRATA.md);
> figures in [`figures/`](figures/).
>
> The previous framing ("three-layer negation") was refuted by this work itself and is kept
> in `internal/superseded/RESULTS_v1_refuted_20260829.md` as a record.

---

## 0. One sentence

Configuring four model families for function calling on vLLM **following official
documentation**, every family fails silently, each at a different layer. The failure presents
as "the model does not know how to use tools"; it is in fact an interface mismatch. That
mismatch contaminates both **evaluation numbers** and **RL training signal** — in our own
training it made tool calls unreachable by the gradient.

## 1. Starting point: something that did not add up in a training result

Qwen2.5-Coder-1.5B, verl GRPO / RLOO, 150 steps x 3 independent runs, held-out EvalPlus
(HumanEval+ 319 / MBPP+ 677, minus 2 known-defective items, denominators 540 / 454).

| run | multi, final | turn-1 | repair |
|---|---|---|---|
| GRPO seed42 | 64.8% -> 67.6% (p=0.064) | +2.6% (p=0.093) | +4.6% (**p=0.030**) |
| GRPO seed1 | 64.3% -> 67.0% (p=0.078) | +3.0% (p=0.053) | +3.7% (p=0.097) |
| RLOO seed42 | 64.6% -> 67.2% (p=0.077) | +2.6% (p=0.082) | +0.9% (p=0.786) |

The main effect sits at **p = 0.06–0.08, not significant at 0.05**. What credibility it has
comes from three independent runs agreeing in direction with the same magnitude (+2.6–2.8%).

**The real problem is in the decomposition.** Of the newly-passing items, **91–94% passed on
the first turn**; the count of items rescued by turn 2 or later stayed flat at **6–9 / 540**
from step 0 to step 150.

> RL improved draft quality. Multi-turn debugging ability **did not change at all**.

That is where this work begins: **why does multi-turn not move?**

## 2. The answer is not in the model; it is in the interface

Probing family by family on 100 decontaminated KodCode items (one item set shared by every
experiment; verified distinct-set count = 1) yields four mutually distinct failures:

![Four-family failure taxonomy](figures/fig2_family_taxonomy.png)

| Family | Failure layer | Symptom | Repairable? |
|---|---|---|---|
| DeepSeek-Coder | template | the chat template never injects tools at all | no (a template check finds it) |
| **Qwen2.5-Coder** | **parser** | emits ```json rather than `<tool_call>`; server silently returns an empty array | yes — dedicated adapter, 0 -> 84 |
| **Llama-3.1-8B** | **schema** | 23% hallucinate a call to **the task's own function** instead of `run_tests` | yes — `strict: true`, 23 -> 0 |
| **Mistral-7B** | **token** | repeats `[TOOL_CALLS]`, triggering HTTP 400 | no — all four configurations fail |

**The decisive property: only a missing `--enable-auto-tool-choice` raises a hard error.
Everything else degrades silently.** The server returns 200 as usual, `tool_calls` is an empty
array, and everything downstream sees only "the model did not call a tool."

## 3. The silent underestimate grows monotonically with scale

![Intent–parse gap](figures/fig1_intent_parse_gap.png)

Qwen2.5-Coder, `tool_choice=auto`, hermes parser (vLLM's documented recommendation for
Qwen2.5), across a 21x scale range from 1.5B to 32B:

| Scale | Parsed by the server | Well-formed calls the model actually emitted |
|---|---|---|
| 1.5B | 0 | 0 |
| 3B | 0 | 4 |
| 7B | 0 | 21 |
| 14B | 0 | 36 |
| **32B** | **0** | **80** |

The criterion for "well-formed" is strict: `{"name":"run_tests", "arguments":{"code":"<string
literal containing real Python>"}}`, excluding schema echoes and illustrative pseudocode.

> **The larger the model, the more capability is silently discarded.** At 32B the model wrote
> a fully usable tool call on 80 of 100 items, and the evaluation recorded "this model never
> uses tools."

A positive control rules out "the configuration is simply broken": switching `tool_choice` to
`required` on the same live server, hermes parsing works perfectly (function name, `arguments`
and `code` all correct). But that control only proves **the pipeline works** — it does not
show the model would use it spontaneously, because `required` goes through constrained
decoding.

## 4. Llama's failure took four controls to characterise

Initial observation: under FC, Llama-3.1-8B treats **the very function the task asks it to
implement** as a tool on 23% of items, filling `arguments` with that function's parameters
(`{"tiles":..., "word":...}`). Four alternative explanations, each ruled out in turn:

| Alternative explanation | Control | Empty-argument rate | Verdict |
|---|---|---|---|
| the schema's parameters lack descriptions | rich vs terse | 23 -> 22 (p=1.000) | ruled out |
| ReAct merely adds a thinking step | FC + Thought scaffold | 23 -> **59** (p<0.001, **worse**) | ruled out |
| the server was not configured as officially recommended | + official chat template | 23 -> 22 (p=0.125) | ruled out |
| **schema constraint was not enabled** | **+ `strict: true`** | **23 -> 0 (p=0.0001)** | **holds** |

The fourth control overturned the conclusion the first three supported. The correct statement
is: **the model genuinely does suffer role confusion, and `strict: true` suppresses it via
constrained decoding.** The defect is real; it only becomes visible when the constraint is
absent — and vLLM's `auto` mode does not constrain by default.

**This section is itself an instance of the paper's thesis.** We came in with an explicit
suspicion, ran three single-variable controls, and still wrote up a configuration problem as a
"model capability defect" until the fourth control caught it.

## 5. Main table (unified configuration)

All arms genuinely at `max_tokens=2048`, same 100 items, `temperature=0`:

| Model | ReAct turn-1 | ReAct final | FC turn-1 | FC final | b/c | p |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 61 | **80** | 46 | 61 | 27/8 | **0.0019** |
| Qwen2.5-Coder-7B | 57 | **74** | 53 | 62 | 17/5 | **0.0169** |
| Mistral-7B-v0.3 | 26 | 33 | N/A | N/A | N/A | N/A |

All four of Mistral's FC configurations **contain request errors** (2 / 42 / 3 / 39); arms with
missing data get no pass rate and no *p*-value. The correct statement is not "the two protocols
do not differ" but **Mistral cannot produce a single error-free 100-item run under FC**.

**The protocol gap persists after the interface is repaired**, and it does so consistently
across two families.

## 6. Closing the loop: how much does the right adapter recover?

Qwen2.5-Coder-7B, same 100 items, `tool_choice=auto` held fixed, changing only the adapter
combination:

| Arm | Parsed | Mean turns | >=2 turns | turn-1 pass | Final pass | Multi-turn rescues |
|---|---|---|---|---|---|---|
| ReAct | 95 | 1.92 | 40 | 57 | **74** | 17 |
| FC + hermes | **0** | 0.96 | **0** | 53 | 53 | **0** |
| FC + dedicated adapter | **84** | 1.82 | **37** | 53 | **62** | **9** |

**Every mechanism is restored** (parsed 0->84, multi-turn 0->37, rescues 0->9); **the pass rate
only partially recovers** (53->62, p=0.093, not significant).

**Internal-validity check**: the two FC arms have **identical turn-1 pass (53 vs 53)** — the
adapter can only affect what happens after the first turn, and the data says exactly that.

## 7. Training side: interface mismatch makes tool calls unreachable by the gradient

![Training-side causal chain](figures/fig3_training_causal.png)

Same model, data, hyperparameters and seed, with **prompt strength aligned on both sides** (FC
uses the mandatory wording: "*you must first call the run_tests tool to verify your code*"):

| Arm | Steps | num_turns/mean | Tool time | Tool calls per rollout |
|---|---|---|---|---|
| FC (tool_agent + hermes) | 10 | **2.000** constant | **0.00 s** | **0.000** |
| ReAct (react_agent) | 3 | 5.883 | 12.21 s | **2.052** |

The counters are instrumented at the two agent-side call sites, `CodeTool.execute` and
`ReActAgentLoop._run_tests`. **They must not be placed in `sandbox.run_tests`** — the reward
function evaluates the final program through that same function.

**Boundary of the claim**: what is measured is that **calls accepted and executed by the parser
number zero**, not that "the model never tried." Replicating with the same mandatory prompt on
the same 1.5B as a probe (n=100): **0 well-formed calls, 54 direct answers, 46 unparseable, and
66 containing a JSON fragment of which none writes `"name":"run_tests"`**.

### Cold start

Valid calls are **zero from step 1 onward**, while the reward permits scoring by emitting code
directly. So the policy gradient contains **no signal at all pushing the policy toward tool
use** — the tool path has a constant zero-return sample. **A mandatory prompt plus GRPO is not
enough to cross this cold start**: the model cannot discover the tool path by exploration,
because every attempt is discarded at the protocol layer and no positively-rewarded tool
trajectory has ever existed to reinforce.

> **Interface mismatch renders a policy branch unreachable during optimisation.**

### Full causal chain

```
Qwen2.5-Coder-1.5B (the model being trained) does not emit hermes-parseable calls
  (S7 probe replication, same mandatory prompt: parsed 0/100, well-formed under the
   strict criterion 0/100, 66/100 contain a JSON fragment and none writes "name":"run_tests")
  -> calls accepted by the parser during training rollout = 0 (S7: num_turns constant at 2)
    -> Observation never flows back, trajectories degenerate to a single turn
      -> RL can only optimise on single-turn signal
        -> S1: multi-turn rescues flat at 6-9/540, and 91-94% of newly-passing items
           pass on turn 1
```

**Scope**: this holds for the specific combination Qwen2.5-Coder + hermes. It does not
generalise to "all FC training."

**A note on which numbers belong where**: the first link above uses figures for **the 1.5B
model actually being trained**. The curve in §3 that rises to 80/100 is a **cross-scale
phenomenon** (32B) and must not be substituted into this causal chain — at 1.5B, well-formed
calls under the strict criterion are zero.

## 8. Sampling variance

Llama-3.1-8B, `temperature=0.6`, n=100 x 3 seeds:

| | seed 1 | seed 2 | seed 3 | valid arms |
|---|---|---|---|---|
| ReAct final | 72 | 72 | 72 | 3/3 |
| FC final | 62 | 66 | ~~57~~ | **2/3** |

FC seed 3 contains one parallel tool call rejected by vLLM (the official documentation states
Llama 3 does not support parallel calls) and is marked invalid by rule. With only 2 valid FC
arms we **report no standard deviation**, only the interval [62, 66]. The worst case,
ReAct 72 vs FC 66, is **+6**, and the direction is consistent across every valid replicate.

**The noise structures are opposite**: ReAct's turn-1 spans 43–56 (sd 6.8) yet its final
converges; FC's turn-1 spans 44–45 (sd 0.58) yet its final diverges. The mechanism is inverse
compensation in the rescue count (19 / 16 / **29**).

**A necessary qualification**: the three identical final values of 72 are **a coincidence of
totals** — pairwise Jaccard between the three seeds' passing sets is only 0.71–0.80, differing
on 15–24 items. **This must not be stated as "ReAct's results are perfectly stable."** The
accurate statement is that **multi-turn repair absorbs turn-1 sampling noise, holding the total
steady while the composition still moves**.

## 9. Relation to prior work

The specific phenomenon that Qwen2.5-Coder does not emit hermes-format tool calls was
**reported by hanXen on 2026-01-23 in vLLM issue #32926**, together with the `<tools>`
proposal. That issue was closed as *not planned* (stale label) — **the trap is still live
today**.

This work's contribution is not discovering the phenomenon. It is: (1) quantifying how much it
steals across a 21x scale range; (2) separating five layers — model intent / text format /
server parsing / actual execution / multi-turn rescue; (3) giving a cross-family failure
taxonomy; (4) showing it contaminates RL training signal as well; (5) closing the loop by
measuring how much the right adapter recovers.

## 10. Limitations

1. A single item bank (a KodCode subset), 100 items per arm, and **`clean[:100]` is not a
   random sample**
2. The main table is single-sample at `temperature=0`; variance is estimated only on
   Llama-3.1-8B, and there only 2 FC arms are valid
3. A dozen-odd McNemar tests with **no multiple-comparison correction**; the main effects
   (order 1e-3) survive Bonferroni, the marginal results (p=0.093 / 0.125) do not
4. The dedicated adapter changes **parser, chat template and few-shot examples together**, so
   it can only be called an "adapter combination" — attribution to the parser alone is not
   available, and the template x parser 2x2 ablation was not run
5. Model scale stops at 32B; no frontier models
6. The training side is a 10-step / 3-step mechanism demonstration; **no 150-step outcome
   comparison was run**
7. Data errata (7 arms with wrong provenance records, 5 arms inadmissible for pass rates) are
   in `ERRATA.md`

## 11. Reproduce

```bash
# probe (protocol layer)
python probe_react_full.py --model <path> --port 8000 --n 100 \
  --protocol {react,fc} --strength {optional,mandatory} \
  --fc-schema {terse,rich,strict} --parser-adapter cross_family \
  --temperature 0.0 --out traj.jsonl

# strict per-arm admissibility gate (line count / rc / n_err / provenance / script hash)
python validate_arms.py

# main and variance tables (arms with missing data are marked N/A and refuse to compute)
python runs/final/final_table.py

# the three main figures
python figures/make_figs.py
```

The training line is reproduced by `analyze_all.py` — the single source of every EvalPlus
number in this report.

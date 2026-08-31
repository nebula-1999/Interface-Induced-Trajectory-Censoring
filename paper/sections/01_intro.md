1. Introduction

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

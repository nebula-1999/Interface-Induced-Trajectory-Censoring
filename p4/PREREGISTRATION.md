# Preregistration v2: Prospective mechanism test under a fixed parser (written before the run)

> This is a translation of [`PREREGISTRATION.zh.md`](PREREGISTRATION.zh.md), first committed 2026-09-02T16:13:33+08:00 and last amended before the run at 2026-09-02T20:52:52+08:00. The predictions and dates are unchanged; see git history for provenance.

> v1 contained four material errors. They were corrected following external review; the
> corrections are listed at the end of this document. v1 remains in git history (commit
> `10eb9b7`) and has not been deleted.

## Hypothesis tested by this experiment (narrowly stated)

**H: Given a fixed parser, can the serialization convention in a chat template alone predict,
before execution, whether that parser will observe tool actions emitted by the model?**

The object of this test is the **"template → observability" prediction rule**, not the
censoring mechanism of the paper itself. The two must remain separate. Qwen2.5-Coder is direct
evidence that **template and training behavior can diverge**: it inherits the Instruct template
without having been trained on that format. The prediction may therefore fail, and such a
failure would falsify only the present form of this rule.

## Critical distinction from the paper's main result (must be stated in the paper)

| | Qwen2.5-Coder (§5.2/§5.3) | This experiment |
|---|---|---|
| How hermes was selected | **Following the official documentation selects it** | **We specify it deliberately** |
| Nature of the result | A real measurement-validity failure | A controlled mechanism test |

The vLLM documentation recommends `--tool-call-parser granite` for Granite, and GLM likewise
has its own path. Therefore, **Granite+hermes=0 is an expected incompatibility, not "another bug
in the official stack."** This experiment will never make the latter claim.

## Arms and predictions (committed before execution)

Fixed across arms: the same item set (`clean_ids.json[:100]`), `temperature=0`, `seed=0`,
`fc-schema terse`, `tool_choice=auto`, the same tool schema, and the same `max_tokens`.

### A. Positive control + counterfactual scale ladder: Qwen3 under a **matched** interface

The Qwen3 template injects tools and requests `<tool_call>`, matching hermes; the vLLM
documentation also demonstrates hermes for Qwen3. **Prediction: parsing rates at every size
will be significantly greater than zero, with no size-dependent increase in the silent
fraction.**

| Arm | Parser | Prediction |
|---|---|---|
| Qwen3-0.6B / 1.7B / 4B / 8B (14B may be added if disk permits) | hermes | Parsing rate > 0; undercount does not grow with scale |

**The value of this ladder is as a counterfactual control.** It directly addresses the largest
confound in §5.2: "the emitted count may rise with scale merely because larger models are
generally better at writing JSON." If Qwen3 under a **matched** interface does not show
increasing undercount, then "scale + matched interface ⇒ no increasing undercount" can be
contrasted with "scale + mismatched interface ⇒ increasing undercount," removing that
confound. This ladder **will not** provide a second 0→80-style censoring curve, nor is it
intended to.

### B. Negative arm + rescue control: Granite-3.1-8B

The template injects tools, but its envelope is `<|tool_call|>`, a different token from
hermes's `<tool_call>`.

| Arm | Parser | Prediction |
|---|---|---|
| Granite-3.1-8B | hermes | Parsing rate ≈ 0 |
| Granite-3.1-8B | **granite** | Parsing rate **> 0** |

**The rescue arm is required, not optional.** A zero from hermes alone cannot rule out that the
model simply never called a tool. Only if the same batch of outputs is parsed under the
dedicated parser can we attribute the zero to the serialization contract rather than missing
capability.

### C. GLM-4-9B: **not run in this round**

Its template represents tools in Chinese markdown under `# 可用工具` ("Available Tools"), without a
hermes-style envelope, so the prediction would be zero. However, **vLLM 0.27.1 includes only
`glm47_moe` (for GLM-4.7 MoE), with no parser for GLM-4-9B**, so a rescue control cannot be
constructed. By the standard above, a zero without a rescue arm is uninterpretable; this arm
will therefore not be run. It is listed to document the decision, not as an omission.

## What counts as falsification

- Qwen3 parsing rate ≈ 0 under hermes → falsifies "matching template envelope ⇒ observable."
- Granite parsing rate significantly > 0 under hermes → falsifies "mismatched template
  envelope ⇒ unobservable."
- Granite parsing rate remains ≈ 0 under the granite parser → its zero cannot be attributed to
  the contract, and the entire arm is invalidated.

If any of these occurs, **what is falsified is the prediction rule stated at the beginning of
this document**, not the paper's censoring mechanism.

## Limitations (recorded in advance)

1. **This survey inspected chat templates only, not each family's recommended parser.** It
   therefore **cannot** support the claim that "interface mismatches are widespread and
   agreement between template and parser is rare." That sentence from v1 was removed.
2. The criterion inspects template text, not the model's actual training. Divergence between
   template and training is a known phenomenon.
3. Granite has only 2B/8B and GLM-4-9B only one size. Qwen3 in group A is a scale ladder, but it
   uses a **matched** interface, so this experiment **does not provide** a second undercount
   ladder. P2 in its original sense remains open.
4. The default templates of six of the nine surveyed families (Phi-4 / InternLM2.5 / Yi-1.5 /
   StarCoder2 / DeepSeek-Coder-V2 / OLMo-2) do not inject OpenAI-style tools. **This does not
   mean they cannot censor.** Under the paper's taxonomy, failure to inject tools is itself a
   failure layer. The correct statement is: **they are unsuitable for testing censoring at the
   envelope/parser layer.**

## Errors in v1 (retained for comparison and provenance)

1. It said, "the six families that do not inject tools cannot exhibit censoring." **Wrong:**
   missing template injection is itself a failure layer in the paper's taxonomy. The corrected
   statement is "they are unsuitable for testing censoring at the envelope/parser layer."
2. It said, "interface mismatch is not a rare accident; what is rare is a template and parser
   that happen to agree." **Insufficient evidence:** only templates, not each family's
   recommended parser, had been surveyed. The sentence was removed.
3. It said, "the remaining three families have only one or two sizes each." **Factually
   wrong:** Qwen3 has a complete dense ladder at 0.6B/1.7B/4B/8B/14B/32B; all six sizes were
   verified to exist.
4. It said, "failure of either prediction would falsify the mechanism." **Overstated:** what
   would be falsified is the present form of the template-based prediction rule.

---

## Protocol-deviation record #1: disabling Qwen3 thinking mode (written before the batch run)

**Discovery:** the Qwen3 template defaults to `enable_thinking=true`. A preflight test on 0.6B
produced:

```
[auto]      tool_calls empty   finish=length
            content: '<think>\nOkay, I need to implement the add function...'
[required]  tool_calls=1  name='run_tests' ✅
```

The model spent all `max_tokens=2048` tokens in the thinking block and was **truncated before
reaching a tool call**, with `finish=length`. This is unrelated to the envelope; it is budget
exhaustion.

**Why this must be addressed:** running the original configuration would give Qwen3 a parsing
rate near zero, apparently falsifying the preregistered prediction ("even a family with a
matching envelope is censored"), although the cause is unrelated to the experiment's
independent variable. This is a confound, not a discovery.

**Treatment:** add `--default-chat-template-kwargs '{"enable_thinking": false}'` on the server.
This was chosen instead of modifying the probe for two reasons:

1. The probe remains **byte-for-byte identical** to the version that produced every historical
   arm, preserving cross-experiment comparability.
2. It is a server configuration, the same class of object studied in this paper, and introduces
   no new analysis code.

**Verification:** after disabling thinking, `finish=stop` and the thinking block disappears.
Under auto, 0.6B still returns code directly rather than calling a tool. This is bottom-scale
model behavior, consistent with Coder-1.5B (0) and Instruct-1.5B (1), and **does not by itself
falsify the prediction**; the prediction tests the shape of the ladder, not a single point.

**To be recorded in the paper:** all Qwen3 arms are served with `enable_thinking=false`; every
other parameter is matched item-by-item to the 50-item arms in §5.2. This difference must be
reported alongside the Qwen3 results.

## Final arm table under the disk constraint

The data disk has 75 GB free, so the ladder uses four sizes (14B would require another 28 GB and
is omitted):

| Arm | Parser | Thinking | Prediction |
|---|---|---|---|
| Qwen3-0.6B / 1.7B / 4B / 8B | hermes | off | Parsing rate > 0 and increasing with scale |
| Granite-3.1-8B | hermes | — | ≈ 0 |
| Granite-3.1-8B | granite | — | > 0 (rescue control) |

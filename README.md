# Interface-Induced Trajectory Censoring

**The more capable the model, the more of its tool-using behaviour the evaluation hides — and the hidden part is exactly what reinforcement learning cannot reach.**

Wenbo Wang · City University of Hong Kong · `wenbwang3-c@my.cityu.edu.hk`

> **Language.** The paper, this README, [`RESULTS.md`](RESULTS.md) and
> [`runs/final/ERRATA.md`](runs/final/ERRATA.md) are in English; each has a `_zh` counterpart.
> [`writeup/section_config.md`](writeup/section_config.md) — the full evidence and control
> design, 11 sections — is in English too. What stays Chinese is the interview narrative and
> the operational checklists: working records rather than claims, where translation would risk
> laundering the hedges. One Chinese passage is load-bearing and deliberately untranslated: the
> verbatim model output in §7 there and §4 of the paper, which is data.

![Intent–parse gap](figures/fig1_intent_parse_gap.png)

Across a 21× scale range the serving stack reports **zero** parsed tool calls at every size,
while well-formed calls the model actually emits rise to **80/100** at 32B. Nothing errors:
HTTP 200, `tool_calls: []`, a well-formed single-turn trajectory. **The observed tool-call rate is not a
property of the model alone; it is a property of the model–interface stack that
measures it.**

---

## What this is

We set out to train a multi-turn code agent with RL and could not explain our own curve:
pass@1 rose 2.6–2.8 points across three runs, but **91–94% of newly-passing items passed on
the first turn**, and items rescued by turn 2 or later stayed flat at **6–9 out of 540** from
step 0 to step 150. RL improved first drafts and left multi-turn debugging untouched.

The reason was not the model. **The tool was never successfully called** — and nothing in the
stack said so: HTTP 200, `tool_calls: []`, a well-formed single-turn trajectory.

Configuring four families for function calling on vLLM *following official documentation*,
each fails at a different layer, and the most consequential failures are silent:

| Family | Layer | Symptom | Remedy |
|---|---|---|---|
| DeepSeek-Coder | template | chat template never injects tools | none |
| Qwen2.5-Coder | parser | emits ```json, not `<tool_call>` | dedicated adapter (0→84) |
| Llama-3.1-8B | schema | calls *the task function* as a tool (23%) | `strict: true` (23→0) |
| Mistral-7B | token | repeats `[TOOL_CALLS]` → HTTP 400 | none found |

The same mismatch reaches RL training: under function calling a 10-step GRPO run executes
**zero** tool calls while `critic/rewards/mean` climbs from 0.233 to 0.281 — the dashboard
looks healthy while the branch being trained does not exist.

**But opening the channel is not sufficient.** A 75-step ReAct run executing **23,676** tool
calls leaves multi-turn rescues flat at 6–8/540, indistinguishable from runs where none
executed. The bottleneck is not singular, and we report that rather than the closure we
wanted.

## Paper

[`paper/`](paper/) is an Overleaf-ready LaTeX package (compile with **XeLaTeX**).
Full draft in Markdown: [`writeup/paper_draft.md`](writeup/paper_draft.md).

**Read [`runs/final/ERRATA.md`](runs/final/ERRATA.md) before citing any number.** It lists
seven arms whose recorded provenance is wrong, five arms inadmissible for pass-rate
comparison, and one retracted *p*-value.

## Reproduce

No GPU needed for analysis and figures:

```bash
pip install -r requirements.txt
python analysis/intent.py            # the single intent criterion (tight/strong/weak)
python analysis/reparse_matrix.py    # same bytes, four parser rules
python analysis/funnel.py            # five-layer funnel (Figure 1)
python runs/final/final_table.py     # main / variance / error-rate tables
python figures/make_figs.py
```

Probe (needs a vLLM server):

```bash
python probe_react_full.py --model <path> --port 8000 --n 100 \
  --protocol {react,fc} --strength {optional,mandatory} \
  --fc-schema {terse,rich,strict} --parser-adapter cross_family \
  --temperature 0.0 --seed 0 --out traj.jsonl
python validate_arms.py              # per-arm admissibility gate
```

## Take this with you

[`preflight_toolcall.py`](preflight_toolcall.py) — 98 lines, runs in seconds. It issues one
canonical request, asserts `tool_calls` is non-empty with the right `name` and parseable
`arguments`, then repeats under `tool_choice: required` as a positive control.
**It would have caught every silent failure in this paper.**

```bash
python preflight_toolcall.py --port 8000
```

## Layout

```
paper/                  LaTeX package (Overleaf-ready)
writeup/                full draft, evidence appendix, interview narrative
analysis/               single intent criterion, re-parse matrix, funnel, training curves
figures/                5 figures + generator
runs/final/             50 full-length arms, ERRATA, per-configuration index
validation/             human-validation pack, adjudication pack, scorer
fixes/                  AUDIT.md and the run scripts for each experiment generation
preflight_toolcall.py   the deliverable check
```

## Where to find the paper artifacts

| Claim or artifact in the paper | Repository location |
|---|---|
| the intent criterion | [`analysis/intent.py`](analysis/intent.py) |
| replaying vLLM's hermes extractor | [`analysis/failure_layer.py`](analysis/failure_layer.py) |
| the same bytes re-parsed under four rules | [`analysis/reparse_matrix.py`](analysis/reparse_matrix.py) |
| a prediction committed before the run | [`p4/PREREGISTRATION.md`](p4/PREREGISTRATION.md) |
| a 98-line preflight check | [`preflight_toolcall.py`](preflight_toolcall.py) |
| data errata | [`runs/final/ERRATA.md`](runs/final/ERRATA.md) |
| BFCL 2×2 | [`p1/`](p1/) |
| τ-bench | [`p5/`](p5/) |
| the rollout-path probe | [`p3/rollout_probe.py`](p3/rollout_probe.py) |

## Prior work

The specific observation that Qwen2.5-Coder does not emit `hermes`-format tool calls was
reported in [vLLM issue #32926](https://github.com/vllm-project/vllm/issues/32926)
(2026-01-23), closed as *not planned*. **We did not discover that phenomenon.** This work
quantifies it across scale, separates five layers of the failure, gives a cross-family
taxonomy, measures its effect inside RL training, and tests what a repair recovers.

## License

MIT (this work). The bundled Qwen2.5-Coder `<tools>` parser is Apache-2.0 from
[hanXen/vllm-qwen2.5-coder-tool-parser](https://github.com/hanXen/vllm-qwen2.5-coder-tool-parser)
at commit `1b92150`; its few-shot examples were rewritten from `get_weather` to `run_tests`
and the original is retained with both hashes recorded in `ERRATA.md`.

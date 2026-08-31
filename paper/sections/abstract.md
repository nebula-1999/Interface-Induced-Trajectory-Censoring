Abstract

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

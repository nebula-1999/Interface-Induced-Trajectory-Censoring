2. Related work

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

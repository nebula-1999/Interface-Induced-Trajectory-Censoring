6. Discussion

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

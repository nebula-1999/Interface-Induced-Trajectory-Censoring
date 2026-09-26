# Native FC: CPU preparation and bounded GPU smoke complete

Update: the authorized six-task smoke for base/broken/repaired and a separate
synthetic GPU feedback-path check passed. See
[GPU smoke receipt](NATIVE_FC_GPU_SMOKE_20260923.md). The historical plan below
describes the executed protocol; full evaluation remains pending, and the six
single-turn examples do not establish scientific performance or multi-turn cost.

## Completed on the no-GPU instance

- Two final-lineage LoRA exports in `/root/autodl-tmp/native_fc_20260923/`:
  `broken_step150` and `repaired_step150`, approximately 155 MiB each.
- Every exported tensor was read back and compared exactly to its source; ordered
  float32 SHA-256 matches the previous endpoint audit. 392 tensors per arm,
  80,740,352 parameters, bf16; rank/alpha 32, standard scaling 1, seven target modules.
- Real installed verl constructor inspected: rank/alpha and target selection from
  model config, bias none; default standard LoRA (no rsLoRA/DoRA).
- Real tokenizer plus trusted toy executor smoke passed, including tool_response
  rendering and independent terminal score. An initial toy fixture incorrectly
  imported `solution` in script mode; corrected to inline assertion and re-tested.
- Six controller/export-key unit tests passed locally and on the server.
- Tiny random CPU Qwen2+PEFT test: enabled adapter changes logits; save/load restores
  identical tensors and logits. This does not prove real 7B GPU loading.
- Export receipts are copied to `p3/results/native_fc_20260923/` locally.
- No training, real-model generation, GPU work or original-checkpoint deletion.

## Deployed isolated workspace

`/root/autodl-tmp/native_fc_20260923/workspace`

It contains the new evaluator/controller/tests, copies of the unchanged repaired
parser, formatting and sandbox code, and the frozen evaluation manifest. Nothing
in the original training directory was overwritten. Runtime:
`/root/code-venv/bin/python` (torch 2.13.0, transformers 5.10.4, peft 0.20.0).

## Scientific protocol

All three weights use the same original template plus tools, repaired extraction
and argument/name checks, tool-role observations, one-call dispatch cap, four
assistant generations, greedy decoding, 1024 tokens per generation, 8192 context,
and 1200-character observation cap. Raw assistant bytes are preserved in history;
the local native generation path does not silently rewrite them into Hermes calls.
The full conversation is re-rendered using the template each turn. This is a new
common held-out FC evaluator, not byte-identical replay of verl's AgentLoop.

The held-out prompt and 542 records come from the frozen shared-eval manifest.
Tool tests use EvalPlus script-mode checks rather than the training reward's pytest
partial-credit tests. Observation formatting matches training, but task/test content
does not. The comparison reduces interface/protocol confounding; it does not erase
all train/evaluation distribution shift. Do not call it an exact training-path eval.

Only an accepted call causes interactive execution. Non-call final code receives
one terminal test without feedback. That test is not counted as a tool interaction.
Tool success alone does not stop generation; the model can finish or use the cap.
A later explicit final code submission supersedes the previous tool code; plain
text such as “Done” leaves the last tool submission as final. Rescue is failure on
the first actual tool test followed by a passing tool test, not a no-test-to-pass
transition. Original shared-eval rescue counts must not be pooled with this metric.

## Next step: explicitly authorized GPU smoke only

Use a fresh output path for each invocation. Example base smoke, six tasks (first
three and last three of the frozen task list):

```bash
cd /root/autodl-tmp/native_fc_20260923/workspace
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/code-venv/bin/python p3/native_fc_eval.py \
  --base /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
  --arm base --manifest eval_manifest.json \
  --out /root/autodl-tmp/native_fc_20260923/smoke_base --limit 6 --allow-gpu
```

Repeat separately for broken/repaired with the matching adapter directory and new
output paths. The runner checks the expected arm digest before loading, exported
file hashes, and every in-memory LoRA tensor after loading; also checks scaling.
CUDA absence, context overflow, wrong identity or existing output directory fail
closed. No fallback to base weights, automatic training or checkpoint overwrite.

Inspect all three COMPLETE markers, raw trajectories, identities, tool/test/feedback
counts, length stops, memory and elapsed time. A model choosing no tool is a valid
behavioral result, not permission to silently strengthen its prompt. Context/length
changes must be common to all arms and fixed before the full evaluation.

## Budget caveat

This first runner intentionally uses sequential Transformers generation for a
simple, directly auditable adapter-loading path. It is not the old batched vLLM
evaluator. The earlier 1-3 GPU-hour allowance is **not validated** for this backend.
Estimate from the smoke and a small pilot before approving 3 x 542 tasks; if too
slow, implement and test batching before running the full dataset. Do not spend
unbounded GPU time to meet the old estimate. The 454 fixed-repair tasks are not
included in the initial scope.

Remaining: real-GPU adapter activation/throughput smoke, frozen common run manifest,
full paired evaluation and statistical analysis. No paper result has changed yet.

# Native-FC evaluation: offline acceptance and bounded plan

## Decision

**NOT READY TO LAUNCH.** The local evidence establishes which checkpoints to use,
not that the remote files still exist or that an evaluation server loads them.
Update: the 2026-09-23 CPU-only remote check now confirms both endpoint LoRA digests
and base/tokenizer config hashes; see REMOTE_WEIGHT_ACCEPTANCE_20260923.md. Runtime
loading, adapter export and the FC evaluator remain unverified.
Subsequent CPU preparation is complete: see NATIVE_FC_RUNBOOK_20260923.md. Both
adapters were exported and round-trip verified; the common FC controller and real
tokenizer/executor fixture pass CPU tests. Only GPU smoke/pilot is the next approved
candidate, not full-run acceptance. Original budget assumptions need pilot revision
because this first implementation uses sequential Transformers generation.
Do not start training, use `run_p3_pair.sh`, or rely on an old auto-relay monitor.
This is evaluation-only work on existing weights.

## What the offline inspection established

- Final lineage: broken/global_step_150 and
  repaired_resume_00090/global_step_150, not the abandoned repaired branch.
- Historical CPU audit (2026-09-19): each endpoint has 392 LoRA tensors and
  80,740,352 parameters. Expected ordered float32 tensor digests are in
  `checkpoint_lora_audit.json`. These are not whole-checkpoint SHA-256 values.
- No `.pt` or `.safetensors` model files are present in this local repository.
  Old server paths are not proof of current retention.
- The frozen evaluation manifest contains 542 unique tasks, with 454 fixed repair
  probes. Use its actual records, not a new download or silently filtered subset.
- Existing `NativeRolloutPolicy` means native **generation RPC**, not native FC
  evaluation. It does not pass tools to the template; `eval_decompose.py` executes
  extracted code automatically and returns user-message feedback. Reusing it as-is
  would repeat the protocol problem the reviewer raised.
- An OpenAI endpoint on the training server previously served base weights while
  LoRA was injected only through native rollout RPC. Changing the model label or
  seeing different answers is not proof of correct checkpoint loading.

## Proposed paired design (freeze before inference)

Three weights: common step-0 base, broken step 150, final-lineage repaired step 150.
Primary dataset: the same 542-task multi-turn channel, all retained. Keep the
454-task fixed-repair channel optional and separately budgeted; it is not needed
to answer the first autonomous-initiation question.

Use one common, pinned, working interface for all three weights: original model
tokenizer/template with the same run_tests schema and repaired verl parser behavior.
Do not swap in the BFCL dedicated few-shot template: that would add another prompt
intervention. The training registry parser and serving plugin are different code.
If serving via vLLM, implement/test a wrapper of the exact repaired extraction and
validation logic, or use the actual verl AgentLoop without optimizer updates.

At most four assistant generations, temperature 0, top_p 1, seed 0 and 1024 output
tokens per generation, initially matching the shared held-out evaluation budget.
Record length stops; freeze any common budget adjustment during smoke testing,
before endpoint comparisons. Pass tools into the chat template explicitly. Use
tool-role feedback, one-call dispatch cap, original observation formatting and
1200-character feedback cap. Pin all configuration/data/source hashes.

Only an accepted `run_tests` call with valid nonempty code causes interactive
execution and feedback. Plain final code does **not** trigger automatic feedback.
For task success, independently test the final submitted code once after the
dialogue, without sending those results back. Keep this terminal scoring separate
from tool-execution counts, and report missing submissions explicitly.

Record raw generations, extracted calls, allowed-name/argument checks, ignored
extra calls, dispatch ID, execution result, exact observation, final submission,
and truncation/error flags. Define rescue as a failed first actual tool test later
passing; separately report untested-first-turn cases, not as failures rescued.
Report initiation, accepted/executed/observed counts, >=2 tested steps and final
success. Use paired task comparisons and intervals, not an assertion that equal
counts prove no learning. Describe this as post-hoc evaluation, not preregistered
training evidence.

## Gates before spending GPU hours

1. On the existing storage host (CPU-only sufficient), rerun
   `python3 p3/native_fc_preflight.py --verify-weights --output /tmp/native_fc_check.json`.
   Return code 2 means overall not launch-ready, even if weight verification passed.
   Confirm both endpoint digests, base revision, disk and available memory.
2. Export LoRA using actual stored PEFT config and state keys. Do not guess target
   modules from rank/alpha alone. Round-trip every tensor and compare digest,
   dimensions and scaling. Never overwrite original checkpoints.
3. Implement and unit-test native FC conversations, terminal scoring and fail-closed
   identities. Forced fixture calls must return a tool-role observation; invalid
   names/malformed arguments must never dispatch. An ordinary model returning zero
   calls is a result, not a preflight failure by itself.
4. Run a short GPU smoke: attest the loaded adapter identity and check in-memory
   adapter tensors/config, zero request errors and complete raw logs. A serving
   model name or training-step metadata alone is insufficient.
5. Freeze manifest and measure pilot throughput; only then approve the full run.
   No checkpoint saving during inference and no automatic repaired-arm training.

## Budget, not a runtime guarantee

Historical elapsed times in `native_fc_preflight_report.json` are for a different,
programmatic-feedback evaluator and both channels. They constrain expectations but
cannot predict new FC response lengths. Provisionally reserve **1-3 A800-80GB GPU
hours** for three-weight loading, smoke and the 542-task comparison; this is an
engineering allowance, not measured performance. CPU exporter/runner implementation
is separate. Stop and re-estimate after the pilot rather than letting a broken run
consume the allowance. Additional fixed-repair evaluation is optional.

Existing full actor files are about 15.7 GB each. Inference need not write more
15 GB checkpoints: prefer compact LoRA adapters, sequential loading and bounded
logs. Actual disk headroom must be checked before export; delete nothing as part
of this audit. If both originals are gone, do not silently substitute another step
or recommend retraining without a new user decision.

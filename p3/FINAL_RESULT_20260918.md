# P3 final result: broken-FC versus repaired-FC RL

Date validated: 2026-09-18 (Asia/Shanghai)

## Design

- Model: Qwen2.5-Coder-7B-Instruct.
- Optimisation: GRPO, LoRA rank/alpha 32 over all linear modules, 150 steps, seed 0.
- Per step: 16 prompts x 8 rollouts.
- Both arms share model, data, prompt, decoding, reward/verifier, evaluation, seed, and training configuration.
- Sole intervention: verl `multi_turn.format=hermes` (broken) versus the registered `qwen2_5_coder` parser (repaired).
- Training rollouts were retained in full with SHA-256 digests. The verifier regex was anchored to pytest's summary and skipped/xfailed/xpassed outcomes were included in the denominator before either arm ran.

## Persisted-evidence validation

- Both arms exited with return code 0.
- Repaired step 90/120/150 evaluations attest the corresponding native checkpoint weights.
- Every held-out evaluation has 996 unique rows, zero request failures, and a matching content hash.
- Final repaired actor checkpoint size: 15,728,105,024 bytes.
- Parser, agent-loop, trajectory-identity, and code-tool hooks all fired; no dead hook or `P3_INVALID` marker was present.

The repaired job was interrupted after step 116 and resumed from the step-90 checkpoint. Before resumption, the model, optimizer, scheduler, RNG, dataloader state, source hashes, configuration, and pair-manifest hash were verified. The formal curve uses the original step-90 evaluation and the resumed, weight-attested step-120 and step-150 evaluations. Repeating step 90 changed final passes from 428 to 426 while leaving turn-1 passes at 415; this is retained as test-retest variation rather than silently selecting one value.

## Training-rollout mechanism

Counts below exclude held-out evaluation. The repaired total combines original steps 1-90 with resumed steps 91-150; discarded steps 91-116 from the interrupted attempt are excluded.

| Arm | Generation/extract events | Tight emissions | Parser-accepted calls | Executed calls | Returned observations |
|---|---:|---:|---:|---:|---:|
| broken FC | 19,203 | 8,689 | 10 | 10 | 10 |
| repaired FC | 33,367 | 17,608 | 16,912 | 16,844 | 16,844 |

The number of generation events is post-treatment: repaired trajectories continue after observations, so it is not a common denominator. The defensible contrast is that changing only the registered parser moves the training stack from a nearly closed channel to 16,844 executed and observed tool interactions.

## Held-out result (paired multi-turn split, n=542)

| Arm / checkpoint | Turn-1 pass | Final pass | Rescued after turn 1 |
|---|---:|---:|---:|
| repaired step 0 | 418 | 430 | 12 |
| repaired step 120 | 415 | 427 | 12 |
| repaired step 150 | 415 | 427 | 12 |
| broken step 150 | 418 | 430 | 12 |

- Repaired step 0 -> 150 final: one gain and four losses, exact paired McNemar p=0.375.
- Broken step 150 -> repaired step 150 final: zero gains and three losses, exact paired McNemar p=0.25.
- These non-significant differences are not evidence that repair is harmful.

## Admissible claim

Parser repair causally restores tool-mediated samples to the RL experience distribution in this setup. It does not produce a measurable held-out multi-turn learning benefit at 7B, 150 LoRA steps, and one seed. Access to tool-mediated trajectories and learning from those trajectories are therefore separate empirical questions.

Do not write that the broken arm executed zero calls, that repaired FC improves performance, or that a working channel is generally insufficient.

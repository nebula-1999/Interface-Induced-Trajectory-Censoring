# P3 formal-run handoff to Claude (2026-09-16, Asia/Shanghai)

## One-line status

The real-GPU acceptance suite passed, and the clean paired run
`p3-formal-20260916-a` is now running the **broken** arm on `autodl-code`.
The same detached driver will start the repaired arm only after broken exits 0.
Do not modify or redeploy P3 runtime files on the server while either arm runs.

## Live run

- Host: SSH alias `autodl-code`
- GPU: A800 80GB
- Run ID: `p3-formal-20260916-a`
- Combined live log:
  `/root/autodl-tmp/runs/p3-formal-20260916-a-pair-relaunch.log`
- Run root:
  `/root/autodl-tmp/runs/p3/p3-formal-20260916-a/`
- Current arm: `broken`
- Driver order: `broken` then `repaired`
- Formal configuration: 150 updates per arm, step-0 baseline, eval every 30,
  checkpoint every 30, retain two checkpoints per arm.
- Shared paired-manifest digest:
  `96fa0f0085ae2b32adaec9ca1ecf24c9f06abef01291e10df73800b4081bc106`
- Starting data-disk state: 350G total, 226G used, 125G free.
- Each actor checkpoint is 15,728,105,024 bytes. The keep-two policy and
  sequential arms fit, but continue checking disk before every save boundary.

Basic monitoring (read-only):

```bash
ssh autodl-code 'tail -n 120 /root/autodl-tmp/runs/p3-formal-20260916-a-pair-relaunch.log'
ssh autodl-code 'pgrep -af "p3-formal-20260916-a|launch_ppo"; df -h /root/autodl-tmp'
```

Completion markers in the combined log:

```text
[p3-pair] ===== completed broken =====
[p3-pair] ===== start repaired =====
[p3-pair] ===== completed repaired =====
```

Do not infer completion from a missing process alone. Check the arm's
`exit.json`, the combined log, `events/summary.json`, every expected
`eval/step_*.complete.json`, and checkpoint directories.

## What was accepted before launch

The formal launcher is fail-closed and re-verifies
`p3/GPU_ACCEPTANCE_20260916.json` against live immutable artifacts before each
arm. Acceptance-record SHA-256:
`1131d1b6fb9b4a855f6400f3f6e22a4ac58184b891d49f84f3fc30c5f359aa51`.

### Broken one-step smoke

- Root: `/root/autodl-tmp/runs/p3/acceptance-20260916-e/broken`
- Native current-LoRA eval attested weight steps 0 and 1, zero request errors.
- 174 tight **heuristic candidates**, 0 parser acceptances, 0 executions.
- Checkpoint written and complete (15.7GB).

Do not call the 174 candidates “valid calls” without the independent
JSON/schema audit. The code intentionally labels them heuristic candidates.

### Checkpoint-resume smoke

- Root: `/root/autodl-tmp/runs/p3/acceptance-20260916-g/broken`
- Source: smoke-e `global_step_1` (every source file size and SHA-256 recorded).
- verl loaded model, optimizer, RNG and scheduler, set global step to 1, then
  produced `global_step_2`.
- Native current-LoRA eval attested steps 1 and 2. This proves it did not
  silently restart from step 0.

### Repaired one-step smoke

- Root: `/root/autodl-tmp/runs/p3/acceptance-20260916-h/repaired`
- Only intervention relative to broken: verl parser registry name
  `hermes` -> `qwen2_5_coder`.
- Native current-LoRA eval attested steps 0 and 1, zero request errors.
- Parser accepted 369 calls; `_call_tool` and `CodeTool.execute` each ran 367.
- 161 distinct trajectory identities received non-empty observations; all 161
  performed a later assistant generation under the same
  `(step, sample_index, rollout_n, validate)` identity.
- The 369-vs-367 difference is fully localized to one emission containing
  three `run_tests` objects. verl has `max_parallel_calls: 1`, so it dispatched
  one. This is an explicit execution cap, not parser loss.

The repaired smoke emitted one non-fatal DataLoader-worker teardown traceback
after final validation. It was not an OOM: the host had 959GiB available RAM,
the kernel had no OOM record, driver rc=0, both native eval artifacts completed,
and the checkpoint completed. Preserve this disclosure.

## Important repair: evaluation now uses the actual training LoRA

The old hook called the OpenAI HTTP endpoint. In this verl/vLLM integration that
endpoint serves the registered base model and does **not** prove use of the LoRA
that verl injects into native rollout RPC calls.

The new `NativeRolloutPolicy` uses
`async_rollout_manager.llm_client.generate`. Each result must report
`min_global_steps == max_global_steps == expected_step`; otherwise evaluation
fails. Therefore step-N evaluation is now tied directly to the current LoRA.
Do not revert this to HTTP.

Full rollout text is retained with SHA-256. Tool events include step,
sample index, rollout index, validation flag, request ID/turn where available,
full observation, and observation hash. The verifier regex is anchored to the
pytest summary, and skipped/xfail/xpass are included in the denominator.

## Failed launch that must remain disclosed

The first formal attempt wrote only:
`/root/autodl-tmp/runs/p3-formal-20260916-a-pair.log`.
It failed before creating a run directory or touching the GPU because
`python -I` could not import `acceptance_gate.py`. We fixed this by explicitly
adding the launcher's directory to `sys.path`, reran 29 local tests and 29 remote
tests, then relaunched to the `*-pair-relaunch.log` above. Do not confuse the
first log with the live run.

## Failure and recovery rules

1. Never delete or reuse a run/arm directory. Every failed attempt is evidence.
2. Never relaunch while `launch_ppo.py` for this run is alive.
3. The current pair driver automatically starts repaired only if broken exits 0.
4. If the machine stops, checkpoints exist every 30 steps, but the checked-in
   launcher currently exposes verified resume only as `resume-smoke`; it does
   **not** yet authorize formal resume. Do not pretend a fresh restart is a
   continuation. Add and test a formal-resume mode that hashes the source
   checkpoint and attests the first resumed eval step before using it.
5. Do not edit/deploy any runtime file between arms. The shared paired manifest
   should reject drift, but avoid creating the situation.
6. Keep at least one checkpoint per reported evaluation step until the result is
   audited; the present keep-two policy retains only the latest two by design.

## Result interpretation after both arms finish

Use the programmatic-feedback artifacts, not `critic/score/mean`, to decide the
paper branch. For each arm and each of steps 0/30/60/90/120/150, validate the
`.complete.json` hash and compute unique-task:

- final pass rate;
- turn-1 pass rate;
- rescue count/rate = final pass and turn-1 fail;
- mean turns;
- parser acceptances, tool executions and observations.

The prior branch criterion based on aggregate reward is invalid. Also do not use
the historical duplicated 1108-row baseline or silently relabel it as 542.

- If repaired-FC shows materially increasing multi-turn rescues while broken
  does not: withdraw the old “working channel was not sufficient” claim.
- If repaired-FC opens the channel but rescues remain flat: the sufficiency
  result becomes a clean parser-only paired control, but still report the
  precise model scale, LoRA setup, one seed and training budget.

PEFT wording is mandatory: both arms use identical rank-32 LoRA because 7B
full-parameter RL does not fit the single 80GB card. The arm contrast remains
clean; it is not strictly comparable to historical full-parameter baselines.

## Code and tests

- Local and remote: 29 `unittest` tests pass.
- Real verifier tests previously passed, including spoofed pytest-summary and
  skipped/xfail denominator cases.
- Key files:
  - `p3/launch_arm.py`: fail-closed launch planning and acceptance gate
  - `p3/acceptance_gate.py`: live evidence re-verification
  - `p3/run_p3_pair.sh`: sequential broken -> repaired driver
  - `p3/GPU_ACCEPTANCE_20260916.json`: acceptance record
  - `chat_policy.py`: native current-LoRA policy
  - `code_eval_hook.py`: immutable per-step evaluation
  - `eval_artifacts.py`: manifests, hashes and exclusive step artifacts
  - `p3/rollout_probe.py`: emitted/accepted/executed/observation instrumentation
  - `test_p3_acceptance.py`: guardrail tests

## Worktree ownership

This commit intentionally excludes the user's unrelated/unfinalized changes:

- `paper/main.tex`
- `paper/main.pdf`
- `writeup/PROJECT_RETROSPECTIVE.md`
- `writeup/PROJECT_LEARNING_GUIDE.md`
- `writeup/learning_lab.py`

Preserve them. Do not reset, overwrite or bundle them into a P3 runtime fix.

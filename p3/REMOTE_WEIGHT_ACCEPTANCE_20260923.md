# CPU-only endpoint verification, 2026-09-23

Status: **weight-identity gate PASS; GPU evaluation not yet launch-ready**.

Read-only SSH checks on the original storage instance confirmed both final-lineage
step-150 actor files exist. Single-thread PyTorch 2.8.0 CPU loading used mmap and
weights_only=True; no inference, training, export, merge or deletion was performed.

| Arm | LoRA ordered-float32 SHA-256 | Tensors | Parameters |
| --- | --- | ---: | ---: |
| broken | 84251925cbf53418e9404869f58cf9f5e80a71d98aace7763273368da2f24217 | 392 | 80,740,352 |
| repaired_resume_00090 | 07a7cd9627798d7e94bba9afe8630b5c0c316b6353a5c34bbb151e9de088c779 | 392 | 80,740,352 |

Both digests equal the 2026-09-19 audit, exactly. Each actor file has
15,393,364,507 bytes. These digests identify LoRA tensor values, not the full
checkpoint contents or inference-loaded weights.

Observed target modules: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj,
v_proj. Stored lora_train_meta.json specifies rank 32, alpha 32, CAUSAL_LM.
Export must still verify key mapping/scaling and loaded runtime identity.

Base config SHA-256:
`c0242402ad6a13b331ea320feea8c7e3776ffb7a4eff0757b9cd667e116d9a28`

Tokenizer config SHA-256:
`959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4`

Both match the frozen training manifest. Base weight shards were not independently
hashed in this check; matching configuration is not a full base-weight audit.

Disk: 350 GB total, 314 GB used, 37 GB available (df rounded values).
Actual cgroup allowance: 2 GiB memory and 0.5 CPU. Host-level free output is not
the usable container budget. The initial default-thread read-only audit was stopped
and replaced with a single-thread audit; memory failure count remained zero at the
intermediate check. The replacement completed successfully for both arms.

Remaining gates: compact adapter export/round-trip, common native-FC runner and
unit tests, then explicit runtime loading/fixture smoke on GPU. Do not launch old
training drivers or treat this CPU-only verification as GPU acceptance.

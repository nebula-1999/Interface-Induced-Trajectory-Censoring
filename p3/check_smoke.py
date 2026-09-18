"""Check persisted smoke artifacts, including native weight-step attestation."""
import hashlib
import json
from pathlib import Path
import sys


def check(root):
    root = Path(root)
    launch = json.loads((root / "launch.json").read_text())
    config = dict(a.split("=", 1) for a in launch["argv"][2:])
    steps = int(config["trainer.total_training_steps"])
    freq = int(config["trainer.test_freq"])
    start_step = int(launch.get("start_step", 0))
    manifests = set()
    sample_keys = None
    expected_steps = [start_step, *[s for s in range(start_step + 1, steps + 1)
                                    if s % freq == 0 or s == steps]]
    for step in expected_steps:
        stem = root / "eval" / f"step_{step:05d}"
        done = json.loads(stem.with_suffix(".complete.json").read_text())
        raw = stem.with_suffix(".jsonl").read_bytes()
        if done["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"Evaluation hash mismatch at step {step}")
        if done["metrics"]["code/request_failures"] != 0:
            raise ValueError("Request failures")
        if (done["metrics"].get("code/native_weight_step") != step
                or done["metrics"].get("code/native_requests", 0) <= 0):
            raise ValueError(f"Native rollout did not attest weight step {step}")
        rows = [json.loads(line) for line in raw.splitlines()]
        keys = [(r["channel"], r["task_id"]) for r in rows]
        if len(keys) != len(set(keys)) or len(rows) != done["rows"]:
            raise ValueError("Duplicate rows or wrong count")
        if not rows or any(r["run_id"] != f"{launch['run_id']}/{launch['arm']}"
                           or r["step"] != step for r in rows):
            raise ValueError("Missing/mixed evaluation identity")
        if sample_keys is not None and set(keys) != sample_keys:
            raise ValueError("Evaluation sample changed across steps")
        sample_keys = set(keys)
        manifests.add(done["manifest_sha256"])
    if len(manifests) != 1:
        raise ValueError("Evaluation manifest changed across steps")
    ckpt = root / "ckpt" / f"global_step_{steps}" / "actor"
    files = [p for p in ckpt.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    if size == 0:
        raise ValueError("Final checkpoint actor files missing/empty")
    restored = launch.get("mode") in ("resume-smoke", "formal-resume") and start_step > 0
    return dict(smoke_artifacts_complete=True, checkpoint_actor_bytes=size,
                native_weight_steps_attested=expected_steps,
                formal_accepted=False, weight_sync_verified=True,
                checkpoint_load_verified=restored,
                note="Formal 150-step pair remains separately gated.")


if __name__ == "__main__":
    try:
        print(json.dumps(check(sys.argv[1]), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        print(f"[p3] Smoke artifacts INVALID: {exc}", file=sys.stderr)
        sys.exit(2)

"""Immutable evaluation identities and fail-closed, non-appending step artifacts."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(canonical(value) + "\n")
            f.flush()
            os.fsync(f.fileno())
            os.link(temp, path)  # atomic publish, refuses an existing target
        finally:
            temp.unlink(missing_ok=True)


def bind_manifest(path, value):
    try:
        exclusive_json(path, value)
    except FileExistsError:
        if json.loads(Path(path).read_text()) != value:
            raise ValueError(f"Evaluation manifest changed: {path}") from None
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unique_ids(ids, label):
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs in {label}")


class StepArtifact:
    def __init__(self, out, step, run_id, manifest_sha):
        self.out = Path(out)
        self.stem = f"step_{step:05d}"
        self.identity = dict(step=step, run_id=run_id, manifest_sha256=manifest_sha,
                             schema_version=2, eval_protocol="programmatic_feedback")
        if (self.out / f"{self.stem}.jsonl").exists():
            raise FileExistsError("Existing evaluation must not be appended to")
        exclusive_json(self.out / f"{self.stem}.started.json", self.identity)

    def finish(self, multi, repair, expected_multi, expected_repair, metrics):
        rows = []
        for channel, tasks, expected in (("multi", multi, expected_multi),
                                         ("repair", repair, expected_repair)):
            ids = [t.task_id for t in tasks]
            unique_ids(ids, channel)
            if set(ids) != set(expected):
                raise ValueError(f"Missing or unexpected tasks in {channel}")
            for task in tasks:
                if not task.turns:
                    raise ValueError(f"No turns for {task.task_id}")
                rows.append({**asdict(task), **self.identity, "channel": channel})
        if metrics.get("code/request_failures") != 0:
            raise ValueError("Request failures invalidate evaluation")
        partial = self.out / f"{self.stem}.partial.jsonl"
        with partial.open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(canonical(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        target = self.out / f"{self.stem}.jsonl"
        os.link(partial, target)
        partial.unlink()
        weight_sync_verified = (
            metrics.get("code/native_weight_step") == self.identity["step"]
            and metrics.get("code/native_requests", 0) > 0
        )
        exclusive_json(self.out / f"{self.stem}.complete.json",
                       {**self.identity, "rows": len(rows), "metrics": metrics,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "weight_sync_verified": weight_sync_verified})

    def fail(self, error):
        exclusive_json(self.out / f"{self.stem}.failed.json",
                       {**self.identity, "error": str(error)})

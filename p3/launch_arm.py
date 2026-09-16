"""Fail-closed P3 launch planning; standard library only, safe with DRY=1.

Fresh runs never reuse event/evaluation directories. Resume remains blocked until
the installed verl loader has been verified, rather than silently starting over.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def positive(env, key, default):
    value = int(env.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def plan(env):
    env = dict(env)
    arm = env.get("ARM")
    if arm not in ("broken", "repaired"):
        raise ValueError("ARM must be broken or repaired")
    run_id = env.get("RUN_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("RUN_ID is required (letters/digits/underscore/hyphen)")
    mode = env.get("P3_MODE", "smoke")
    if mode not in ("smoke", "resume-smoke", "formal"):
        raise ValueError("P3_MODE must be smoke, resume-smoke or formal")
    resume_from = env.get("RESUME_FROM")
    start_step = 0
    if mode == "resume-smoke":
        if not resume_from:
            raise ValueError("resume-smoke requires RESUME_FROM")
        source = Path(resume_from)
        match = re.fullmatch(r"global_step_(\d+)", source.name)
        if not source.is_absolute() or not match:
            raise ValueError("RESUME_FROM must be an absolute global_step_N directory")
        start_step = int(match.group(1))
        if not (source / "actor").is_dir() or not (source / "data.pt").is_file():
            raise ValueError("RESUME_FROM lacks actor/ or data.pt")
        if not any(p.is_file() and p.stat().st_size for p in (source / "actor").rglob("*")):
            raise ValueError("RESUME_FROM actor checkpoint is empty")
    elif resume_from:
        raise ValueError("RESUME_FROM is only valid with P3_MODE=resume-smoke")
    default_steps = start_step + 1 if mode in ("smoke", "resume-smoke") else 150
    steps = positive(env, "STEPS", default_steps)
    freq = positive(env, "EVAL_FREQ", 1 if mode in ("smoke", "resume-smoke") else 30)
    save = positive(env, "SAVE_FREQ", 1 if mode in ("smoke", "resume-smoke") else 30)
    keep = positive(env, "KEEP_CKPT", 2)
    if steps % freq or steps % save:
        raise ValueError("STEPS must be divisible by both EVAL_FREQ and SAVE_FREQ")
    if mode == "smoke" and steps > 2:
        raise ValueError("Smoke is capped at 2 steps; it cannot authorize formal training")
    if mode == "resume-smoke" and steps != start_step + 1:
        raise ValueError("resume-smoke must run exactly one additional update")
    acceptance = None
    if mode == "formal" and env.get("DRY") != "1":
        record = env.get("P3_ACCEPTANCE_RECORD")
        if not record:
            raise ValueError("Formal training requires P3_ACCEPTANCE_RECORD")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from acceptance_gate import verify
        acceptance = verify(record, env["PROJ_DIR"])
    root = Path(env.get("P3_RUN_BASE", "/root/autodl-tmp/runs/p3"))
    if not root.is_absolute():
        raise ValueError("P3_RUN_BASE must be absolute")
    root = root / run_id / arm
    if root.exists():
        raise FileExistsError(f"Refusing existing run directory: {root}")
    proj = Path(env["PROJ_DIR"]).resolve()
    for key in ("P3_OUT", "P3_CKPT_DIR", "CODE_EVAL_OUT"):
        if env.get(key):
            raise ValueError(f"{key} is managed by RUN_ID; do not override it")
    ray_tag = hashlib.sha256(f"{run_id}/{arm}".encode()).hexdigest()[:12]
    ray_root = Path(env.get("P3_RAY_BASE", "/root/autodl-tmp/rp3")) / ray_tag
    # Ray appends a long session/sockets suffix; AF_UNIX is capped at 107 bytes.
    if len(str(ray_root)) > 42:
        raise ValueError("P3_RAY_BASE/RUN_ID path is too long for Ray AF_UNIX sockets")
    env.update(P3_RUN_ID=run_id, P3_ARM=arm, P3_MODE=mode,
               P3_START_STEP=str(start_step), P3_RESUME_FROM=resume_from or "",
               P3_OUT=str(root / "events"), P3_CKPT_DIR=str(root / "ckpt"),
               CODE_EVAL_OUT=str(root / "eval"),
               CODE_EVAL_REQUIRE_REPAIR="1", CODE_EVAL_REQUIRE_NATIVE="1",
               CODE_EVAL_RUN_ID=f"{run_id}/{arm}",
               CODE_EVAL_LIMIT=env.get("CODE_EVAL_LIMIT", "4"
                                       if mode in ("smoke", "resume-smoke") else "0"),
               CODE_EVAL_PROBES=str(proj / "probes_repair.jsonl"),
               CODE_EVAL_MANIFEST=str(root.parent / "eval_manifest.json"),
               P3_PAIR_MANIFEST=str(root.parent / "training_manifest.json"),
               TOOL_CALL_COUNTER=str(root / "tool_call_count"),
               TOOL_CALLED_FLAG=str(root / "tool_called"),
               RAY_TMPDIR=str(ray_root),
               HF_HOME=env.get("HF_HOME", "/root/autodl-tmp/hf"),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    if acceptance:
        env["P3_ACCEPTANCE_SHA256"] = acceptance["acceptance_record_sha256"]
    args = ["bash", str(proj / "p3/run_p3_probe.sh"),
            "actor_rollout_ref.rollout.multi_turn.format=" +
            ("hermes" if arm == "broken" else "qwen2_5_coder"),
            f"trainer.total_training_steps={steps}", f"trainer.test_freq={freq}",
            f"trainer.save_freq={save}", f"trainer.max_actor_ckpt_to_keep={keep}",
            "trainer.val_before_train=True",
            f"trainer.experiment_name=p3-{run_id}-{arm}"]
    if mode == "resume-smoke":
        args.extend(["trainer.resume_mode=resume_path",
                     f"trainer.resume_from_path={resume_from}"])
    return root, env, args, acceptance


def main():
    root, env, args, acceptance = plan(os.environ)
    start_step = int(env["P3_START_STEP"])
    resume_from = env["P3_RESUME_FROM"] or None
    if env.get("DRY") == "1":
        print(json.dumps({"run_root": str(root), "command": args}, indent=2), flush=True)
        return subprocess.call(args, env=env)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()  # exclusive launch lock; retained on failure, never removed
    proj = Path(env["PROJ_DIR"])
    sources = ["p3/launch_arm.py", "p3/run_p3_probe.sh", "code_eval_hook.py",
               "code_patch.py", "sandbox.py", "qwen_tools_parser.py",
               "p3/rollout_probe.py", "p3/parser_diagnostics.py", "eval_artifacts.py",
               "chat_policy.py", "eval_decompose.py", "p3/record_launch.py", "code_tool.py",
               "p3/check_smoke.py", "p3/acceptance_gate.py"]
    resume_files = None
    if resume_from:
        resume_files = {}
        for path in sorted(Path(resume_from).rglob("*")):
            if path.is_file():
                resume_files[str(path.relative_to(resume_from))] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    manifest = {"run_id": env["P3_RUN_ID"], "arm": env["P3_ARM"],
                "mode": env["P3_MODE"], "argv": args, "ray_tmpdir": env["RAY_TMPDIR"],
                "start_step": start_step, "resume_from": resume_from,
                "resume_checkpoint_files": resume_files,
                "gpu_acceptance": acceptance,
                "source_sha256": {p: hashlib.sha256((proj / p).read_bytes()).hexdigest()
                                  for p in sources}}
    (root / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rc = subprocess.call(args, env=env)
    if rc == 0:
        rc = subprocess.call([env.get("P3_PYTHON", "/root/code-venv/bin/python"), "-I",
                              str(proj / "p3/check_smoke.py"), str(root)], env=env)
    (root / "exit.json").write_text(json.dumps({"returncode": rc,
        "formal_accepted": False, "note": "Smoke completion is not GPU acceptance."}) + "\n")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError) as exc:
        print(f"[p3] REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)

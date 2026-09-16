"""Pin effective Hydra settings, input files and code before launching a P3 arm."""
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import sys

# Executed with -I: explicitly import only the local artifact utility.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_artifacts import bind_manifest


def effective_config(args):
    result = {}
    for arg in args:
        key, sep, value = arg.partition("=")
        if not sep:
            raise ValueError(f"Not a Hydra assignment: {arg}")
        result[key.lstrip("+")] = value
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    config = effective_config(args)
    if not os.environ.get("P3_PAIR_MANIFEST"):
        return  # standalone legacy probe, not the gated arm launcher
    for key in ("trainer.save_freq", "trainer.test_freq"):
        if int(config[key]) <= 0:
            raise ValueError(f"P3 arm requires positive {key}")
    if config["trainer.val_before_train"].lower() != "true":
        raise ValueError("P3 arm requires a step-0 baseline")
    proj = Path(os.environ["PROJ_DIR"])
    required = ["code_tool.py", "code_tool_core.py", "code_tool_config.yaml",
                "sandbox.py", "reward_code.py", "qwen_tools_parser.py", "code_patch.py",
                "code_eval_hook.py", "eval_decompose.py", "chat_policy.py", "eval_artifacts.py",
                "p3/rollout_probe.py", "p3/parser_diagnostics.py", "sitecustomize.py",
                "p3/sitecustomize.py", "p3/apply_verl_vllm_compat.py"]
    inputs = {name: sha(proj / name) for name in required}
    for key in ("data.train_files", "data.val_files"):
        inputs[key] = sha(config[key])
    inputs["probes_repair"] = sha(os.environ["CODE_EVAL_PROBES"])
    model = Path(config["actor_rollout_ref.model.path"])
    for name in ("config.json", "tokenizer_config.json"):
        inputs[f"model/{name}"] = sha(model / name)
    # These are the intervention and two bookkeeping fields, not confounders.
    for key in ("actor_rollout_ref.rollout.multi_turn.format",
                "trainer.default_local_dir", "trainer.experiment_name"):
        config.pop(key)
    manifest = dict(config=config, input_sha256=inputs,
                    packages={p: version(p) for p in ("verl", "vllm", "torch", "pytest")})
    digest = bind_manifest(os.environ["P3_PAIR_MANIFEST"], manifest)
    print(f"[p3] Paired training manifest: {digest}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

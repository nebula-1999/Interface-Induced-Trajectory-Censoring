#!/usr/bin/env python3
"""Backport verl's vLLM>=0.25 unstacked-LoRA mapper fix.

verl 0.9.0 passes Qwen's full HF->vLLM weight mapper to the in-memory LoRA
loader.  On vLLM >= 0.25 that mapper fuses q/k/v names into qkv_proj before
the LoRA manager can pack them, so only one bare tensor reaches a three-slice
layer and ``set_lora`` crashes.  verl main fixes this by using
``get_unstacked_mapper()`` for LoRA tensors.

The patch is intentionally narrow, idempotent, version-gated, and refuses an
unknown source layout.  It also preserves the exact original beside the
installed file, named by SHA256, so the environment change is auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import shutil
from pathlib import Path

from packaging.version import Version


OLD = """        if hasattr(model, \"hf_to_vllm_mapper\") and model.hf_to_vllm_mapper is not None:
            hf_to_vllm_mapper = model.hf_to_vllm_mapper
"""
NEW = OLD + """            if is_version_ge(minver=\"0.25.0\"):
                hf_to_vllm_mapper = hf_to_vllm_mapper.get_unstacked_mapper()
"""


def patch_text(text: str) -> tuple[str, str]:
    if NEW in text:
        return text, "already_patched"
    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(f"refusing unknown verl source layout: old block count={count}")
    return text.replace(OLD, NEW, 1), "patched"


def installed_utils_path() -> Path:
    spec = importlib.util.find_spec("verl.utils.vllm.utils")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate verl.utils.vllm.utils")
    return Path(spec.origin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, help="override installed utils.py (tests only)")
    args = ap.parse_args()

    verl_version = importlib.metadata.version("verl")
    vllm_version = importlib.metadata.version("vllm")
    if args.path is None:
        if Version(verl_version) != Version("0.9.0"):
            raise RuntimeError(f"patch is pinned to verl 0.9.0, got {verl_version}")
        if Version(vllm_version) < Version("0.25.0"):
            raise RuntimeError(f"unstacked mapper API requires vLLM>=0.25, got {vllm_version}")

    path = args.path or installed_utils_path()
    before = path.read_text(encoding="utf-8")
    after, status = patch_text(before)
    before_sha = hashlib.sha256(before.encode()).hexdigest()
    after_sha = hashlib.sha256(after.encode()).hexdigest()

    if status == "patched":
        backup = path.with_name(f"{path.name}.p3-original-{before_sha[:12]}")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(after, encoding="utf-8")
    else:
        backup = None

    print(
        f"[p3-compat] status={status} verl={verl_version} vllm={vllm_version} "
        f"path={path} before_sha256={before_sha} after_sha256={after_sha} "
        f"backup={backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

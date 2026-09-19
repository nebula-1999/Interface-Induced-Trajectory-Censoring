#!/usr/bin/env python3
"""Compare LoRA tensors between two verl FSDP checkpoints.

This script must run where the full ``model_world_size_1_rank_0.pt`` files are
available.  ``torch.load(..., mmap=True)`` avoids materialising the frozen base
model twice; only LoRA tensors are converted to float32 for the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def audit(start: Path, end: Path) -> dict:
    before = torch.load(start, map_location="cpu", mmap=True, weights_only=False)
    after = torch.load(end, map_location="cpu", mmap=True, weights_only=False)
    keys = sorted(key for key in before if ".lora_" in key)
    assert keys and keys == sorted(key for key in after if ".lora_" in key)

    count = changed = nonzero_start = nonzero_end = 0
    square_start = square_end = square_delta = 0.0
    max_delta = 0.0
    hash_start = hashlib.sha256()
    hash_end = hashlib.sha256()
    for key in keys:
        left = before[key].detach().float().contiguous()
        right = after[key].detach().float().contiguous()
        assert left.shape == right.shape
        delta = right - left
        count += left.numel()
        changed += int(torch.count_nonzero(delta).item())
        nonzero_start += int(torch.count_nonzero(left).item())
        nonzero_end += int(torch.count_nonzero(right).item())
        square_start += float(torch.sum(left * left).item())
        square_end += float(torch.sum(right * right).item())
        square_delta += float(torch.sum(delta * delta).item())
        max_delta = max(max_delta, float(torch.max(torch.abs(delta)).item()))
        hash_start.update(left.numpy().tobytes())
        hash_end.update(right.numpy().tobytes())

    return {
        "start_path": str(start),
        "end_path": str(end),
        "lora_tensor_count": len(keys),
        "lora_parameter_count": count,
        "nonzero_start": nonzero_start,
        "nonzero_end": nonzero_end,
        "changed_parameters": changed,
        "changed_fraction": changed / count,
        "l2_start": math.sqrt(square_start),
        "l2_end": math.sqrt(square_end),
        "l2_delta": math.sqrt(square_delta),
        "max_abs_delta": max_delta,
        "sha256_start": hash_start.hexdigest(),
        "sha256_end": hash_end.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pair",
        nargs="+",
        help="LABEL=START_CHECKPOINT:END_CHECKPOINT",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "pairs": {}}
    for specification in args.pair:
        label, paths = specification.split("=", 1)
        start, end = paths.split(":", 1)
        report["pairs"][label] = audit(Path(start), Path(end))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

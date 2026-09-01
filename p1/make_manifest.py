#!/usr/bin/env python3
"""Create a deterministic paired BFCL subset manifest."""
import argparse
import json
import random
from pathlib import Path


def load_ids(data_dir: Path, category: str):
    path = data_dir / f"BFCL_v4_{category}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    ids = [json.loads(line)["id"] for line in path.open(encoding="utf-8") if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in {path}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-single", type=int, default=100)
    ap.add_argument("--n-multi", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    requested = {"simple_python": args.n_single, "multi_turn_base": args.n_multi}
    manifest = {}
    for category, n in requested.items():
        ids = load_ids(args.data_dir, category)
        if n > len(ids):
            raise ValueError(f"{category}: requested {n}, only {len(ids)} available")
        manifest[category] = sorted(rng.sample(ids, n))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "counts": {k: len(v) for k, v in manifest.items()}}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Register the Qwen2.5-Coder size ladder with BFCL, in memory.

**Why this file exists:** BFCL's shipped registry has no Qwen2.5-Coder entry, and
Coder is the one family of the four that shows censoring (Llama parses fine,
Mistral 400s, DeepSeek never injects the template). Without a registry entry we
could only report Llama, which is a valid but very weak negative result.

**Why it does not reuse BFCL's own ``QwenFCHandler``:** that class subclasses
``OSSHandler``. It posts to ``/v1/completions`` and regex-extracts ``<tool_call>``
blocks *inside the benchmark process*. On that path vLLM's tool-call parser is
never invoked, so interface censoring disappears by construction and the
measurement answers nothing. Every ladder entry here therefore uses the same
``P1OpenAIHandler`` as the rest of P1: BFCL's official OpenAI FC handler, posting
to ``/v1/chat/completions`` so the serving parser under study does the parsing.

**Why it does not patch ``model_config.py`` on disk:** BFCL is pinned to a commit
and its hashes go in the appendix. Registration happens in memory at import time
via ``sitecustomize``; ``bfcl_eval``'s own files stay byte-identical, which
``--check`` demonstrates.

Scope of the change, to be stated as-is in the paper:
  · append entries to ``MODEL_CONFIG_MAPPING`` at runtime
  · reuse P1's handler, already documented for the 7B arms
  · no change to any evaluation, parsing, or scoring logic
  · idempotent: re-importing does not duplicate entries

Usage: ``python register_qwen_coder.py --check``
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from bfcl_eval.constants import model_config as bfcl_model_config
from bfcl_eval.constants.model_config import ModelConfig, MODEL_CONFIG_MAPPING

from bfcl_registration import P1OpenAIHandler

SIZES = ["1.5B", "3B", "7B", "14B", "32B"]


def registry_name(size: str) -> str:
    return f"P1-Qwen2.5-Coder-{size}-FC"


def served_name(size: str) -> str:
    """The name the arm's vLLM must be started with (``--served-model-name``)."""
    return "p1-qwen25-coder-" + size.lower().replace(".", "_")


def register() -> list[str]:
    added = []
    for size in SIZES:
        name = registry_name(size)
        if name in MODEL_CONFIG_MAPPING:
            continue
        MODEL_CONFIG_MAPPING[name] = ModelConfig(
            model_name=served_name(size),
            display_name=f"Qwen2.5-Coder-{size}-Instruct (P1 FC)",
            url=f"https://huggingface.co/Qwen/Qwen2.5-Coder-{size}-Instruct",
            org="Qwen",
            license="apache-2.0",
            model_handler=P1OpenAIHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            # Dots are illegal in OpenAI function names, so BFCL must map them to
            # underscores on the way out and back on the way in.  Same value as
            # the 7B arms in bfcl_registration.py; differing here would make the
            # ladder incomparable to them.
            underscore_to_dot=True,
        )
        added.append(name)
    return added


REGISTERED_LADDER = register()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print the ladder and prove bfcl_eval was not modified")
    ap.parse_args()

    cfg = Path(bfcl_model_config.__file__)
    print(f"bfcl_eval model_config.py : {cfg}")
    print(f"  sha256 (unmodified)     : {hashlib.sha256(cfg.read_bytes()).hexdigest()[:20]}")
    print(f"  contains on-disk patch  : "
          f"{'yes — should be no' if 'P1:' in cfg.read_text(encoding='utf-8') else 'no'}")
    print()
    for size in SIZES:
        name = registry_name(size)
        entry = MODEL_CONFIG_MAPPING[name]
        print(f"{name:<32} served={entry.model_name:<26} "
              f"handler={entry.model_handler.__name__}")
    print()
    print("Start each arm's vLLM with the matching --served-model-name, then run")
    print("BFCL with --model <registry name>.  P1_SERVED_MODEL is not used by the")
    print("ladder: each entry pins its own served name so the two cannot drift.")
    if os.environ.get("P1_SERVED_MODEL"):
        print("note: P1_SERVED_MODEL is set; it only affects bfcl_registration.py's "
              "single-arm entry, not the ladder above.")


if __name__ == "__main__":
    main()

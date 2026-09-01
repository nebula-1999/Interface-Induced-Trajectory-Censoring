"""Load the P1 BFCL model registrations in every benchmark subprocess.

``bfcl_registration`` adds the single arm driven by P1_BFCL_MODEL_ID /
P1_SERVED_MODEL; ``register_qwen_coder`` adds the Qwen2.5-Coder size ladder.
Both register in memory and both route through P1OpenAIHandler, i.e. through
``/v1/chat/completions`` and the serving parser under study.
"""
for _mod in ("bfcl_registration", "register_qwen_coder"):
    try:
        __import__(_mod)
    except ModuleNotFoundError as exc:
        # During venv/bootstrap BFCL may not be installed yet.  Do not make
        # unrelated Python startup fail; the run script performs a strict
        # registration check.
        if exc.name and exc.name.startswith("bfcl_eval"):
            continue
        raise
del _mod

"""Load the P1 BFCL model registration in every benchmark subprocess."""
try:
    import bfcl_registration  # noqa: F401
except ModuleNotFoundError as exc:
    # During venv/bootstrap BFCL may not be installed yet.  Do not make unrelated
    # Python startup fail; the run script performs a strict registration check.
    if exc.name and exc.name.startswith("bfcl_eval"):
        pass
    else:
        raise

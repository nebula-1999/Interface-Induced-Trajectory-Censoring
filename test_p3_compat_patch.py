import pytest

from p3.apply_verl_vllm_compat import NEW, OLD, patch_text


def test_backport_is_exact_and_idempotent():
    source = "before\n" + OLD + "after\n"
    patched, status = patch_text(source)
    assert status == "patched"
    assert NEW in patched
    assert OLD in patched
    assert patch_text(patched) == (patched, "already_patched")


def test_backport_refuses_unknown_source():
    with pytest.raises(RuntimeError, match="old block count=0"):
        patch_text("different implementation")


def test_backport_refuses_ambiguous_source():
    with pytest.raises(RuntimeError, match="old block count=2"):
        patch_text(OLD + OLD)

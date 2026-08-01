from __future__ import annotations

import pytest
from scripts.cuda_smoke_test import compatible_compiled_arch, validate_compiled_arch


def test_pascal_sm60_cubin_is_compatible_with_tesla_p4_sm61() -> None:
    compiled = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]

    assert compatible_compiled_arch("sm_61", compiled) == "sm_60"


def test_incompatible_newer_major_does_not_satisfy_pascal() -> None:
    assert compatible_compiled_arch("sm_61", ["sm_70", "sm_75", "sm_80"]) is None


def test_exact_build_requirement_fails_when_arch_is_absent() -> None:
    with pytest.raises(
        RuntimeError, match="Required CUDA architecture sm_61 is absent"
    ):
        validate_compiled_arch("sm_61", ["sm_60", "sm_70"])


def test_exact_build_requirement_passes_when_arch_is_present() -> None:
    validate_compiled_arch("sm_60", ["sm_60", "sm_70"])

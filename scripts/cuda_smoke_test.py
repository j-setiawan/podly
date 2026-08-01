"""Validate a PyTorch CUDA build and optionally exercise an installed GPU."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from typing import Any

CUDA_ARCH_PATTERN = re.compile(r"^sm_(\d+)$")


def _arch_number(arch: str) -> int:
    match = CUDA_ARCH_PATTERN.fullmatch(arch)
    if match is None:
        raise ValueError(f"Invalid CUDA architecture {arch!r}; expected e.g. 'sm_61'")
    return int(match.group(1))


def compatible_compiled_arch(
    required_arch: str, compiled_arches: Sequence[str]
) -> str | None:
    """Return the best cubin target compatible with a requested device target.

    CUDA cubins are forward-compatible within one compute-capability major version.
    For example, an sm_60 cubin can execute on an sm_61 desktop GPU.
    """
    required = _arch_number(required_arch)
    compatible: list[tuple[int, str]] = []
    for arch in compiled_arches:
        try:
            candidate = _arch_number(arch)
        except ValueError:
            continue
        if candidate // 10 == required // 10 and candidate <= required:
            compatible.append((candidate, arch))
    return max(compatible, default=(0, ""))[1] or None


def validate_compiled_arch(required_arch: str, compiled_arches: Sequence[str]) -> None:
    """Require an exact architecture in ``torch.cuda.get_arch_list()``."""
    _arch_number(required_arch)
    if required_arch not in compiled_arches:
        compatible = compatible_compiled_arch(required_arch, compiled_arches)
        compatibility_note = (
            f" A compatible lower-minor target exists ({compatible}), but this build-time "
            "check intentionally requires an exact entry."
            if compatible
            else ""
        )
        raise RuntimeError(
            f"Required CUDA architecture {required_arch} is absent from the compiled "
            f"architecture list: {list(compiled_arches)}.{compatibility_note}"
        )


def get_compiled_arches(torch_module: Any) -> list[str]:
    """Return compiled CUDA targets, including on hosts without a CUDA device.

    ``torch.cuda.get_arch_list()`` returns an empty list when CUDA is unavailable,
    which is always the case in an ordinary Docker build. The underlying build
    metadata does not require a device and is therefore safe to use as a fallback.
    """
    compiled_arches = list(torch_module.cuda.get_arch_list())
    if compiled_arches:
        return compiled_arches

    torch_c = getattr(torch_module, "_C", None)
    get_arch_flags = getattr(torch_c, "_cuda_getArchFlags", None)
    if get_arch_flags is None:
        return []
    return get_arch_flags().split()


def print_torch_build(torch_module: Any) -> list[str]:
    """Print the CUDA build facts requested for image verification."""
    compiled_arches = get_compiled_arches(torch_module)
    print(f"torch.__version__: {torch_module.__version__}")
    print(f"torch.version.cuda: {torch_module.version.cuda}")
    print(f"torch.cuda.get_arch_list(): {compiled_arches}")
    return compiled_arches


def runtime_smoke_test(torch_module: Any, compiled_arches: Sequence[str]) -> None:
    """Run a real CUDA kernel after checking the detected device architecture."""
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; run the container with NVIDIA GPU access"
        )

    gpu_name = torch_module.cuda.get_device_name(0)
    major, minor = torch_module.cuda.get_device_capability(0)
    device_arch = f"sm_{major}{minor}"
    compatible_arch = compatible_compiled_arch(device_arch, compiled_arches)

    print(f"GPU name: {gpu_name}")
    print(f"GPU compute capability: {major}.{minor} ({device_arch})")
    if compatible_arch is None:
        raise RuntimeError(
            f"No compiled CUDA target in {list(compiled_arches)} can execute on {device_arch}"
        )
    print(f"Compatible compiled architecture: {compatible_arch}")

    values = torch_module.tensor([1.0, 2.0, 3.0], device="cuda")
    result = values * 2 + 1
    torch_module.cuda.synchronize()
    expected = torch_module.tensor([3.0, 5.0, 7.0], device="cuda")
    if not torch_module.equal(result, expected):
        raise RuntimeError(f"Unexpected CUDA arithmetic result: {result}")
    print(f"CUDA arithmetic result: {result.cpu().tolist()}")
    print("CUDA smoke test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="print and validate build metadata without requiring a GPU",
    )
    parser.add_argument(
        "--require-arch",
        default=os.environ.get("REQUIRE_CUDA_ARCH", ""),
        help="require an exact entry in torch.cuda.get_arch_list()",
    )
    args = parser.parse_args()

    import torch

    compiled_arches = print_torch_build(torch)
    if args.require_arch:
        validate_compiled_arch(args.require_arch, compiled_arches)
        print(f"Required CUDA architecture: {args.require_arch} (present)")
    if not args.build_only:
        runtime_smoke_test(torch, compiled_arches)


if __name__ == "__main__":
    main()

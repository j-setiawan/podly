# NVIDIA Pascal / Tesla P4 image

The published `ghcr.io/j-setiawan/podly` image is a linux/amd64 NVIDIA build pinned
to PyTorch 2.5.1 and the official CUDA 12.4 (`cu124`) wheel channel. Its Docker
build fails unless the wheel contains the `sm_60` cubins needed by Pascal GPUs.

The [official PyTorch 2.5.1/cu124 wheel][pytorch-wheel] reports this compiled
architecture list:

```text
sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
```

That retains native targets from Pascal through Hopper (including Volta,
Turing, Ampere, and Ada). It does not retain the `sm_100`/`sm_120` Blackwell
targets from current Torch wheels; supporting those alongside this P4-focused
pin would require a separately tagged image or a custom Torch build.

A Tesla P4 has compute capability 6.1 (`sm_61`). It runs the wheel's `sm_60`
cubins because [CUDA guarantees cubin compatibility][cuda-compatibility] from a
lower minor compute capability to a higher minor capability within the same
major version. The wheel therefore supports the P4 even though
`torch.cuda.get_arch_list()` does not contain a literal `sm_61` entry. The
wheel-resolved cuDNN 9.1.0 also [officially supports CUDA 12.4 and compute
capability 6.1][cudnn-support].

## Runtime smoke test

Run the image with NVIDIA Container Toolkit GPU access, overriding the normal
entrypoint with the included smoke-test script:

```bash
docker run --rm --gpus all --entrypoint python \
  ghcr.io/j-setiawan/podly:latest scripts/cuda_smoke_test.py
```

`run_podly_docker.sh` selects this image automatically for a production NVIDIA
run. Set `PODLY_NVIDIA_IMAGE` to override its registry path. CPU, ROCm, and lite
production runs keep their existing upstream image names and tags.

The script prints the Torch version, CUDA runtime version, compiled architecture
list, GPU name and compute capability. It then selects a compatible compiled
architecture, creates a CUDA tensor, performs arithmetic, synchronizes the GPU,
and verifies the result. This catches missing kernels that a standalone
`torch.cuda.is_available()` check does not.

The build arguments may be overridden for controlled testing:

```text
PYTORCH_VERSION=2.5.1
PYTORCH_CUDA_TAG=cu124
REQUIRE_CUDA_ARCH=sm_60
```

`REQUIRE_CUDA_ARCH` is an exact check against `torch.cuda.get_arch_list()`. For
that reason, setting it to `sm_61` intentionally fails with the official wheel;
use `sm_60` to validate the compatible Pascal cubin included by this build.

## Dependency audit and lockfile scope

`openai-whisper` is the only local-Whisper package in `pyproject.toml`. Its
runtime metadata requires unversioned `torch` and, on x86-64 Linux,
`triton>=2`; it does not require `torchaudio` or `torchvision`. No other Podly
dependency constrains the Torch version. The official Torch 2.5.1 wheel pins
its matching CUDA libraries, cuDNN 9.1.0.70, Triton 3.1.0, and SymPy 1.13.1.

The general-purpose `uv.lock` currently resolves PyPI Torch independently of
the image variant. The NVIDIA Docker branch therefore replaces that Torch
installation after `uv sync --frozen`, then runs `uv pip check`. This is the
smallest change and leaves CPU, ROCm, and lite dependency resolution unchanged.
It can leave packages used only by the newer locked Torch build installed but
unused in the NVIDIA image; they are not dependency conflicts. A future,
separate cleanup could define platform-specific uv dependency groups and lock
the PyTorch wheel channel, eliminating the replacement and those unused files.

[pytorch-wheel]: https://download.pytorch.org/whl/cu124/torch/
[cuda-compatibility]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/01-introduction/cuda-platform.html#binary-compatibility
[cudnn-support]: https://docs.nvidia.com/deeplearning/cudnn/backend/v9.1.0/reference/support-matrix.html

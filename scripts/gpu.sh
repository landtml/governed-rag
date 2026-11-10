#!/usr/bin/env bash
# Run a command in the ROCm (AMD GPU) environment: scripts/gpu.sh rag eval ...
# The embedder and reranker pick the GPU up on their own once torch can see it.
#
# One-time setup (RX 6700/6750 XT, ROCm 10.1; other cards need another device extra):
#   uv venv --python 3.12 .venv-gpu
#   uv pip install --python .venv-gpu/bin/python \
#     --index https://stable.repo.amd.com/rocm/whl-next/ --index-strategy unsafe-best-match \
#     "torch[device-gfx1030]==2.14.0+rocm10.1.0" "sentence-transformers>=3" pytest -e .
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$root/.venv-gpu/bin/python" ] || { echo "no .venv-gpu: see the setup lines in $0" >&2; exit 1; }

# The RX 6700 XT is gfx1031, which ROCm does not ship kernels for: run it as gfx1030.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
# hipBLASLt has no gfx1030 kernels in this build: send matrix products through rocBLAS.
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PATH="$root/.venv-gpu/bin:$PATH"
exec "$@"

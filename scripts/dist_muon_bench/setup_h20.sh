#!/usr/bin/env bash
# Python environment for the DistMuon session on one multi-GPU host.
#   bash scripts/dist_muon_bench/setup_h20.sh [WORK_DIR]
# Creates WORK_DIR/venv (default ~/distmuon_h20/venv) with the PyTorch nightly
# TorchTitan main requires, TorchTitan's requirements, torchao nightly and
# TensorBoard. Needs internet access. Run from the optimized checkout.
set -euo pipefail
WORK=${1:-$HOME/distmuon_h20}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$WORK"
# CUDA 13 wheels need driver >= 580; older drivers fall back to CUDA 12.8 wheels.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
CUDA_INDEX=${CUDA_INDEX:-$([ "${DRIVER:-0}" -ge 580 ] && echo cu130 || echo cu128)}
echo "driver $DRIVER -> wheels $CUDA_INDEX"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$WORK/venv"
source "$WORK/venv/bin/activate"
uv pip install --pre torch torchvision --index-url "https://download.pytorch.org/whl/nightly/$CUDA_INDEX"
uv pip install -r "$SRC/requirements.txt" -r "$SRC/requirements-dev.txt" -r "$SRC/.ci/docker/requirements-vlm.txt"
USE_CPP=0 uv pip install --pre --upgrade torchao --index-url "https://download.pytorch.org/whl/nightly/$CUDA_INDEX"
uv pip install tensorboard
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), "nccl", torch.cuda.nccl.version())
PY
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "SETUP_DONE venv=$WORK/venv"

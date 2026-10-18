#!/usr/bin/env bash
# Create a dedicated experiment environment; never change system packages/drivers.
set -euo pipefail
H20_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H20_REPO_ROOT=$(git -C "$H20_SCRIPT_DIR" rev-parse --show-toplevel)
H20_VENV_DIR=${1:-"$H20_REPO_ROOT/outputs/k3-h20-venv"}
if [ -e "$H20_VENV_DIR" ]; then
  echo "Refusing to overwrite existing environment: $H20_VENV_DIR" >&2
  echo "Use its Python directly, or choose a new environment directory." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  H20_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
  if [ "${H20_DRIVER%%.*}" -lt 580 ]; then
    echo "The recommended CUDA 13 environment expects R580+ or administrator-managed CUDA compatibility." >&2
    echo "This script does not install drivers. An existing compatible environment can run run.py directly." >&2
    exit 1
  fi
fi
if command -v uv >/dev/null 2>&1; then
  H20_UV=$(command -v uv)
  "$H20_UV" venv --python 3.12 "$H20_VENV_DIR"
else
  python3.12 -m venv "$H20_VENV_DIR"
  "$H20_VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple uv
  H20_UV="$H20_VENV_DIR/bin/uv"
fi
H20_PYTHON="$H20_VENV_DIR/bin/python"
"$H20_UV" pip install --python "$H20_PYTHON" \
  'torch==2.16.0.dev20261016+cu130' 'torchvision==0.30.0.dev20261016+cu130' \
  --index-url https://download.pytorch.org/whl/nightly/cu130
"$H20_UV" pip install --python "$H20_PYTHON" -r "$H20_REPO_ROOT/requirements.txt"
USE_CPP=0 "$H20_UV" pip install --python "$H20_PYTHON" \
  'torchao==0.19.0.dev20261017+cu130' --index-url https://download.pytorch.org/whl/nightly/cu130
"$H20_UV" pip install --python "$H20_PYTHON" 'flash-attn-3==3.0.0' \
  --index-url https://download.pytorch.org/whl/nightly/cu130
echo "Environment ready: $H20_PYTHON"
echo "No GPU training was started. Next:"
echo "cd $H20_REPO_ROOT"
echo "$H20_PYTHON -m scripts.distmuon_k3_h20.run --output $H20_REPO_ROOT/outputs/k3-h20/manual-01"

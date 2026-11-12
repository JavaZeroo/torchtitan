#!/bin/bash
# Installs everything exp_h20/run.sh needs, inside a fresh
# nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 container. Run from the repo root:
#   bash exp_h20/setup.sh
# Then: source /opt/k31/bin/activate
#
# CU picks the PyTorch nightly index. cu132 is what torchtitan CI uses and
# needs an NVIDIA driver for CUDA 13.2 (595 or newer). With an older 13.x
# driver (580+), use CU=cu130: its newest nightly is from 2026-11-02.
set -eux
cd "$(dirname "$0")/.."
CU=${CU:-cu132}
VENV=${VENV:-/opt/k31}
AG_SRC=${AG_SRC:-$PWD/../attention-gym}

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git python3 python3-venv python3-dev build-essential ca-certificates curl jq

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip

INDEX=https://download.pytorch.org/whl/nightly/$CU
python -m pip install --pre torch --index-url "$INDEX"
python -m pip install --pre -r .github/requirements-torchao.txt --index-url "$INDEX"
# requirements.txt pins attn-gym and torch_remat from GitHub.
python -m pip install -r requirements.txt -r .ci/docker/requirements-dev.txt
# FA3, as the H100 CI job installs it; only some Kimi K3 paths use it.
python -m pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/test/cu130 \
  || echo "flash-attn-3 install failed; continuing"

# Attention Gym source for the K3 bisect stage.
if [ ! -d "$AG_SRC/.git" ]; then
  git clone https://github.com/meta-pytorch/attention-gym.git "$AG_SRC"
fi
git -C "$AG_SRC" fetch -q origin

python - <<'EOF'
import torch, attn_gym
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("attn_gym", attn_gym.__file__)
EOF
echo "setup done. Next: source $VENV/bin/activate && bash exp_h20/run.sh all"

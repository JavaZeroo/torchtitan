#!/usr/bin/bash
# Launch the DistMuon harness under torchrun.
#   NGPU=8 REPO=/path/to/checkout CONFIG=moonlight_slice run_bench.sh --mb.out=... <tyro args>
# SHARED_GPU=1 lets several NCCL ranks share one device (local validation only).
set -e
NGPU=${NGPU:-8}
export LOG_RANK=${LOG_RANK:-0}
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-$(cd "$HARNESS_DIR/../.." && pwd)}
VENV=${VENV:-}
MODULE=${MODULE:-bench_configs}
CONFIG=${CONFIG:-moonlight_slice}
SCRIPT=${SCRIPT:-muon_bench.py}
cd "$REPO"
if [ "${SHARED_GPU:-0}" = "1" ]; then
  export NCCL_MULTI_RANK_GPU_ENABLE=1
fi
PYTHONPATH="$REPO:$HARNESS_DIR:$PYTHONPATH" PYTORCH_ALLOC_CONF="expandable_segments:True" \
${VENV:+$VENV/bin/}torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
  "$HARNESS_DIR/$SCRIPT" --module ${MODULE} --config ${CONFIG} "$@"

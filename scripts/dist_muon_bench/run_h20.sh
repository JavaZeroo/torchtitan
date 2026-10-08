#!/usr/bin/env bash
# One-shot DistMuon session on one 8-GPU host (written for 8x H20; any 8-GPU
# NVLink box works). Run from the optimized checkout after setup_h20.sh:
#   bash scripts/dist_muon_bench/run_h20.sh [WORK_DIR]
# It checks out the branch's history as variant worktrees (base = the commit
# below the first DistMuon commit, abc = balanced placement + BF16 wire and
# scratch, g = + Newton-Schulz graph replay, full = HEAD), runs every
# experiment base vs variants, the GPU tests and the bmm probe, then writes
# WORK_DIR/results/summary.txt and WORK_DIR/results_h20_<date>.tgz. Send back
# the tarball. Failed experiments are logged and skipped, never fatal.
set -uo pipefail
WORK=${1:-$HOME/distmuon_h20}
export NGPU=${NGPU:-8} VENV=${VENV:-$WORK/venv}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
H=$SRC/scripts/dist_muon_bench
CODE=$WORK/code; RES=$WORK/results
mkdir -p "$CODE" "$RES"
cd "$SRC"

commit_by_subject() { git log --format=%H --fixed-strings --grep="$1" -n1 HEAD; }
FIRST=$(commit_by_subject "Pack redistributed inputs and directions in BF16")
declare -A VARIANT=(
  [base]="$FIRST^"
  [abc]="$(commit_by_subject "Keep every runtime scratch tensor in the Newton-Schulz dtype")"
  [g]="$(commit_by_subject "Start a new graph pool after the cache is cleared")"
  [full]="HEAD"
)
for v in base abc g full; do
  sha=$(git rev-parse "${VARIANT[$v]}")
  if [ ! -d "$CODE/$v" ]; then git worktree add -f --detach "$CODE/$v" "$sha" >/dev/null; fi
  echo "$sha" > "$CODE/$v/.commit"
  echo "variant $v = $sha $(git log -1 --format=%s "$sha")"
done

run() {  # variant experiment-name command...
  local v=$1 name=$2; shift 2
  export REPO=$CODE/$v; local OUT=$RES/$v; mkdir -p "$OUT"
  if [ -f "$OUT/$name/rank0.json" ]; then echo "=== [$v] $name (done)"; return; fi
  local start=$(date +%s)
  echo "=== [$v] $name"
  if "$@" --mb.out="$OUT/$name" --output-dir "$OUT/$name/dump" > "$OUT/$name.log" 2>&1; then
    echo "    ok ($(( $(date +%s) - start )) s)"; grep -h "muon_bench\]" "$OUT/$name.log" | tail -1
  else
    echo "    FAILED ($(( $(date +%s) - start )) s)"
    grep -v register_constant "$OUT/$name.log" | grep -E "Error|error:|Traceback|OutOfMemory" | head -5
  fi
}
optim() {  # variant experiment-name config extra-env...
  local v=$1 name=$2 cfg=$3; shift 3
  run "$v" "$name" env "$@" CONFIG=$cfg bash "$H/run_bench.sh" \
    --mb.steps=12 --mb.warmup=3 --mb.profile_step=8 --mb.profile_ranks=0,3
}
train() {  # variant experiment-name config steps extra-env...
  local v=$1 name=$2 cfg=$3 steps=$4; shift 4
  run "$v" "$name" env "$@" MB_STEPS=$steps MB_TB=1 CONFIG=$cfg bash "$H/run_bench.sh" \
    --mb.mode=train --mb.warmup=3
}

echo "##### 1. ablation: base / abc / g / full"
for v in base abc g full; do
  optim $v k25_3_ep8 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16
  optim $v moonlight13_ep8 moonlight_slice MB_LAYERS=13
done
for v in base full; do
  optim $v moonlight27_ep8 moonlight_slice MB_LAYERS=27
done

echo "##### 2. memory at the real per-rank expert count (K2.5: 384 experts / EP 8 = 48 per rank)"
for v in base full; do
  optim $v k25_3_e256_ep8 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=256
  optim $v k25_3_e384_ep8 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=384
done

echo "##### 3. graph threshold sweep (full): none, 2^22, 2^24 (default), 2^26, 2^30"
optim full moonlight13_ep8_graph_none moonlight_slice MB_LAYERS=13 MB_NO_NS_GRAPHS=1
optim full moonlight13_ep8_graph_22 moonlight_slice MB_LAYERS=13 MB_NS_GRAPH_MAX_NUMEL=4194304
optim full moonlight13_ep8_graph_26 moonlight_slice MB_LAYERS=13 MB_NS_GRAPH_MAX_NUMEL=67108864
optim full moonlight13_ep8_graph_30 moonlight_slice MB_LAYERS=13 MB_NS_GRAPH_MAX_NUMEL=1073741824
optim full k25_3_ep8_graph_none kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16 MB_NO_NS_GRAPHS=1
optim full k25_3_ep8_graph_30 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16 MB_NS_GRAPH_MAX_NUMEL=1073741824

echo "##### 4. piece size sweep (full, 32 experts per rank): 2^24, 2^26 (default), 2^28, none"
optim full k25_3_e256_ep8_piece_24 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=256 MB_NS_PIECE_NUMEL=16777216
optim full k25_3_e256_ep8_piece_28 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=256 MB_NS_PIECE_NUMEL=268435456
optim full k25_3_e256_ep8_piece_none kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=256 MB_NS_PIECE_NUMEL=1099511627776

echo "##### 5. HSDP: replicate 2 x shard 4, EP 4"
for v in base full; do
  optim $v moonlight13_hsdp2x4 moonlight_slice MB_LAYERS=13 MB_DP_REPLICATE=2 MB_DP_SHARD=4 MB_EP=4
  optim $v k25_3_hsdp2x4 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16 MB_DP_REPLICATE=2 MB_DP_SHARD=4 MB_EP=4
done

echo "##### 6. numerics and optimizer share of a training step"
for v in base full; do
  train $v train_debug debugmodel 40
  train $v train_k3_debug kimi_k3_debug 40
  train $v train_moonlight27 moonlight_slice 8 MB_LAYERS=27 MB_SEQ_LEN=4096 MB_TOKENS_PER_MB=8192
done

echo "##### 7. tests and the bmm chunk-invariance probe (full)"
(cd "$CODE/full" && "$VENV/bin/python" -m pytest tests/unit_tests/gpu/flex_shard tests/unit_tests/cpu/flex_shard -q 2>&1 | tail -3) | tee "$RES/tests_full.log"
"$VENV/bin/python" "$H/bmm_invariance_probe.py" "$CODE/full" > "$RES/bmm_probe.log" 2>&1 && tail -8 "$RES/bmm_probe.log"
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RES/gpus.txt"; nvidia-smi topo -m >> "$RES/gpus.txt" 2>/dev/null
"$VENV/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda)" > "$RES/torch.txt"
"$VENV/bin/python" "$H/summarize_session.py" "$RES" > "$RES/summary.txt" 2>&1 || true

STAMP=$(date +%Y%m%d_%H%M)
tar -C "$WORK" -czf "$WORK/results_h20_$STAMP.tgz" --exclude='comm_traces' --exclude='structured_logs' --exclude='*/dump/profile_trace' results
echo "SESSION_DONE -> $WORK/results_h20_$STAMP.tgz"
echo "summary:"; cat "$RES/summary.txt" | head -80

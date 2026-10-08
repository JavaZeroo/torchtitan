#!/usr/bin/env bash
# Round 2 of the 8-GPU DistMuon session. Run from the optimized checkout after
# run_h20.sh (same WORK_DIR; finished experiments are skipped):
#   bash scripts/dist_muon_bench/run_h20_r2.sh [WORK_DIR]
# Sections: run-to-run noise, HSDP replica duplication (FSDP 4 on 4 GPUs vs
# HSDP 2x4 and 4x2), graph/piece thresholds at the real expert counts, small
# expert matrices, production-like training steps, the Moonlight 16B-A3B
# production recipe (downloads its tokenizer; train mode streams c4), and a
# Kimi K3 debug optimizer trace. Ends with results_h20_r2_<date>.tgz.
set -uo pipefail
WORK=${1:-$HOME/distmuon_h20}
export NGPU=${NGPU:-8} VENV=${VENV:-$WORK/venv}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
H=$SRC/scripts/dist_muon_bench
CODE=$WORK/code; RES=$WORK/results
mkdir -p "$CODE" "$RES"
cd "$SRC"
source "$H/h20_lib.sh"
G30=1073741824

echo "##### A. run-to-run noise (full, repeats of round-1 experiments)"
optim full k25_3_ep8_rep1 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16
optim full k25_3_ep8_rep2 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16
optim full moonlight13_ep8_rep1 moonlight_slice MB_LAYERS=13

echo "##### B. HSDP replica duplication: FSDP 4 on 4 GPUs vs HSDP 2x4 / 4x2 on 8"
for v in base full; do
  NGPU=4 optim $v k25_3_fsdp4 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16 MB_DP_SHARD=4 MB_EP=4
  NGPU=4 optim $v moonlight13_fsdp4 moonlight_slice MB_LAYERS=13 MB_DP_SHARD=4 MB_EP=4
  optim $v k25_3_hsdp4x2 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16 MB_DP_REPLICATE=4 MB_DP_SHARD=2 MB_EP=2
  optim $v moonlight13_hsdp4x2 moonlight_slice MB_LAYERS=13 MB_DP_REPLICATE=4 MB_DP_SHARD=2 MB_EP=2
done
# Trace the heavy shard coordinate (rank 1) of the 2x4 layout.
run full k25_3_hsdp2x4_trace01 env MB_LAYERS=3 MB_EXPERTS=16 MB_DP_REPLICATE=2 MB_DP_SHARD=4 MB_EP=4 \
  CONFIG=kimi_k2_5_slice bash "$H/run_bench.sh" --mb.steps=12 --mb.warmup=3 --mb.profile_step=8 --mb.profile_ranks=0,1

echo "##### C. thresholds at the real expert counts and with small expert matrices"
optim full k25_3_e256_ep8_graph_30 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=256 MB_NS_GRAPH_MAX_NUMEL=$G30
optim full k25_3_e384_ep8_graph_30 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=384 MB_NS_GRAPH_MAX_NUMEL=$G30
optim full k25_3_e384_ep8_piece_27 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=384 MB_NS_PIECE_NUMEL=134217728
optim full k25_3_e384_ep8_piece_28 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=384 MB_NS_PIECE_NUMEL=268435456
optim full k25_3_e384_ep8_piece_27_graph_30 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=384 MB_NS_PIECE_NUMEL=134217728 MB_NS_GRAPH_MAX_NUMEL=$G30
optim full moonlight27_ep8_graph_30 moonlight_slice MB_LAYERS=27 MB_NS_GRAPH_MAX_NUMEL=$G30
for v in base full; do
  optim $v moonlight13_e128_ep8 moonlight_slice MB_LAYERS=13 MB_EXPERTS=128
  optim $v moonlight13_e256_ep8 moonlight_slice MB_LAYERS=13 MB_EXPERTS=256
done
optim full moonlight13_e256_ep8_graph_30 moonlight_slice MB_LAYERS=13 MB_EXPERTS=256 MB_NS_GRAPH_MAX_NUMEL=$G30

echo "##### D. production-like training steps (optimizer share, memory)"
for v in base full; do
  train $v train_k25_3_e256_4k kimi_k2_5_slice 8 MB_LAYERS=3 MB_EXPERTS=256 MB_SEQ_LEN=4096 MB_TOKENS_PER_MB=8192
  train $v train_moonlight27_16k moonlight_slice 8 MB_LAYERS=27 MB_SEQ_LEN=4096 MB_TOKENS_PER_MB=16384
  train $v train_moonlight27_8k_seq moonlight_slice 8 MB_LAYERS=27 MB_SEQ_LEN=8192 MB_TOKENS_PER_MB=16384
done

echo "##### E. Moonlight 16B-A3B production recipe (real vocabulary, 27 layers, EP 8)"
for v in base full; do
  if [ ! -d "$CODE/$v/assets/hf/Moonlight-16B-A3B" ]; then
    (cd "$CODE/$v" && "$VENV/bin/python" scripts/download_hf_assets.py --repo_id moonshotai/Moonlight-16B-A3B --assets tokenizer > "$RES/assets_$v.log" 2>&1) || echo "    asset download failed for $v (see $RES/assets_$v.log)"
  fi
  run $v moonlight16b_optim env MODULE=torchtitan_recipes.tests.models.kimi_k2_7 CONFIG=moonlight_16b_a3b \
    bash "$H/run_bench.sh" --mb.steps=12 --mb.warmup=3 --mb.profile_step=8 --mb.profile_ranks=0,3
  run $v moonlight16b_train env MODULE=torchtitan_recipes.tests.models.kimi_k2_7 CONFIG=moonlight_16b_a3b MB_STEPS=8 MB_TB=1 \
    bash "$H/run_bench.sh" --mb.mode=train --mb.warmup=3
done

echo "##### F. Kimi K3 debug: optimizer trace, and run-to-run determinism of the training itself"
for v in base full; do
  optim $v k3_debug_optim kimi_k3_debug
  # Round 1 found K3 (BF16 parameters) not bitwise: the per-block ratio was
  # applied in FP32 instead of the storage dtype. v2 runs the fixed branch;
  # base/train_k3_debug_v2 against base/train_k3_debug is the determinism check.
  train $v train_k3_debug_v2 kimi_k3_debug 40
done

echo "##### G. Newton-Schulz time per matrix shape (static plan of rank 0, fake backend)"
for cfg in "moonlight_slice MB_LAYERS=27" "kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=384" "kimi_k3_debug"; do
  set -- $cfg; name=$1; shift
  ( cd "$CODE/full" && env "$@" PYTHONPATH="$CODE/full:$H" NGPU=8 LOCAL_RANK=0 RANK=0 WORLD_SIZE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29777 \
      "$VENV/bin/python" "$H/plan_dump.py" --module bench_configs --config $name --comm-backend fake --mb.out="$RES/plan_$name.json" > "$RES/plan_$name.log" 2>&1 \
    && PYTHONPATH="$CODE/full:$H" "$VENV/bin/python" "$H/ns_shape_bench.py" "$RES/plan_$name.json" > "$RES/ns_shapes_$name.txt" 2>&1 \
    && tail -14 "$RES/ns_shapes_$name.txt" ) || echo "    ns shape bench failed for $name"
done

"$VENV/bin/python" "$H/summarize_session.py" "$RES" > "$RES/summary.txt" 2>&1 || true
bash "$H/h20_digest.sh" "$WORK"
STAMP=$(date +%Y%m%d_%H%M)
tar -C "$WORK" -czf "$WORK/results_h20_r2_$STAMP.tgz" --exclude='comm_traces' --exclude='structured_logs' --exclude='*/dump/profile_trace' results
echo "SESSION_DONE -> $WORK/results_h20_r2_$STAMP.tgz"

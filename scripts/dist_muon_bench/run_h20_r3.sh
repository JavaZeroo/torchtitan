#!/usr/bin/env bash
# Round 3 of the 8-GPU DistMuon session. Run from the optimized checkout after
# rounds 1 and 2 (same WORK_DIR; finished experiments are skipped):
#   bash scripts/dist_muon_bench/run_h20_r3.sh [WORK_DIR]
# Sections: Kimi K3 at released shapes (dim 7168, 96 heads, latent MoE, BF16
# training dtype), the memory headline at 64 experts per rank and training
# with 48, dense-only stacks, pipeline parallelism, checkpoint save and
# resume, a 200-step run, collective bandwidth for a replica dedupe, HSDP
# with the real expert count, and Moonlight 16B at 32k tokens per rank.
set -uo pipefail
WORK=${1:-$HOME/distmuon_h20}
export NGPU=${NGPU:-8} VENV=${VENV:-$WORK/venv}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
H=$SRC/scripts/dist_muon_bench
CODE=$WORK/code; RES=$WORK/results
mkdir -p "$CODE" "$RES"
cd "$SRC"
source "$H/h20_lib.sh"

echo "##### A. Kimi K3 at released shapes: 5 layers (dense, KDA, KDA, MLA, MLA), 32 and 64 experts"
for v in base abc g full; do
  optim $v k3_5_ep8 kimi_k3_slice MB_LAYERS=5 MB_EXPERTS=32
done
for v in base full; do
  optim $v k3_5_e64_ep8 kimi_k3_slice MB_LAYERS=5 MB_EXPERTS=64
  optim $v k3_9_ep8 kimi_k3_slice MB_LAYERS=9 MB_EXPERTS=32
  train $v train_k3_5 kimi_k3_slice 8 MB_LAYERS=5 MB_EXPERTS=32 MB_SEQ_LEN=4096 MB_TOKENS_PER_MB=8192
done

echo "##### B. memory headline: K2.5 at 64 experts per rank, and training with 48"
for v in base full; do
  optim $v k25_3_e512_ep8 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=512
  train $v train_k25_3_e384_4k kimi_k2_5_slice 6 MB_LAYERS=3 MB_EXPERTS=384 MB_SEQ_LEN=4096 MB_TOKENS_PER_MB=8192
done

echo "##### C. dense-only stacks (no experts): K2.5 shapes 4 layers, Moonlight shapes 8 layers"
for v in base full; do
  optim $v k25_dense4_ep8 kimi_k2_5_slice MB_LAYERS=4 MB_DENSE_LAYERS=4
  optim $v moonlight_dense8_ep8 moonlight_slice MB_LAYERS=8 MB_DENSE_LAYERS=8
done

echo "##### D. pipeline parallelism: PP 2 x FSDP 4 x EP 4, 1F1B, 4 microbatches"
for v in base full; do
  train $v train_moonlight13_pp2 moonlight_slice 8 MB_LAYERS=13 MB_PP=2 MB_DP_SHARD=4 MB_EP=4 MB_PP_MICROBATCHES=4 MB_SEQ_LEN=2048 MB_TOKENS_PER_MB=2048
  train $v train_k25_3_pp2 kimi_k2_5_slice 8 MB_LAYERS=3 MB_EXPERTS=16 MB_PP=2 MB_DP_SHARD=4 MB_EP=4 MB_PP_MICROBATCHES=4 MB_SEQ_LEN=2048 MB_TOKENS_PER_MB=2048
done

echo "##### E. checkpoint save at step 5 and resume to step 10 (digests must match the uninterrupted run)"
for v in base full; do
  train $v ckpt_moonlight5_full10 moonlight_slice 10 MB_LAYERS=5 MB_CKPT_INTERVAL=5
  run $v ckpt_moonlight5_resume5 env MB_LAYERS=5 MB_STEPS=10 MB_CKPT_INTERVAL=5 MB_RESUME_STEP=5 MB_TB=1 \
    MB_DUMP_DIR="$RES/$v/ckpt_moonlight5_full10/dump" CONFIG=moonlight_slice \
    bash "$H/run_bench.sh" --mb.mode=train --mb.warmup=0
done

echo "##### F. 200 training steps of Moonlight 27 layers (bitwise over a long run, memory stability)"
for v in base full; do
  train $v train_moonlight27_200 moonlight_slice 200 MB_LAYERS=27 MB_SEQ_LEN=4096 MB_TOKENS_PER_MB=8192
done

echo "##### G. collective bandwidth over the world, halves and quarters"
"$VENV/bin/torchrun" --nproc_per_node="$NGPU" --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  "$H/nccl_bw_probe.py" > "$RES/nccl_bw.txt" 2>&1 && grep -E "group" "$RES/nccl_bw.txt" | head -12

echo "##### H. HSDP 2x4 with the real K2.5 expert count, and Moonlight 16B at 32k tokens per rank"
for v in base full; do
  optim $v k25_3_e256_hsdp2x4 kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=256 MB_DP_REPLICATE=2 MB_DP_SHARD=4 MB_EP=4
  train $v moonlight16b_train_32k moonlight_16b 6 MB_SEQ_LEN=4096 MB_TOKENS_PER_MB=32768
done

echo "##### I. Newton-Schulz shapes of the K3 slice (rank 0)"
( cd "$CODE/full" && env MB_LAYERS=5 MB_EXPERTS=32 PYTHONPATH="$CODE/full:$H" NGPU=8 LOCAL_RANK=0 RANK=0 WORLD_SIZE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29778 \
    "$VENV/bin/python" "$H/plan_dump.py" --module bench_configs --config kimi_k3_slice --comm-backend fake --mb.out="$RES/plan_kimi_k3_slice.json" > "$RES/plan_kimi_k3_slice.log" 2>&1 \
  && PYTHONPATH="$CODE/full:$H" "$VENV/bin/python" "$H/ns_shape_bench.py" "$RES/plan_kimi_k3_slice.json" > "$RES/ns_shapes_kimi_k3_slice.txt" 2>&1 \
  && tail -12 "$RES/ns_shapes_kimi_k3_slice.txt" ) || echo "    ns shape bench failed for the K3 slice"

"$VENV/bin/python" "$H/summarize_session.py" "$RES" > "$RES/summary.txt" 2>&1 || true
bash "$H/h20_digest.sh" "$WORK"
STAMP=$(date +%Y%m%d_%H%M)
tar -C "$WORK" -czf "$WORK/results_h20_r3_$STAMP.tgz" --exclude='comm_traces' --exclude='structured_logs' --exclude='*/dump/profile_trace' --exclude='*/dump/checkpoint' results
echo "SESSION_DONE -> $WORK/results_h20_r3_$STAMP.tgz"

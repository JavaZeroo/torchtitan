#!/usr/bin/env bash
# 8x H20 round 5: main, #5468, PR3 and the #5468 + PR3 stack at more sizes,
# layouts and modes.
#
# Usage:
#   PY=/path/to/venv/bin/python bash scripts/distmuon_h20_round5/run.sh [OUT_DIR]
# PARTS="1 2 3 4 5" selects parts (default all). Every run starts the selected
# parts from scratch and overwrites their old results. The summary is printed
# and saved to OUT_DIR/summary.txt; paste that text back.
set -uo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
OUT=$(mkdir -p "${1:-$REPO/outputs/h20-round5}" && cd "${1:-$REPO/outputs/h20-round5}" && pwd)
PY=${PY:-python3}
PARTS=${PARTS:-"1 2 3 4 5"}
H=$REPO/scripts/dist_muon_bench
VARIANTS="main pr5468 pr3 both"

declare -A SHA=(
  [main]=8a3f5bcaa4fd51ea6af6fb4bbb90ef03bcb139d0
  [pr5468]=690671216495121a738684abc9a451b76823a123
  [pr3]=1a69066b1bd33ff5722cb80444bc59f906e2a0f2
  [both]=571375f1c3738b0e664b5de7dd4c54516530555a
)

echo "repo $REPO @ $(git -C "$REPO" rev-parse --short HEAD); results in $OUT"
"$PY" -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), 'x', torch.cuda.device_count())"
# Fresh source checkouts every run; prune forgets worktrees whose directories are gone.
rm -rf "$OUT/src"
git -C "$REPO" worktree prune
mkdir -p "$OUT/src" "$OUT/logs"
for v in $VARIANTS; do
  git -C "$REPO" worktree add -q --detach "$OUT/src/$v" "${SHA[$v]}" || exit 1
done

has() { [[ " $PARTS " == *" $1 "* ]]; }

# run PART NAME VARIANT CONFIG [ENV=VALUE...] [-- muon_bench options]
# NP=4 before the call runs 4 processes on GPUs 0-3 (default 8).
run() {
  local part=$1 name=$2 v=$3 cfg=$4; shift 4
  local envs=() extra=() np=${NP:-8}
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ $# -gt 0 ] && shift && extra=("$@")
  local dir=$OUT/part$part/$name/$v log=$OUT/logs/part$part-$name-$v.log
  rm -rf "$dir"
  mkdir -p "$dir"
  local start=$SECONDS
  if (cd "$OUT/src/$v" && env "${envs[@]}" CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((np - 1)))" \
      PYTHONPATH="$OUT/src/$v:$H" PYTORCH_ALLOC_CONF=expandable_segments:True \
      "$PY" -m torch.distributed.run --standalone --nproc_per_node="$np" --local-ranks-filter 0 \
      "$H/muon_bench.py" --module bench_configs --config "$cfg" \
      --mb.out="$dir" --output-dir "$dir/dump" --mb.steps=12 --mb.warmup=3 "${extra[@]}") \
      > "$log" 2>&1; then
    echo "  [$v] $name ok ($((SECONDS - start)) s): $(grep -h 'muon_bench\]' "$log" | tail -1)"
  else
    echo "  [$v] $name FAILED ($((SECONDS - start)) s), see logs/part$part-$name-$v.log"
    grep -E "Error|error:|OutOfMemory" "$log" | grep -v register_constant | tail -2
  fi
  if [ -f "$dir/trace_rank0.json.gz" ]; then
    "$PY" "$H/trace_report.py" "$dir/trace_rank0.json.gz" > "$dir/trace_report.txt" 2>&1
  fi
}

if has 1; then
  echo "== 1: expert count sweep, optimizer only, FSDP8 EP8"
  for e in 16 64 128 256 384; do
    for v in $VARIANTS; do run 1 "k25_e$e" "$v" kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS="$e"; done
  done
  for e in 32 64 128 256 384; do
    for v in $VARIANTS; do run 1 "k3_e$e" "$v" kimi_k3_slice MB_LAYERS=5 MB_EXPERTS="$e"; done
  done
fi

if has 2; then
  echo "== 2: other layouts, optimizer only (K2.5 16 experts, K3 32 experts)"
  for v in $VARIANTS; do
    NP=4 run 2 k25_fsdp4_ep4 "$v" kimi_k2_5_slice MB_EP=4
    NP=4 run 2 k3_fsdp4_ep4 "$v" kimi_k3_slice MB_EP=4
    run 2 k25_fsdp8_ep4 "$v" kimi_k2_5_slice MB_EP=4
    run 2 k3_fsdp8_ep4 "$v" kimi_k3_slice MB_EP=4
    run 2 k25_hsdp2x4_ep4 "$v" kimi_k2_5_slice MB_DP_REPLICATE=2 MB_DP_SHARD=4 MB_EP=4
    run 2 k3_hsdp2x4_ep4 "$v" kimi_k3_slice MB_DP_REPLICATE=2 MB_DP_SHARD=4 MB_EP=4
  done
fi

if has 3; then
  echo "== 3: Moonlight shapes (many small matrices), optimizer only"
  for layers in 13 27; do
    for v in $VARIANTS; do run 3 "moonlight_l$layers" "$v" moonlight_slice MB_LAYERS="$layers"; done
  done
fi

if has 4; then
  echo "== 4: profiles of one optimizer step (step 8, rank 0)"
  for v in main pr5468 both; do
    run 4 k25_e16 "$v" kimi_k2_5_slice -- --mb.profile_step=8 --mb.profile_ranks=0
    run 4 k3_e32 "$v" kimi_k3_slice -- --mb.profile_step=8 --mb.profile_ranks=0
    run 4 moonlight_l13 "$v" moonlight_slice MB_LAYERS=13 -- --mb.profile_step=8 --mb.profile_ranks=0
  done
fi

if has 5; then
  echo "== 5: real training loop (forward, backward, optimizer)"
  for v in $VARIANTS; do
    run 5 k3debug_100steps "$v" kimi_k3_debug MB_STEPS=100 -- --mb.mode=train
    run 5 k25_e16_20steps "$v" kimi_k2_5_slice MB_STEPS=20 -- --mb.mode=train
    run 5 k3_e32_20steps "$v" kimi_k3_slice MB_STEPS=20 -- --mb.mode=train
  done
fi

"$PY" "$REPO/scripts/distmuon_h20_round5/summarize.py" "$OUT" | tee "$OUT/summary.txt"

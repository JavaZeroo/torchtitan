#!/usr/bin/env bash
# 8x H20 session: PR3 numerics, real-shape slices, K3 debug profile, GPU tests.
#
# Usage (from the repository root):
#   PY=/path/to/venv/bin/python bash scripts/distmuon_h20_round4/run.sh [OUT_DIR]
# PARTS="A B C D" selects parts (default all). Finished experiments are skipped,
# so re-running the same command resumes. The final summary is printed and saved
# to OUT_DIR/summary.txt; paste that text back.
set -uo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
OUT=$(mkdir -p "${1:-$REPO/outputs/h20-round4}" && cd "${1:-$REPO/outputs/h20-round4}" && pwd)
PY=${PY:-python3}
PARTS=${PARTS:-"A B C D"}
H=$REPO/scripts/dist_muon_bench
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

declare -A SHA=(
  [main]=8a3f5bcaa4fd51ea6af6fb4bbb90ef03bcb139d0
  [pr5468]=690671216495121a738684abc9a451b76823a123
  [pr3]=1a69066b1bd33ff5722cb80444bc59f906e2a0f2
)

mkdir -p "$OUT/src" "$OUT/logs"
echo "repo $REPO @ $(git -C "$REPO" rev-parse --short HEAD); results in $OUT"
"$PY" -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), 'x', torch.cuda.device_count())"
for v in main pr5468 pr3; do
  if [ ! -d "$OUT/src/$v" ]; then
    git -C "$REPO" worktree add -q --detach "$OUT/src/$v" "${SHA[$v]}" || exit 1
  fi
  [ "$(git -C "$OUT/src/$v" rev-parse HEAD)" = "${SHA[$v]}" ] || { echo "src/$v is not at ${SHA[$v]}"; exit 1; }
done

has() { [[ " $PARTS " == *" $1 "* ]]; }

bench() {  # variant name config [ENV=VALUE...] -- [extra muon_bench options...]
  local v=$1 name=$2 cfg=$3; shift 3
  local envs=() extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ $# -gt 0 ] && shift && extra=("$@")
  local dir=$OUT/bench/$name/$v
  if [ -f "$dir/rank0.json" ]; then echo "  [$v] $name: done"; return; fi
  echo "  [$v] $name: running"
  mkdir -p "$dir"
  local start=$SECONDS
  if (cd "$OUT/src/$v" && env "${envs[@]}" PYTHONPATH="$OUT/src/$v:$H" \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 --local-ranks-filter 0 \
      "$H/muon_bench.py" --module bench_configs --config "$cfg" \
      --mb.out="$dir" --output-dir "$dir/dump" --mb.steps=12 --mb.warmup=3 "${extra[@]}") \
      > "$OUT/logs/$name-$v.log" 2>&1; then
    echo "    ok ($((SECONDS - start)) s): $(grep -h 'muon_bench\]' "$OUT/logs/$name-$v.log" | tail -1)"
  else
    echo "    FAILED ($((SECONDS - start)) s), see logs/$name-$v.log"
    grep -E "Error|error:|OutOfMemory" "$OUT/logs/$name-$v.log" | grep -v register_constant | tail -3
  fi
}

if has A; then
  echo "== A: K3 debugmodel real training, main vs PR3 (numerics, then 3 timed pairs)"
  resume=()
  [ -f "$OUT/k3debug-pr3/manifest.json" ] && resume=(--resume)
  (cd "$REPO" && "$PY" -m scripts.distmuon_k3_h20.run --output "$OUT/k3debug-pr3" \
      --timeout 1800 "${resume[@]}") > "$OUT/logs/A-k3debug-pr3.log" 2>&1 \
    && echo "  A done" || echo "  A FAILED, see logs/A-k3debug-pr3.log"
fi

if has B; then
  echo "== B: real-shape slices, optimizer only (synthetic gradients), two rounds"
  for round in 1 2; do
    order=(main pr5468 pr3)
    [ "$round" = 2 ] && order=(pr3 pr5468 main)
    for v in "${order[@]}"; do
      bench "$v" "k3_slice_r$round" kimi_k3_slice MB_LAYERS=5 MB_EXPERTS=32
      bench "$v" "k25_3_r$round" kimi_k2_5_slice MB_LAYERS=3 MB_EXPERTS=16
    done
  done
fi

if has C; then
  echo "== C: K3 debugmodel optimizer profile (step 8, rank 0)"
  for v in main pr5468; do
    bench "$v" k3debug_profile kimi_k3_debug -- --mb.profile_step=8 --mb.profile_ranks=0
    trace=$OUT/bench/k3debug_profile/$v/trace_rank0.json.gz
    [ -f "$trace" ] && "$PY" "$H/trace_report.py" "$trace" > "$OUT/bench/k3debug_profile/$v/trace_report.txt" 2>&1
  done
fi

if has D; then
  echo "== D: flex_shard GPU unit tests"
  for v in main pr5468 pr3; do
    log=$OUT/logs/D-gputest-$v.log
    if [ -f "$log" ] && grep -qE "passed|failed" "$log"; then echo "  [$v] done"; continue; fi
    (cd "$OUT/src/$v" && PYTHONPATH="$OUT/src/$v" timeout 3600 "$PY" -m pytest -q -p no:cacheprovider \
        tests/unit_tests/gpu/distributed/flex_shard) > "$log" 2>&1
    echo "  [$v] $(tail -1 "$log")"
  done
fi

"$PY" "$REPO/scripts/distmuon_h20_round4/summarize.py" "$OUT" | tee "$OUT/summary.txt"

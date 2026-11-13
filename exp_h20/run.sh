#!/bin/bash
# Kimi K3.1 (#6007) validation on 8x H20. Experiment-only, not for upstream.
#
# Usage (from the repo root, inside the container prepared by setup.sh):
#   bash exp_h20/run.sh all                 # every stage, in order
#   bash exp_h20/run.sh par k3bisect        # only the named stages
# Stages: env unit ci par pp neg k3bisect
# Each stage deletes its own output folder first and reruns from scratch.
# Then: python exp_h20/analyze.py   -> prints the short summary to send back.

set -u
cd "$(dirname "$0")/.."
REPO=$PWD
OUT=${OUT:-$REPO/../k31_h20_out}
AG_SRC=${AG_SRC:-$REPO/../attention-gym}
STEPS=${STEPS:-20}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
mkdir -p "$OUT"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

log() { echo "[$(date +%H:%M:%S)] $*"; }

stage_env() {
  local d=$OUT/env; rm -rf "$d"; mkdir -p "$d"
  {
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    python - <<'EOF'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cc", torch.cuda.get_device_capability())
for m in ("torchvision", "av", "triton", "torchao", "spmd_types", "torch_remat", "attn_gym", "cutlass", "flash_attn_interface"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "MISSING", type(e).__name__, e)
EOF
    git -C "$REPO" log --oneline -1
    git -C "$AG_SRC" log --oneline -1 2>/dev/null || echo "attention-gym source missing at $AG_SRC"
  } > "$d/env.txt" 2>&1
  log "env -> $d/env.txt"
}

stage_unit() {
  local d=$OUT/unit; rm -rf "$d"; mkdir -p "$d"
  python -m pytest -q -p no:cacheprovider \
    tests/unit_tests/cpu/models/kimi_k3_1 tests/unit_tests/cpu/models/kimi_k3 \
    tests/unit_tests/cpu/models/test_build_model_config.py \
    tests/unit_tests/cpu/models/common/test_optional_imports.py \
    tests/unit_tests/gpu/models/kimi_k3 \
    --junitxml="$d/unit.xml" > "$d/unit.log" 2>&1
  echo $? > "$d/exit"
  log "unit exit=$(cat "$d/exit")"
}

stage_ci() {
  # The H100 integration tests that touch Kimi, real process groups.
  local d=$OUT/ci; rm -rf "$d"; mkdir -p "$d"
  for t in kimi_k3_1_fsdp+ep kimi_k3_mm_allgather_kv_cp kimi_k3_mm_ulysses_cp; do
    NCCL_NVLS_ENABLE=0 python -m tests.integration_tests.run_tests \
      --test_suite h100 --execution_mode real_pg --gpu_arch_type cuda \
      --gpu_arch h100 --ngpu "$GPUS" --test_name "$t" "$d/out_$t" > "$d/$t.log" 2>&1
    echo $? > "$d/$t.exit"
    log "ci $t exit=$(cat "$d/$t.exit")"
  done
}

# One training run of an exp_h20.k31_exp config. $1 group dir, $2 config,
# $3 ngpu, $4 mode (seed|load).
_train() {
  local g=$1 cfg=$2 ngpu=$3 mode=$4
  EXP_MODE=$mode EXP_STEPS=$STEPS EXP_TB=tb_$cfg NGPU=$ngpu \
    MODULE=exp_h20.k31_exp CONFIG=$cfg ./run_train.sh --output-dir="$g" \
    > "$g/$cfg.log" 2>&1
  echo $? > "$g/$cfg.exit"
  log "par $cfg (ngpu=$ngpu) exit=$(cat "$g/$cfg.exit")"
}

_twice() {
  # Runs $2 twice; the first run is kept as ${2}_run1 (run-to-run noise).
  local g=$1 cfg=$2 ngpu=$3
  _train "$g" "$cfg" "$ngpu" load
  for x in "tb_$cfg" "$cfg.log" "$cfg.exit"; do mv "$g/$x" "$g/${x/$cfg/${cfg}_run1}"; done
  _train "$g" "$cfg" "$ngpu" load
}

_fits() { [ "$1" -le "$GPUS" ] || { log "skip $2 (needs $1 GPUs, have $GPUS)"; return 1; }; }

stage_par() {
  # Same seed checkpoint and same data per comparison, 20 steps, seed 42,
  # nondeterministic. Baselines run twice to measure run-to-run noise.
  # dp_shard=2 runs compare against *_fsdp2; dp_shard=1 runs against *_dp1.
  local g
  g=$OUT/par_p; rm -rf "$g"; mkdir -p "$g"
  _train "$g" p_seed 1 seed
  _twice "$g" p_dp1 1
  _train "$g" p_pp2 2 load
  _twice "$g" p_fsdp2 2
  _train "$g" p_fsdp2_ep2 2 load
  _fits 4 p_fsdp2_pp2 && _train "$g" p_fsdp2_pp2 4 load
  _fits 4 p_fsdp2_ep2_pp2 && _train "$g" p_fsdp2_ep2_pp2 4 load

  g=$OUT/par_t; rm -rf "$g"; mkdir -p "$g"
  _train "$g" t_seed 1 seed
  _twice "$g" t_dp1 1
  _train "$g" t_tp2_ep2 2 load
  _fits 4 t_fsdp2 && _twice "$g" t_fsdp2 2
  _fits 4 t_fsdp2_ep2 && _train "$g" t_fsdp2_ep2 2 load
  _fits 4 t_fsdp2_tp2_ep2 && _train "$g" t_fsdp2_tp2_ep2 4 load
  _fits 8 t_fsdp2_tp2_ep2_pp2 && _train "$g" t_fsdp2_tp2_ep2_pp2 8 load
  return 0
}

stage_pp() {
  # PP2 runs 4 microbatches per step, so its baseline is dp_shard=1 with 4-step
  # gradient accumulation. PP logs the loss on its last stage (LOG_RANK=1).
  local g=$OUT/par_pp; rm -rf "$g"; mkdir -p "$g"
  _train "$g" p_seed 1 seed
  _twice "$g" p_ga4 1
  LOG_RANK=1 _train "$g" p_pp2 2 load
}

stage_neg() {
  local g=$OUT/neg; rm -rf "$g"; mkdir -p "$g"
  STEPS=2 _train "$g" n_tp2_16_heads 2 ""
}

stage_k3bisect() {
  # Kimi K3 debug model, 1 GPU, deterministic, 10 steps, loss + grad_norm from
  # TensorBoard, with Attention Gym from a git worktree on PYTHONPATH. v0.0.19
  # runs twice (export, then compare) to check that it repeats bitwise; then a
  # binary search finds the first commit up to ff06a9221 that changes K3.
  local d=$OUT/k3bisect; rm -rf "$d"; mkdir -p "$d"
  local base=0d0197fb36d95695dccef03436bb0649f985b32d
  local tip=ff06a922140c03fc118e6eaa64001ad139c536ec
  git -C "$AG_SRC" fetch -q origin 2>/dev/null
  mapfile -t C < <(git -C "$AG_SRC" rev-list --reverse --first-parent "$base..$tip")
  printf "%s\n" "${C[@]}" > "$d/commits.txt"
  local wt=$OUT/ag_worktrees; mkdir -p "$wt"
  # $1 sha, $2 run name, $3 extra loss_compare args; prints SAME, DIFF or CRASH.
  _k3run() {
    local sha=$1 name=$2 extra=$3
    [ -d "$wt/$sha" ] || git -C "$AG_SRC" worktree add -q --detach "$wt/$sha" "$sha"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$wt/$sha:$PYTHONPATH \
      python scripts/loss_compare.py . . \
      --baseline-module=torchtitan_recipes.tests.models.kimi_k3 \
      --baseline-config=kimi_k3_debugmodel --steps=10 --no-seed-checkpoint \
      --metrics=loss,grad_norm --baseline-ngpus=1 --test-ngpus=1 \
      --job-dump-folder="$d/dump_$name" --assert-equal $extra \
      > "$d/$name.log" 2>&1
    local code=$? status
    if [ $code -eq 0 ]; then status=SAME
    elif [ -d "$d/dump_$name/tb_baseline" ]; then status=DIFF
    else status=CRASH; fi
    echo "$name $sha $status" >> "$d/results.txt"
    log "k3bisect $name ${sha:0:9} $status" >&2
    echo $status
  }
  local r
  r=$(_k3run "$base" base "--export-result=$d/base_v0.0.19.txt")
  [ "$r" = SAME ] || return 0
  _k3run "$base" base_repeat "--import-result=$d/base_v0.0.19.txt" > /dev/null
  local n=${#C[@]} lo=-1 hi=$(( ${#C[@]} - 1 ))
  r=$(_k3run "${C[$hi]}" "c$hi" "--import-result=$d/base_v0.0.19.txt")
  [ "$r" = DIFF ] || return 0
  while [ $((hi - lo)) -gt 1 ]; do
    local mid=$(( (lo + hi) / 2 ))
    r=$(_k3run "${C[$mid]}" "c$mid" "--import-result=$d/base_v0.0.19.txt")
    case $r in SAME) lo=$mid ;; DIFF) hi=$mid ;; *) return 0 ;; esac
  done
  echo "first_diff ${C[$hi]}" >> "$d/results.txt"
  log "k3bisect first commit that changes K3: ${C[$hi]:0:9} (of $n)"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && { echo "usage: $0 all | env unit ci par pp neg k3bisect"; exit 1; }
[ "${stages[0]}" = all ] && stages=(env unit ci par pp neg k3bisect)
for s in "${stages[@]}"; do
  log "=== $s ==="
  "stage_$s"
done
log "done; run: python exp_h20/analyze.py"

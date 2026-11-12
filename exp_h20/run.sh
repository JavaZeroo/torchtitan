#!/bin/bash
# Kimi K3.1 (#6007) validation on 8x H20. Experiment-only, not for upstream.
#
# Usage (from the repo root, inside the container prepared by setup.sh):
#   bash exp_h20/run.sh all                 # every stage, in order
#   bash exp_h20/run.sh par k3bisect        # only the named stages
# Stages: env unit ci par neg k3bisect
# Each stage deletes its own output folder first and reruns from scratch.
# Then: python exp_h20/analyze.py   -> prints the short summary to send back.

set -u
cd "$(dirname "$0")/.."
REPO=$PWD
OUT=${OUT:-$REPO/../k31_h20_out}
AG_SRC=${AG_SRC:-$REPO/../attention-gym}
STEPS=${STEPS:-20}
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
for m in ("triton", "torchao", "spmd_types", "torch_remat", "attn_gym", "cutlass", "flash_attn_interface"):
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
      --gpu_arch h100 --ngpu 8 --test_name "$t" "$d/out_$t" > "$d/$t.log" 2>&1
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

stage_par() {
  # Same seed checkpoint, same data (dp_shard=2 everywhere), 20 steps, seed 42,
  # nondeterministic. *_fsdp2 runs twice to measure run-to-run noise.
  local g
  g=$OUT/par_p; rm -rf "$g"; mkdir -p "$g"
  _train "$g" p_seed 1 seed
  _train "$g" p_fsdp2 2 load
  mv "$g/tb_p_fsdp2" "$g/tb_p_fsdp2_run1"; mv "$g/p_fsdp2.log" "$g/p_fsdp2_run1.log"; mv "$g/p_fsdp2.exit" "$g/p_fsdp2_run1.exit"
  _train "$g" p_fsdp2 2 load
  _train "$g" p_fsdp2_ep2 2 load
  _train "$g" p_fsdp2_pp2 4 load
  _train "$g" p_fsdp2_ep2_pp2 4 load

  g=$OUT/par_t; rm -rf "$g"; mkdir -p "$g"
  _train "$g" t_seed 1 seed
  _train "$g" t_fsdp2 2 load
  mv "$g/tb_t_fsdp2" "$g/tb_t_fsdp2_run1"; mv "$g/t_fsdp2.log" "$g/t_fsdp2_run1.log"; mv "$g/t_fsdp2.exit" "$g/t_fsdp2_run1.exit"
  _train "$g" t_fsdp2 2 load
  _train "$g" t_fsdp2_ep2 2 load
  _train "$g" t_fsdp2_tp2_ep2 4 load
  _train "$g" t_fsdp2_tp2_ep2_pp2 8 load
}

stage_neg() {
  local g=$OUT/neg; rm -rf "$g"; mkdir -p "$g"
  STEPS=2 _train "$g" n_tp2_16_heads 2 ""
}

stage_k3bisect() {
  # Kimi K3 debug model, 1 GPU, deterministic, 10 steps, loss + grad_norm from
  # TensorBoard. Attention Gym comes from a git worktree on PYTHONPATH, one per
  # commit. Each loss_compare call runs the model twice with that commit (so
  # it also checks run-to-run repeatability) and compares against v0.0.19.
  local d=$OUT/k3bisect; rm -rf "$d"; mkdir -p "$d"
  local base=0d0197fb36d95695dccef03436bb0649f985b32d
  local tip=ff06a922140c03fc118e6eaa64001ad139c536ec
  git -C "$AG_SRC" fetch -q origin 2>/dev/null
  git -C "$AG_SRC" rev-list --reverse --first-parent "$base..$tip" > "$d/commits.txt"
  local wt=$OUT/ag_worktrees; mkdir -p "$wt"
  _k3run() {
    local sha=$1 gpu=$2 extra=$3
    [ -d "$wt/$sha" ] || git -C "$AG_SRC" worktree add -q --detach "$wt/$sha" "$sha"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$wt/$sha:$PYTHONPATH \
      python scripts/loss_compare.py . . \
      --baseline-module=torchtitan_recipes.tests.models.kimi_k3 \
      --baseline-config=kimi_k3_debugmodel --steps=10 --no-seed-checkpoint \
      --metrics=loss,grad_norm --baseline-ngpus=1 --test-ngpus=1 \
      --job-dump-folder="$d/dump_$sha" --assert-equal $extra \
      > "$d/$sha.log" 2>&1
    echo $? > "$d/$sha.exit"
  }
  _k3run "$base" 0 "--export-result=$d/base_v0.0.19.txt"
  log "k3bisect base exit=$(cat "$d/$base.exit")"
  local i=0
  while read -r sha; do
    _k3run "$sha" $((i % 8)) "--import-result=$d/base_v0.0.19.txt" &
    i=$((i + 1))
    if [ $((i % 8)) -eq 0 ]; then wait; fi
  done < "$d/commits.txt"
  wait
  log "k3bisect done ($i commits)"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && { echo "usage: $0 all | env unit ci par neg k3bisect"; exit 1; }
[ "${stages[0]}" = all ] && stages=(env unit ci par neg k3bisect)
for s in "${stages[@]}"; do
  log "=== $s ==="
  "stage_$s"
done
log "done; run: python exp_h20/analyze.py"

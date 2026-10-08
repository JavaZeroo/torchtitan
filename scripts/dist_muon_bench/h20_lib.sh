#!/usr/bin/env bash
# Shared helpers for the 8-GPU session scripts (sourced, not executed).
# Expects WORK, NGPU, VENV, SRC, H, CODE and RES to be set by the caller.
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

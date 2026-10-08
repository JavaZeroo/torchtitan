#!/usr/bin/env bash
# Text-only digest of a session's results, for hosts that cannot export files.
#   bash scripts/dist_muon_bench/h20_digest.sh [WORK_DIR]   -> WORK_DIR/results/digest.txt
# Collects the environment, the full summary, optimizer share and memory of
# the training runs, the full TensorBoard loss/grad_norm comparisons, and
# trace reports of the experiments that answer the open questions.
set -uo pipefail
WORK=${1:-$HOME/distmuon_h20}
VENV=${VENV:-$WORK/venv}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RES=$WORK/results
PY=$VENV/bin/python
OUT=$RES/digest.txt
{
echo "##### environment"
cat "$RES/torch.txt" "$RES/gpus.txt" 2>/dev/null
echo; echo "##### tests / probe"; cat "$RES/tests_full.log" "$RES/bmm_probe.log" 2>/dev/null
echo; echo "##### summary (all experiments)"
"$PY" "$H/summarize_session.py" "$RES" 2>&1
echo; echo "##### training runs: optimizer share and memory (rank 0, steps after warmup)"
"$PY" - "$RES" <<'PY'
import glob, json, os, statistics, sys
root = sys.argv[1]
for path in sorted(glob.glob(os.path.join(root, "*", "train*", "rank0.json"))):
    d = json.load(open(path))
    steps = d["steps"][3:] or d["steps"]
    opt = statistics.median(s["total_ms"] for s in steps)
    fb = [s["since_previous_ms"] for s in steps if s.get("since_previous_ms")]
    fb_med = statistics.median(fb) if fb else float("nan")
    peak = max(s["peak_bytes"] for s in steps) / 2**30
    resident = steps[-1]["resident_bytes"] / 2**30
    name = "/".join(path.split(os.sep)[-3:-1])
    print(f"{name:<32s} optimizer {opt:8.1f} ms  rest of step {fb_med:8.1f} ms  "
          f"optimizer share {100 * opt / (opt + fb_med):5.1f}%  peak {peak:6.1f} GiB  resident {resident:6.1f} GiB")
PY
echo; echo "##### TensorBoard loss / grad_norm, base vs full, every step"
for exp in $(ls "$RES/base" | grep '^train' | grep -v '\.log$'); do
  b=$RES/base/$exp/dump/tb; f=$RES/full/$exp/dump/tb
  if [ -d "$b" ] && [ -d "$f" ]; then echo "--- $exp"; "$PY" "$H/tb_metrics.py" "$b" "$f" 2>&1 | tail -45; fi
done
echo; echo "##### K3 determinism: base round 1 vs base round 2 (same tree)"
if [ -d "$RES/base/train_k3_debug_v2" ]; then "$PY" "$H/compare_runs.py" "$RES/base/train_k3_debug" "$RES/base/train_k3_debug_v2" 2>&1 | grep digests; "$PY" "$H/tb_metrics.py" "$RES/base/train_k3_debug/dump/tb" "$RES/base/train_k3_debug_v2/dump/tb" 2>&1 | tail -1; fi
echo; echo "##### Newton-Schulz time per shape (round 2)"
cat "$RES"/ns_shapes_*.txt 2>/dev/null
echo; echo "##### trace reports"
report() {  # variant experiment rank
  local t=$RES/$1/$2/trace_rank$3.json.gz
  if [ -f "$t" ]; then echo; echo "=== [$1] $2 rank $3"; "$PY" "$H/trace_report.py" "$t" 2>&1; fi
}
for v in base abc full; do report $v moonlight13_ep8 0; done
for v in base full; do report $v moonlight27_ep8 0; report $v moonlight27_ep8 3; done
for v in base full; do report $v k25_3_ep8 0; done
report full k25_3_ep8_graph_30 0
for e in k25_3_e256_ep8 k25_3_e256_ep8_piece_none k25_3_e256_ep8_piece_28 k25_3_e384_ep8; do report full $e 0; done
report base k25_3_e384_ep8 0
for v in base full; do report $v k25_3_hsdp2x4 0; report $v k25_3_hsdp2x4 3; report $v moonlight13_hsdp2x4 0; done
# round 2 experiments, if present
for e in k25_3_fsdp4 k25_3_hsdp4x2 k25_3_hsdp2x4_trace01 k25_3_e384_ep8_graph_30 k25_3_e384_ep8_piece_27 moonlight27_ep8_graph_30 moonlight13_e256_ep8 moonlight16b_optim k3_debug_optim; do
  for v in base full; do report $v $e 0; done
done
report full k25_3_hsdp2x4_trace01 1
} > "$OUT" 2>&1
echo "digest written: $OUT ($(wc -l < "$OUT") lines, $(du -h "$OUT" | cut -f1))"

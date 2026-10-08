# DistMuon benchmark harness

Tools used for the DistMuon performance and memory work. Everything drives the
unmodified TorchTitan entry points and reads optimizer internals, so the same
scripts run against any checkout (set `REPO=`).

| script | purpose |
|---|---|
| `bench_configs.py` | `moonlight_slice`, `kimi_k2_5_slice` (production shapes cut to `MB_LAYERS` layers, `MB_EXPERTS` experts), `debugmodel` (the Kimi K2.5 CI recipe), `kimi_k3_debug` (the K3 debugmodel at FSDP 8 x EP 8) and `moonlight_16b` (the production recipe on local c4_test data) |
| `plan_dump.py` | static plan: buckets, layouts, compute placement, per-rank Newton-Schulz FLOPs, traffic and reserved buffers; runs as one logical rank under `--comm-backend fake` |
| `muon_bench.py` | `--mb.mode=optim`: DistMuon alone with deterministic synthetic gradients; `--mb.mode=train`: the real training loop with the optimizer phase timed. Both write per-rank JSON with timings, allocator bytes and SHA-256 digests of every Muon parameter and momentum shard, plus an optional Kineto trace |
| `compare_runs.py` | digest equality and timing/memory comparison of two `muon_bench` output directories |
| `tb_metrics.py` | full-precision loss / grad_norm from TensorBoard, compared bitwise |
| `trace_report.py`, `trace_svg.py` | per-stream and per-kernel-family breakdown of one optimizer step, CUDA runtime calls and idle gaps, and an SVG timeline |
| `h20_digest.sh` | text-only digest of a session (summary, optimizer share, loss tables, trace reports) for hosts that cannot export files |
| `summarize_session.py` | tables over a results tree `results/<variant>/<experiment>/rank*.json` |
| `run_bench.sh` | torchrun launcher; `SHARED_GPU=1` lets several NCCL ranks share one GPU for local numerics checks (`NCCL_MULTI_RANK_GPU_ENABLE=1`) |
| `ns_shape_bench.py` | times Newton-Schulz per matrix-view shape of a plan (as issued vs one merged batch per shape class), from a `plan_dump.py` JSON with `view_shapes_by_rank` |
| `bmm_invariance_probe.py` | is batched Newton-Schulz bitwise invariant to how the batch is chunked on this GPU (decides whether piecewise orthogonalization is exact) |
| `setup_h20.sh`, `run_h20.sh`, `run_h20_r2.sh`, `h20_lib.sh` | one-shot sessions on an 8-GPU host: environment, variant worktrees from the branch history, every experiment base vs variants, tests, probe, summary and a results tarball; round 2 adds HSDP duplication, threshold sweeps at scale, production-like training and the Moonlight 16B recipe |

Run-level settings are environment variables, applied by
`common.apply_harness_env` after the recipe is built (the config loader only
takes `--module`, `--config`, `--comm-backend`):

| variable | effect |
|---|---|
| `MB_SEED` (default 42), `MB_DETERMINISTIC` (default 1) | `debug.seed`, `debug.deterministic`; required for stable digests |
| `MB_STEPS`, `MB_TIMEOUT` | `training.steps` in train mode; collective timeout in seconds |
| `MB_TB=1` | TensorBoard every step under `<out>/dump/tb` |
| `MB_DP_SHARD`, `MB_DP_REPLICATE`, `MB_EP` | parallelism degrees, rebuilt through the recipe so expert layouts realign |
| `MB_SLOTS`, `MB_NO_NS_GRAPHS`, `MB_NS_GRAPH_MAX_NUMEL`, `MB_NS_PIECE_NUMEL` | runtime knobs: pipeline slots, disable Newton-Schulz graph replay, graph-replay input threshold, piecewise orthogonalization batch threshold (the last three only act on the optimized branch) |

Examples:

```bash
# static plan of the Moonlight slice at 8 ranks
MB_LAYERS=5 NGPU=8 LOCAL_RANK=0 PYTHONPATH=$REPO:$PWD python plan_dump.py \
  --module bench_configs --config moonlight_slice --comm-backend fake --mb.out=plan.json

# optimizer-only benchmark on 8 GPUs with a trace of step 10 on ranks 0, 1 and 7
MB_LAYERS=5 NGPU=8 CONFIG=moonlight_slice bash run_bench.sh --mb.out=out/base \
  --mb.steps=16 --mb.warmup=4 --mb.profile_step=10 --mb.profile_ranks=0,1,7

# numerics: 40 training steps of the CI recipe, then compare two checkouts
MB_STEPS=40 MB_TB=1 NGPU=8 CONFIG=debugmodel bash run_bench.sh --mb.mode=train --mb.out=out/train_base
python compare_runs.py out/train_base out/train_opt
python tb_metrics.py out/train_base/dump/tb out/train_opt/dump/tb
```

## One-shot session on an 8-GPU host

From the optimized checkout, with internet access:

```bash
bash scripts/dist_muon_bench/setup_h20.sh ~/distmuon_h20   # venv with the PyTorch nightly TorchTitan main needs
bash scripts/dist_muon_bench/run_h20.sh ~/distmuon_h20     # about 1-2 hours on 8x H20
bash scripts/dist_muon_bench/run_h20_r2.sh ~/distmuon_h20  # round 2, about 2-3 hours
```

`run_h20.sh` is restartable (finished experiments are skipped) and ends with
`~/distmuon_h20/results_h20_<date>.tgz` plus `results/summary.txt`. Send the
tarball back for analysis.

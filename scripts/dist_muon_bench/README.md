# DistMuon benchmark harness

Tools used for the DistMuon performance and memory work. Everything drives the
unmodified TorchTitan entry points and reads optimizer internals, so the same
scripts run against any checkout (set `REPO=`).

| script | purpose |
|---|---|
| `bench_configs.py` | `moonlight_slice`, `kimi_k2_5_slice` (production shapes cut to `MB_LAYERS` layers, `MB_EXPERTS` experts) and `debugmodel` (the CI recipe) |
| `plan_dump.py` | static plan: buckets, layouts, compute placement, per-rank Newton-Schulz FLOPs, traffic and reserved buffers; runs as one logical rank under `--comm.backend=fake` |
| `muon_bench.py` | `--mb.mode=optim`: DistMuon alone with deterministic synthetic gradients; `--mb.mode=train`: the real training loop with the optimizer phase timed. Both write per-rank JSON with timings, allocator bytes and SHA-256 digests of every Muon parameter and momentum shard, plus an optional Kineto trace |
| `compare_runs.py` | digest equality and timing/memory comparison of two `muon_bench` output directories |
| `tb_metrics.py` | full-precision loss / grad_norm from TensorBoard, compared bitwise |
| `trace_report.py`, `trace_svg.py` | per-stream and per-kernel-family breakdown of one optimizer step, and an SVG timeline |
| `summarize_session.py` | tables over a results tree `results/<variant>/<experiment>/rank*.json` |
| `run_bench.sh` | torchrun launcher; `SHARED_GPU=1` lets several NCCL ranks share one GPU for local numerics checks (`NCCL_MULTI_RANK_GPU_ENABLE=1`) |

Examples:

```bash
# static plan of the Moonlight slice at 8 ranks
MB_LAYERS=5 NGPU=8 LOCAL_RANK=0 PYTHONPATH=$REPO:$PWD python plan_dump.py \
  --module bench_configs --config moonlight_slice --comm.backend=fake --mb.out=plan.json

# optimizer-only benchmark on 8 GPUs with a trace of step 10 on ranks 0, 1 and 7
MB_LAYERS=5 NGPU=8 CONFIG=moonlight_slice bash run_bench.sh --mb.out=out/base \
  --mb.steps=16 --mb.warmup=4 --mb.profile_step=10 --mb.profile_ranks=0,1,7 \
  --parallelism.data_parallel_shard_degree 8 --parallelism.expert_parallel_degree 8 \
  --debug.seed=42 --debug.deterministic

# numerics: 40 training steps of the CI recipe, then compare two checkouts
NGPU=8 CONFIG=debugmodel bash run_bench.sh --mb.mode=train --mb.out=out/train_base \
  --parallelism.data_parallel_shard_degree 8 --parallelism.expert_parallel_degree 8 \
  --debug.seed=42 --debug.deterministic --metrics.enable_tensorboard --metrics.log_freq=1 \
  --metrics.save_tb_folder=tb --training.steps 40
python compare_runs.py out/train_base out/train_opt
python tb_metrics.py out/train_base/dump/tb out/train_opt/dump/tb
```

`--debug.seed=42 --debug.deterministic` is required for stable digests.

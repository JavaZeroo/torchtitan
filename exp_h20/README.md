# Kimi K3.1（#6007）8 卡 H20 验证

仅用于实验的分支，不进上游。代码和 #6007 的 `kimi-k3.1-pro`（d29924d16）完全一致，只多了 `exp_h20/` 目录。

## 1. 环境

- 镜像：`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`（Ubuntu 24.04 自带 Python 3.12）。
- 驱动：先在宿主机上运行 `nvidia-smi`。
  - 595 及以上：用默认的 `CU=cu132`，和 torchtitan CI 一致。
  - 580 到 594：用 `CU=cu130`。这个源最新的 nightly 是 2026-11-02。
- 网络：需要能访问 download.pytorch.org、PyPI 和 GitHub。`requirements.txt` 里的 attn-gym 和 torch_remat 从 GitHub 安装，二分步骤还要 clone attention-gym。

```bash
git clone -b exp/kimi-k3.1-h20 https://github.com/JavaZeroo/torchtitan.git k31
docker run --gpus all --ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -it -v $PWD:/work -w /work/k31 nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 bash

# 进入容器后
bash exp_h20/setup.sh          # 驱动低于 595 时：CU=cu130 bash exp_h20/setup.sh
source /opt/k31/bin/activate
```

## 2. 运行

```bash
bash exp_h20/run.sh all 2>&1 | tee ../k31_run.log   # 约 1 小时
python exp_h20/analyze.py > ../k31_summary.txt
```

结果在 `../k31_h20_out/`。每个阶段都可以单独重跑，例如 `bash exp_h20/run.sh par`，重跑会先删掉这个阶段的旧结果。

| 阶段 | 内容 | 卡数 |
|---|---|---|
| env | GPU、驱动和各个包的版本 | - |
| unit | K3/K3.1 的 CPU 单测，加 `tests/unit_tests/gpu/models/kimi_k3` | 1 |
| ci | H100 集成测试中所有 Kimi 相关的项（real PG）：`kimi_k3_1_fsdp+ep`、`kimi_k3_mm_allgather_kv_cp`、`kimi_k3_mm_ulysses_cp` | 2 |
| par | K3.1 在各种并行组合下的 loss 对比，共 20 步，从同一个 seed checkpoint 开始，dp_shard 都是 2，非确定模式 | 1 到 8 |
| neg | TP2 加 16 个 head（每卡 8 个 head），预期会报出清楚的错误 | 2 |
| k3bisect | K3 debug 模型，1 卡，确定模式，10 步。逐个检查 attn-gym 从 v0.0.19 到 ff06a9221 之间的 22 个 commit，找出第一个让结果和 v0.0.19 不再逐位一致的 commit | 8 卡并行 |

`par` 阶段的两组配置：

- p 组：16 个 head，DistMuon。对比 FSDP2（跑两次，用来估计跑间噪声）、FSDP2+EP2、FSDP2+PP2、FSDP2+EP2+PP2。PP 用 Interleaved1F1B，4 个 microbatch。
- t 组：32 个 head，AdamW。TP 下每卡至少要有 16 个 head，DistMuon 还不支持 TP。对比 FSDP2（跑两次）、FSDP2+EP2、FSDP2+TP2+EP2、FSDP2+TP2+EP2+PP2（8 卡）。

## 3. 需要传回的内容

只需要把 `../k31_summary.txt` 的内容发回来，大约 50 行。如果某一项 FAIL，summary 里已经带了最后一行报错；如果还需要更多信息，我会再告诉你看哪个日志。

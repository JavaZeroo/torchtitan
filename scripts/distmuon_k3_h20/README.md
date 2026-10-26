# K3 debugmodel：单机 8×H20 真实训练验证

这个分支仅用于手动实验，不应合入 PR #5468。它不租卡、不连接远端机器、不自动上传结果，也不修改 GitHub PR/issue。

本轮由 `scripts/distmuon_h20_round4/run.sh` 调用，验证 PR3 与 main 逐位一致。比较固定版本，避免 `main` 或 PR 分支后续变化混入实验：

| 版本 | commit |
|---|---|
| PR 的 main 基线 | `8a3f5bcaa4fd51ea6af6fb4bbb90ef03bcb139d0` |
| PR3：runtime scratch 全用计算精度（BF16） | `1a69066b1bd33ff5722cb80444bc59f906e2a0f2` |

使用官方 `kimi_k3_debugmodel`：保留视觉模块和 KDA/MLA 路径，CC12M 测试数据，序列长度 512，EP8/FSDP8，BF16，seed=42，`deterministic=True`，`deterministic_warn_only=False`。**所有训练梯度来自真实 forward/backward，没有合成梯度模式。** 这些是 debug 规模实验，不是完整 K3 模型的吞吐或收敛证明。

## 1. 取得实验分支

在 H20 机器上执行，建议使用独立 clone：

```bash
git clone --single-branch --branch chore/distmuon-h20-pr3-scale \
  https://github.com/JavaZeroo/torchtitan.git torchtitan-k3-h20
cd torchtitan-k3-h20
```

脚本会在结果目录内建立两个 detached worktree，不会 checkout/reset 你当前的分支，也不会覆盖已有源码修改。

## 2. Python 环境

推荐 Python 3.12 和独立环境：

```bash
bash scripts/distmuon_k3_h20/setup.sh
```

默认创建 `outputs/k3-h20-venv`；目录已存在时拒绝覆盖。也可传入一个新的目录：

```bash
bash scripts/distmuon_k3_h20/setup.sh /your/new/k3-h20-venv
```

推荐版本为此前跑通真实训练的 PyTorch `2.16.0.dev20261016+cu130`、torchvision `0.30.0.dev20261016+cu130`、torchao `0.19.0.dev20261017+cu130`、attn-gym `0.0.17`、FlashAttention-3 `3.0.0`。nightly wheel 将来可能过期；安装失败时保留错误，不会静默改用最新版或替代 kernel。

CUDA 13.x 的常规驱动兼容范围为 R580 或更新，见 [NVIDIA CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)。脚本不安装或升级驱动。已有管理员配置的兼容环境可以直接尝试下方自检。

**已有可用环境可以跳过 setup。** 用该环境的 Python 启动 `run.py` 即可；main 和 PR 全部使用同一个解释器。每轮会记录并核对包版本，缺少 FA3/KDA 等依赖时明确失败，不自动切换模型或计算路径。

## 3. 一条命令运行

确保所选 8 张 H20 没有同时跑其他训练。可用 `CUDA_VISIBLE_DEVICES` 选择卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  outputs/k3-h20-venv/bin/python -m scripts.distmuon_k3_h20.run \
  --output "$PWD/outputs/k3-h20/manual-01"
```

如果已有环境，把上面的 `outputs/k3-h20-venv/bin/python` 换成它的 Python 路径。

执行顺序：

1. 核对源码、数据和 tokenizer，确认实际 H20/SM90、FA3 导入，并执行 NCCL all-to-all 和 fused KDA 前后向自检。
2. **数值验证**：main、PR 各跑 40 步，逐步比较 loss、grad_norm、Muon 参数和 momentum 摘要。不从这两次运行报告性能。
3. 数值检查通过后，执行 **3 轮配对性能实验**：每次 40 步，前 10 步预热，后 30 步计时；顺序为 main/PR、PR/main、main/PR。
4. 校验全部结果、生成汇总，最后自动打包。总计 8 次真实训练。

每个 torchrun 默认最多 900 秒。首次编译较慢时可以加 `--timeout 1800`。Ctrl-C 或超时只停止本次实验的子进程组，不操作机器上的其他任务；已有结果和日志仍会打包。

只查看计划、不使用 GPU、不写结果：

```bash
python3 -m scripts.distmuon_k3_h20.run --dry-run
```

## 4. 指标与数值口径

- DistMuon、完整优化器阶段（含 AdamW）：实际训练中用 CUDA events 测区间，每步取最慢 rank，再取 30 步中位数，最后取三轮中位数。区间包含等待和 CPU 发射间隙，不是 kernel 时间相加。
- 完整训练迭代：从第 10 步结束到第 40 步结束的墙钟窗口，取最慢 rank 后除以 30，再取三轮中位数。包含数据加载、forward/backward、全部优化器、指标记录及每步末尾一次 CUDA 同步。
- 显存：计时步骤内的 PyTorch allocated/reserved 峰值，先取各 rank/步骤最大值，再取三轮中位数。不是整个设备所有进程的 NVML 总占用。
- 性能轮关闭 profiler、TensorBoard 和逐步参数摘要；仅在计时结束后采集最终 Muon 参数/momentum 摘要。数值轮单独做逐步摘要，避免干扰性能测量。
- 所有性能轮的逐步 loss/grad_norm 和最终 Muon 状态，还会再次与数值轮基线核对。缺 rank、缺步骤、版本不符、NaN、数值不一致时不会输出成功的加速结论。

本轮刻意不生成 trace、截图或 HTML。所有三轮原始数据及变化范围都保留，不只选最快的一轮。

## 5. 中断、复算与发回结果

源码、环境、设备选择和机器均未改变时可继续：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  outputs/k3-h20-venv/bin/python -m scripts.distmuon_k3_h20.run \
  --output "$PWD/outputs/k3-h20/manual-01" --resume
```

只跳过已完整校验的运行；失败尝试会重命名保留，不覆盖。改过环境、脚本或设备后请换新结果目录，不要混合比较。用于最终性能结论时，优先保留一次未中断的完整实验。

不使用 GPU，重新校验结果：

```bash
python3 -m scripts.distmuon_k3_h20.analyze outputs/k3-h20/manual-01
```

重新打包已有结果，不启动训练：

```bash
python3 -m scripts.distmuon_k3_h20.run \
  --output "$PWD/outputs/k3-h20/manual-01" --bundle-only
```

运行成功或失败后，发回这两个文件即可：

```text
outputs/k3-h20/manual-01.tar.gz
outputs/k3-h20/manual-01.tar.gz.sha256
```

包里包含源代码指纹、包版本、设备/拓扑信息、逐 rank 结果、训练/自检日志和实验脚本。不会打包 source worktree、venv、缓存、环境变量全集、Git 凭据或 SSH 文件。日志仍建议发送前自行检查。

成功时还会得到 `summary.md`、`summary.json`、`step_metrics.csv`、`numerics.json`。以 `summary.json` 的 `status=validated_real_training` 为成功标记；失败包同样可以发回用于定位问题。

接收端解压后，也可以在 `k3-h20-results` 目录执行 `python3 -m experiment.analyze .` 离线复算，无需 GPU 或训练依赖。

## 本地脚本自检

```bash
python3 -m scripts.distmuon_k3_h20.self_test
```

这只检查结果校验和启动流程的辅助逻辑，不会运行 GPU 训练，也不能代替 H20 实验。

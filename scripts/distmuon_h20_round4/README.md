# 8×H20 第 4 轮：PR3 验证 + 真实形状测量 + profile + GPU 单测

这个分支仅用于手动实验，不合入任何 PR。比较三个固定 commit，main 基线相同：

| 名称 | commit | 内容 |
|---|---|---|
| main | `8a3f5bcaa4fd51ea6af6fb4bbb90ef03bcb139d0` | PR #5468 的 main 基线 |
| pr5468 | `690671216495121a738684abc9a451b76823a123` | PR #5468 当前 head |
| pr3 | `1a69066b1bd33ff5722cb80444bc59f906e2a0f2` | PR3：runtime scratch 全用计算精度（BF16），基于 main |

## 运行

```bash
git clone --single-branch --branch chore/distmuon-h20-pr3-scale \
  https://github.com/JavaZeroo/torchtitan.git torchtitan-h20-round4
cd torchtitan-h20-round4

# 用上一轮的环境即可（Python 3.12，含 FA3）
PY=/path/to/k3-h20-venv/bin/python bash scripts/distmuon_h20_round4/run.sh
```

结果默认写到 `outputs/h20-round4`。中断后用同一条命令重跑，已完成的部分会跳过。只跑其中几部分：`PARTS="B C" PY=... bash scripts/distmuon_h20_round4/run.sh`。

## 四个部分

- **A**：K3 debugmodel 真实训练，main 对 PR3。先 40 步逐步比较 loss、grad_norm、Muon 参数/momentum，再跑 3 组配对测速（同上一轮，8 次训练）。
- **B**：真实形状切片，只跑优化器（合成梯度，12 步，前 3 步预热）：K3（dim 7168、96 头，5 层，32 个 expert，BF16 参数）和 K2.5（3 层，16 个 expert，FP32 参数）。三个版本各跑两轮，第二轮倒序。记录最慢 rank 的优化器耗时、峰值/临时显存、预留 buffer，并比较各版本与 main 的参数/momentum 摘要是否一致。
- **C**：K3 debugmodel 只跑优化器，在第 8 步抓 rank 0 的 profile（main 和 pr5468），看 GPU 忙碌比例和 kernel 数量。
- **D**：`tests/unit_tests/gpu/distributed/flex_shard` 在三个版本上各跑一遍（需要 4 张卡）。

预计总共 1-1.5 小时。

## 发回结果

跑完会打印一段汇总，同时保存在 `outputs/h20-round4/summary.txt`，把这段文字贴回来即可。失败时汇总里也会标出哪部分失败，对应日志在 `outputs/h20-round4/logs/`。

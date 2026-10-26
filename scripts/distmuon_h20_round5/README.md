# 8×H20 第 5 轮：更多规模、并行方式、模型和真实训练

只用于手动实验。比较四个固定 commit：

| 名称 | commit | 内容 |
|---|---|---|
| main | `8a3f5bcaa4fd51ea6af6fb4bbb90ef03bcb139d0` | 基线 |
| pr5468 | `690671216495121a738684abc9a451b76823a123` | #5468 |
| pr3 | `1a69066b1bd33ff5722cb80444bc59f906e2a0f2` | PR3，基于 main |
| both | `571375f1c3738b0e664b5de7dd4c54516530555a` | #5468 + PR3 |

## 运行

```bash
cd torchtitan-k3-h20-round4
git pull
PY=/path/to/k3-h20-venv/bin/python bash scripts/distmuon_h20_round5/run.sh
```

每次运行都会把所选部分从头重跑，并覆盖旧结果。只跑其中几部分：`PARTS="1 3" PY=... bash scripts/distmuon_h20_round5/run.sh`。预计 2.5-3 小时，需要 8 张卡全空。

## 五个部分

1. **expert 数量**：K2.5 切片（3 层）16/64/128/256/384 个 expert，K3 切片（5 层）32/64/128/256/384 个 expert，只跑优化器。384 个 expert 即每卡 48 个，和真实模型一样。
2. **其他并行方式**：4 卡 FSDP4 EP4、8 卡 FSDP8 EP4、8 卡 HSDP 2×4 EP4。
3. **Moonlight 形状**：13 层和 27 层，小矩阵多，主要卡在 CPU 发 kernel。
4. **profile**：K2.5、K3、Moonlight 各抓一步 rank 0 的 trace（main、pr5468、both）。
5. **真实训练**：K3 debugmodel 训练 100 步，K2.5 和 K3 切片各训练 20 步，看优化器占整步的比例，并逐位比较训练后的参数和 momentum。

## 发回结果

跑完会打印汇总，同时保存在 `outputs/h20-round5/summary.txt`，把这段文字贴回来即可。某个实验失败（比如显存不够）不会影响其他实验，汇总里会显示失败原因。

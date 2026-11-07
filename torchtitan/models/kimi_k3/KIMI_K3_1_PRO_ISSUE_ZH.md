# [RFC][Kimi K3.1 Pro] 支持 Gated DSA、continued pre-training 与 SFT

## 背景

Moonshot AI 已发布 Kimi K3.1 Pro 的权重、配置和模型卡。TorchTitan main 已支持 Kimi K3 的模型，现有工作由 #4272 跟踪。

K3.1 Pro 的核心结构变化是：**保留 69 层 KDA，将 24 层 Gated MLA 替换为 Gated DeepSeek Sparse Attention（Gated DSA）**。因此，本 Issue 跟踪在现有 K3 实现上增加 K3.1 Pro 支持。

目标支持：
- 官方 K3.1 Pro 权重的BF16续训与 SFT；
- 支持 K3 -> K3.1 Pro 结构转换与实验性分阶段训练；
- 数值正确性验证。

本 RFC 当前覆盖结构转换、续训和 SFT，是完整训练复现的前置工作；

## 已确认的模型结构

根据 [K3.1 Pro 模型卡](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)、[HF config](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/config.json) 和[权重索引](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/model.safetensors.index.json)，与 K3 的主要差异是：

- 24 层 Gated MLA 替换为 Gated DSA，层位置不变：配置中按 1 起始编号为 4、8、...、92、93；权重索引使用 0 起始编号。
- DSA 保留 K3 MLA 的 Q/KV 投影、norm、output gate 和 output projection。
- 每个 DSA 层新增 `indexer.wq_b`、`indexer.wk`、`indexer.k_norm` 和 `indexer.weights_proj`。
- Indexer 使用 32 个 index heads、128 head dim 和 top-k 2048，key norm 为带 bias 的 LayerNorm。
- 主 DSA 分支和 indexer 均不使用 RoPE。
- 发布配置将 attention（含 indexer）排除在 MXFP4 量化之外；routed experts 延续 K3 的 packed MXFP4 格式。这不说明 indexer 的训练或推理计算精度。

配置中的关键 MLA 投影维度一致，因此 K3 的 MLA 主分支权重可作为结构转换的初始化候选，新增参数主要集中在 indexer；转换后的前向数值仍需验证，这也不证明官方采用了该初始化流程。

## 训练流程的工作假设

官方模型卡称 K3.1 Pro 从 K3 出发进行了覆盖数百个下游领域的大规模联合强化学习；所引模型卡和配置没有披露 DSA 的具体转换和训练过程。

参考 [DeepSeek V3.2 的 DSA continued pre-training 方法](https://arxiv.org/html/2512.02556v1#S2.SS1)，暂作以下**假设**：

1. **模型转换与初始化（非训练）**：继承 K3 主网及 24 层 MLA 主分支，初始化新增的 indexer。
2. **Dense Warm-up Stage**：保持 dense attention，冻结非 indexer 参数，以当前主分支 dense attention 跨 head 聚合并沿序列归一化后的分布为目标，使用 KL(target || indexer) 训练 indexer，无需额外复制 teacher 模型。
3. **Sparse Training Stage**：启用 top-k 并进行整网训练；主网使用 NTP loss，indexer 继续使用选中集合上归一化分布的 KL。按 DS 的梯度隔离设计，detach indexer 输入和 attention target，使 indexer 仅由 KL 更新、主网仅由 NTP loss 更新。
4. **SFT**：P0 直接从官方 K3.1 Pro 权重开始，不经过上述转换和续训；P1 从 Sparse Training checkpoint 开始。两种来源共用同一套 SFT recipe。

## 工作项

### P0：官方权重 SFT 功能支持

P0 以已发布权重和 HF config 为结构依据。当前 HF 仓库未随附模型前向实现；需选定并记录可核查的 K3.1 推理实现及 revision，以对齐前向语义（包括 `index_topk_freq`、`index_skip_topk_offset`），不能仅凭配置字段名推断。SFT 中的 indexer 暂沿用 DS 风格 sparse indexer-loss 作为工程基线，不等待技术报告确认。

- [ ] 落地 K3.1 Gated DSA eager 基线
- [ ] 实现 sparse DSA 前向/反向及 DS 风格 indexer-loss 基础能力
- [ ] 建立数值验证
- [ ] 实现HF checkpoint加载与保存
- [ ] 复用并验证 K3 的 FSDP2、TP/SP、EP、PP/VPP 和 Per-Head DistMuon 支持
- [ ] 建立官方权重BF16 SFT recipe，并验证训练、DCP 恢复和 HF 导出

### P1：支持 K3 -> K3.1 Pro 训练流程

- [ ] 实现模型转换与初始化
  - 复用 MLA 权重、初始化 indexer，并生成 seed checkpoint
- [ ] 实现 Dense Warm-up
  - 从当前主分支提取 dense attention 监督分布，并冻结所有非 indexer 参数
- [ ] 建立 Sparse Training recipe

## 待从技术报告确认的问题

以下问题主要影响 P1 对 K3.1 训练流程的复现，不阻塞 P0：

1. DSA 在哪一个训练阶段引入？
2. 新增 indexer 如何初始化？
3. Dense Warm-up 是否使用 dense-attention KL？
4. Sparse Training 的 indexer KL 和 detach 方式是否与 DS 一致？

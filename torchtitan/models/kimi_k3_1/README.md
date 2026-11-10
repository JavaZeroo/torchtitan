# Kimi K3.1

Kimi K3.1 Pro keeps the Kimi K3 backbone and replaces its 24 Gated MLA layers
with Gated DeepSeek Sparse Attention (DSA) at the same layer positions.

## Prerequisites

Same as [Kimi K3](../kimi_k3/README.md).

## Architecture

Each DSA layer keeps the Kimi K3 MLA projections and output gate, and adds an
indexer with 32 heads of dimension 128 that selects 2048 tokens per query.
Top-k selection and the indexer loss come from
`torchtitan/models/deepseek_v4/compressor.py`. The indexer is trained only by
the indexer loss. `build_model_config(..., indexer_loss_coeff=...)` sets its
coefficient (default 0.01); 0 turns the loss off.

## Supported Parallelisms

Only single-device runs have been tested.

## Numerical Parity

`KimiK31StateDictAdapter` maps the released Hugging Face checkpoint, including
the indexer weights.

No Hugging Face model code is released for K3.1.
`scripts/checkpoint_conversion/numerical_tests_kimi_k3_1.py` compares one
released DSA layer (layer 3, 2304 tokens) with a plain PyTorch version of vLLM's
Kimi K3.1 DSA:

| Precision | Top-k overlap | Output relative L2 |
| --- | --- | --- |
| fp32 | 100% | 1.4e-6 |
| bf16 | 99.97% (lowest row 99.85%) | 8.2e-3 |

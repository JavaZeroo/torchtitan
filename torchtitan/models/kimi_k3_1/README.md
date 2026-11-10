# Kimi K3.1

Kimi K3.1 Pro keeps the Kimi K3 backbone and replaces its 24 Gated MLA layers
with Gated DeepSeek Sparse Attention (DSA) at the same layer positions. See
[RFC #5936](https://github.com/pytorch/torchtitan/issues/5936).

This package owns only what K3.1 adds:

- `dsa.py`: the DSA indexer and absorbed MLA over the selected tokens;
- `model.py`: `KimiK31Model`, which registers the indexer-loss hook;
- `flavors.py`: the `debugmodel` and `Kimi-K3.1-Pro` flavors.

KDA, LatentMoE, the MoonViT-V2 vision encoder, sharding and parallelization
come from `torchtitan/models/kimi_k3/`. Top-k selection (`Indexer.select`) and
the indexer loss (`SparseIndexerLoss`) come from
`torchtitan/models/deepseek_v4/compressor.py`, with compression ratio 1.

## Architecture

Each DSA layer keeps the K3 MLA projections, norms, output gate and output
projection, and adds an indexer (`wq_b`, `wk`, `k_norm`, `weights_proj`):

```
     +--> wq_a --> q_norm --+--> wq_b -------------------> q ----+
     |                      +--> indexer.wq_b ------+            |
     |                                              v            v
x ---+--> indexer.wk --> indexer.k_norm ------> top-k ---> sparse attention
     +--> indexer.weights_proj -------------------^              |
     +--> wkv_a --> kv_norm --> wkv_b ----------------> k, v ----+
     |                                                           v
     +--> gate ---------------------------------------> out * sigmoid(gate) --> wo
```

The indexer has 32 heads of dimension 128 and selects 2048 tokens per query.
Neither the main branch nor the indexer uses RoPE.

## Reference Implementation

There is no Hugging Face forward implementation for K3.1. The forward follows
vLLM at `b65f94500866d7277a0863baff4369888819091a`
(`vllm/models/kimi_k3/nvidia/dsa.py`) with the Hugging Face config at
`d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`: no Hadamard rotation in the
indexer, key LayerNorm in FP32 with `rms_norm_eps` (1e-5), and BF16 head gates.
SGLang's generic DSA indexer uses a Hadamard rotation, eps 1e-6 and FP32 head
gates for the same config.

## Training

The indexer loss is the DeepSeek-V3.2 sparse-stage KL: the main attention
probabilities on the selected tokens, summed over heads and normalized, are
the target for the indexer scores. The probabilities are rebuilt from the
main attention's LSE. The indexer reads detached inputs, so it is trained
only by this loss. `build_model_config(..., indexer_loss_coeff=...)` sets its
weight (default 0.01; 0 disables it).

Sparse attention runs MLA in its absorbed form: each query head absorbs its
key up-projection, all heads attend one shared 576-channel latent key that also
serves as the value, and the first 512 output channels are up-projected per
head. It runs in eager PyTorch and gathers a `[tokens, 2048, 576]` key tensor,
so memory grows linearly with sequence length. Attention Gym `gather_attn` does
not compile in reasonable time for this shape on Hopper. Top-k selection uses
the fused Attention Gym indexer on SM90 or newer GPUs.

## Supported Parallelisms

Not verified yet: only single-device runs have been tested. Hugging Face
checkpoint loading is not implemented yet.

# Attention Gym implementation selection

TorchTitan pins `attn-gym[linear,cudnn]==0.0.17`. These settings cover
Qwen3.5's GDN (also shared by Qwen3.6/3.8), Kimi K3's KDA, and RL/vLLM GDN.
The newer sparse-attention consumers are outside this configuration scope;
dense SDPA and FlexAttention paths are unchanged.

| Operation | Upstream reference support | TorchTitan selection |
| --- | --- | --- |
| `chunk_gdn`, `chunk_kda`, `bound_gate` | `impl="reference"` | Kernel config `impl`; default is `"fused"` |
| Q/K `l2norm` | `kda.naive.l2norm_fwd_ref`, no public `impl` switch | Follows the kernel config |
| Training `causal_conv1d` | No reference switch | Shared PyTorch convolution with SiLU and document resets |
| RL `gate_transform` | `impl="reference"` | Always reference, preserving existing FP32 arithmetic |
| RL GDN paged convolution, `paged_chunk_gdn`, `recurrent_gdn_decode` | No reference switch | Eager cache adapter using reference convolution and upstream dense `recurrent_gdn` |
| Batch-invariant `recurrent_gdn` | Dense reference only; paged `state_indices` requires fused | Reference is rejected during setup |

The cache adapter preserves fresh/resumed sequences, empty sequences and
padding slots. It reuses the upstream recurrence rather than duplicating it.
See the pinned [GDN API](https://github.com/meta-pytorch/attention-gym/blob/v0.0.17/attn_gym/linear/gdn/api.py),
[KDA API](https://github.com/meta-pytorch/attention-gym/blob/v0.0.17/attn_gym/linear/kda/api.py),
[normalization reference](https://github.com/meta-pytorch/attention-gym/blob/v0.0.17/attn_gym/linear/kda/naive.py)
and [short convolution](https://github.com/meta-pytorch/attention-gym/blob/v0.0.17/attn_gym/linear/short_conv/cute.py).

## Configuration

Set the existing model config tree directly in a recipe:

```python
from torchtitan.models.common.attention_gym import AttentionGymConfig
from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel


def qwen35_reference():
    config = qwen35_debugmodel()
    config.training.disable_cuda_graphs = True
    for _, kernel, _, _ in config.model.traverse(AttentionGymConfig):
        kernel.impl = "reference"
    return config
```

For individual GDN/KDA layers, set `layer.attention.inner_attention.kernel.impl`.
The selection
also controls convolution and normalization. Parameter and checkpoint layouts
are unchanged.

Reference execution is for eager correctness checks on CPU or CUDA. Packed
execution reads document offsets on the host. Training configuration validation
rejects CUDA Graphs, context parallelism and batch-invariant mode before model
construction. Existing tensor-only `local_compile_regions` remain supported;
compiling the packed reference loops as a full graph is unsupported. Fused/reference
comparisons require floating-point tolerances. Default fused execution preserves
the upstream automatic cuDNN selection and autotuning policy.

The RL model replacement preserves this selection. Reference generation requires
`generator.cuda_graph.mode="NONE"` (`enforce_eager=True` in vLLM) and
batch-invariant mode disabled. The adapter checks both during construction.
The upstream vLLM KDA adapter has no reference implementation and rejects
reference selection during replacement. Kimi's head dimension remains 128; its
fused path uses the hardware support of the pinned Attention Gym release.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Is a batched Newton-Schulz bitwise invariant to how the batch is chunked?

Runs the DistMuon kernel on a [B, R, C] batch as one call and as chunks of
several sizes, and reports which chunkings reproduce the full-batch result
bit for bit. Chunking is only safe for a memory budget where this holds.
"""
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from torchtitan.distributed.flex_shard.dist_muon import (  # noqa: E402
    _zeropower_via_newtonschulz as ns,
)

COEF = (3.4445, -4.7750, 2.0315)
dev = torch.device("cuda")
torch.manual_seed(0)
print(torch.cuda.get_device_name(), torch.__version__)
shapes = [
    ("moonlight expert w13 [16,1408,2048]", (16, 1408, 2048)),
    ("moonlight expert w2  [16,2048,1408]", (16, 2048, 1408)),
    ("k2.5 expert w13      [8,2048,7168]", (8, 2048, 7168)),
    ("k2.5 expert w2       [8,7168,2048]", (8, 7168, 2048)),
    ("k2.5 wq_b per-head   [64,128,1536]", (64, 128, 1536)),
    ("moonlight wq per-head [16,128,2048]", (16, 128, 2048)),
    ("dense w13 chunks     [2,11264,2048]", (2, 11264, 2048)),
]
for name, shape in shapes:
    x = torch.randn(shape, device=dev, dtype=torch.bfloat16)
    full = ns(x.clone(), ns_coefficients=COEF, ns_steps=5, eps=1e-7)
    results: list[str] = []
    for chunk in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        if chunk >= shape[0]:
            continue
        parts = [
            ns(x[i : i + chunk].clone(), ns_coefficients=COEF, ns_steps=5, eps=1e-7)
            for i in range(0, shape[0], chunk)
        ]
        out = torch.cat(parts)
        same = torch.equal(out, full)
        diff = (out.float() - full.float()).abs().max().item()
        results.append(f"chunk={chunk}: {'bitwise' if same else f'maxdiff={diff:.2e}'}")
    print(f"{name:<40s} " + " | ".join(results), flush=True)
    del x, full

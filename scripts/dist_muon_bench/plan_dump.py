# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump DistMuon's static plan: buckets, layouts, owners, loads, buffers.

Run as one logical rank under the fake backend, e.g.

  NGPU=8 LOCAL_RANK=0 python plan_dump.py --module bench_configs \
      --config moonlight_slice --comm.backend=fake --mb.out=/path/plan.json

Every rank builds the same plan, so one logical rank can report the compute
load and traffic of all ranks in its transport groups.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402  # pyrefly: ignore [missing-import]
    build_engine,
    dist_muon_optimizers,
    dump_json,
    harness_args,
    ns_flops,
)


def _views_flops(shape, compute_sharding, row_start, ns_steps):
    from torchtitan.distributed.flex_shard import BlockShard
    from torchtitan.distributed.flex_shard.dist_muon import (
        _matrix_batch_views_from_shape,
    )

    if not math.prod(shape):
        return 0, []
    if type(compute_sharding) is BlockShard:
        views = _matrix_batch_views_from_shape(
            torch.Size(shape),
            matrix_row_sizes=compute_sharding.block_sizes,
            logical_row_start=row_start,
        )
        shapes = [tuple(view.shape) for view in views]
    else:
        shapes = [tuple(shape)]
    return sum(ns_flops(s, ns_steps) for s in shapes), shapes


def _reserved(buffers) -> dict:
    out = {}
    for (device, dtype), reserved in buffers.items():
        for name in (
            "storage_exchange",
            "compute_exchange",
            "compute_scratch",
            "storage_scratch",
        ):
            tensor = getattr(reserved, name, None)
            if tensor is not None:
                key = f"{name}[{str(dtype).replace('torch.', '')}]"
                out[key] = out.get(key, 0) + tensor.numel() * tensor.element_size()
    return out


def main() -> None:
    options, argv = harness_args()
    config, engine = build_engine(argv)
    from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
        _device_mesh_ranks,
        _dtensor_storage_region_for_participant,
        _LocalBucketPlan,
    )

    report = {"optimizers": []}
    for optimizer in dist_muon_optimizers(engine):
        ns_steps = optimizer.param_groups[0]["ns_steps"]
        buckets = []
        load_by_rank = defaultdict(int)
        sent_by_rank = defaultdict(int)
        state_bytes = 0
        for bucket_index, plan in enumerate(optimizer._bucket_plans):
            is_local = isinstance(plan, _LocalBucketPlan)
            bucket: dict[str, Any] = {
                "index": bucket_index,
                "kind": "local" if is_local else "redistribution",
                "items": [],
                "load_by_rank": defaultdict(int),
                "sent_bytes_by_rank": defaultdict(int),
            }
            local_items = plan.items if is_local else plan.unredistributed_items
            redistributed = () if is_local else plan.redistributed_items
            redistribution_plans = () if is_local else plan.redistribution_plans

            for item in local_items:
                param = item.param
                element_size = param.element_size()
                state_bytes += param.to_local().numel() * element_size
                entry = _item_entry(item, redistributed=False)
                loads = {}
                for rank in _device_mesh_ranks(param.device_mesh):
                    try:
                        region = _dtensor_storage_region_for_participant(param, rank)
                    except RuntimeError:
                        # Compute-ready strided storage (FSDP over the expert
                        # dim): every rank holds the same number of experts,
                        # so the local shape stands in for each rank.
                        local = param.to_local()
                        flops, shapes = _views_flops(
                            tuple(local.shape), item.compute_sharding, 0, ns_steps
                        )
                        loads[rank] = flops
                        continue
                    flops, shapes = _views_flops(
                        region.shape,
                        item.compute_sharding,
                        region.offsets[0],
                        ns_steps,
                    )
                    loads[rank] = flops
                entry["ns_flops_by_rank"] = loads
                for rank, flops in loads.items():
                    bucket["load_by_rank"][rank] += flops
                bucket["items"].append(entry)

            for item, redistribution_plan in zip(
                redistributed, redistribution_plans, strict=True
            ):
                param = item.param
                element_size = param.element_size()
                state_bytes += param.to_local().numel() * element_size
                entry = _item_entry(item, redistributed=True)
                loads = {}
                compute_shapes = {}
                for partition in redistribution_plan.compute_partitions:
                    row_start = (
                        partition.logical_regions[0].offsets[0]
                        if partition.logical_regions
                        else 0
                    )
                    flops, shapes = _views_flops(
                        partition.tensor_shape,
                        item.compute_sharding,
                        row_start,
                        ns_steps,
                    )
                    loads[partition.participant] = flops
                    if math.prod(partition.tensor_shape):
                        compute_shapes[partition.participant] = list(
                            partition.tensor_shape
                        )
                entry["ns_flops_by_rank"] = loads
                entry["compute_shape_by_rank"] = compute_shapes
                traffic = defaultdict(int)
                for route in redistribution_plan.storage_to_compute_routes:
                    sources = route.source.participants
                    for destination in route.destination.participants:
                        source = destination if destination in sources else sources[0]
                        if source != destination:
                            traffic[source] += route.logical_region.numel * element_size
                entry["gather_bytes_sent_by_rank"] = dict(traffic)
                for rank, flops in loads.items():
                    bucket["load_by_rank"][rank] += flops
                for rank, nbytes in traffic.items():
                    bucket["sent_bytes_by_rank"][rank] += nbytes
                bucket["items"].append(entry)

            if not is_local:
                bucket["participants"] = list(plan.group.participants)
                bucket["local_gather_send_bytes"] = (
                    plan.storage_to_compute_schedule.input_buffer_numel
                    * torch.empty((), dtype=plan.dtype).element_size()
                )
                bucket["local_gather_recv_bytes"] = (
                    plan.storage_to_compute_schedule.output_buffer_numel
                    * torch.empty((), dtype=plan.dtype).element_size()
                )
            for rank, flops in bucket["load_by_rank"].items():
                load_by_rank[rank] += flops
            for rank, nbytes in bucket["sent_bytes_by_rank"].items():
                sent_by_rank[rank] += nbytes
            bucket["load_by_rank"] = dict(bucket["load_by_rank"])
            bucket["sent_bytes_by_rank"] = dict(bucket["sent_bytes_by_rank"])
            buckets.append(bucket)

        runtime = optimizer._redistribution_runtime
        reserved = {"local_slot": _reserved(runtime._local_slot.buffers)}
        if runtime._context is not None:
            for index, slot in enumerate(runtime._context.slots):
                reserved[f"pipeline_slot_{index}"] = _reserved(slot.buffers.buffers)
        reserved_total = sum(sum(group.values()) for group in reserved.values())
        report["optimizers"].append(
            {
                "num_params": len(optimizer._parameter_compute_layouts),
                "ns_steps": ns_steps,
                "buckets": buckets,
                "ns_flops_by_rank": dict(load_by_rank),
                "gather_bytes_sent_by_rank": dict(sent_by_rank),
                "reserved_buffers": reserved,
                "reserved_buffers_total_bytes": reserved_total,
                "local_param_bytes": state_bytes,
            }
        )
        _print_summary(report["optimizers"][-1])

    if "out" in options:
        dump_json(options["out"], report)
        print(f"plan written to {options['out']}")


def _item_entry(item, *, redistributed: bool) -> dict:
    param = item.param
    return {
        "fqn": item.fqn,
        "shape": list(param.shape),
        "local_shape": list(param.to_local().shape),
        "mesh_axes": list(param.device_mesh.mesh_dim_names or ()),
        "mesh_shape": list(param.device_mesh.shape),
        "storage_placements": [str(p) for p in param.placements],
        "compute_sharding": repr(item.compute_sharding),
        "redistributed": redistributed,
        "dtype": str(param.dtype),
    }


def _print_summary(opt: dict) -> None:
    buckets = opt["buckets"]
    num_local = sum(bucket["kind"] == "local" for bucket in buckets)
    print(
        f"\nDistMuon: {opt['num_params']} params, {len(buckets)} buckets "
        f"({len(buckets) - num_local} redistribution, {num_local} local)"
    )
    for bucket in buckets:
        kinds: dict[tuple[str, str], int] = defaultdict(int)
        for item in bucket["items"]:
            tag = "redistributed" if item["redistributed"] else "compute-ready"
            kinds[(item["compute_sharding"].split("(")[0], tag)] += 1
        loads = bucket["load_by_rank"]
        load_text = " ".join(
            f"r{rank}={flops / 1e12:.2f}" for rank, flops in sorted(loads.items())
        )
        print(
            f"  bucket {bucket['index']:2d} {bucket['kind']:<14s} "
            f"{dict(kinds)}\n      NS TFLOP by rank: {load_text}"
        )
    loads = opt["ns_flops_by_rank"]
    if loads:
        mean = sum(loads.values()) / len(loads)
        worst = max(loads.values())
        print(
            "  total NS TFLOP by rank: "
            + " ".join(f"r{r}={v / 1e12:.2f}" for r, v in sorted(loads.items()))
        )
        print(
            f"  critical path / mean = {worst / mean:.3f} "
            f"(max {worst / 1e12:.2f}, mean {mean / 1e12:.2f} TFLOP)"
        )
    sent = opt["gather_bytes_sent_by_rank"]
    if sent:
        print(
            "  gather MiB sent by rank (one direction): "
            + " ".join(f"r{r}={v / 2**20:.1f}" for r, v in sorted(sent.items()))
        )
    print(
        f"  reserved buffers on this rank: "
        f"{opt['reserved_buffers_total_bytes'] / 2**20:.1f} MiB; "
        f"local Muon params: {opt['local_param_bytes'] / 2**20:.1f} MiB"
    )
    for slot, groups in opt["reserved_buffers"].items():
        text = ", ".join(f"{k}={v / 2**20:.1f}MiB" for k, v in groups.items())
        print(f"    {slot}: {text}")


if __name__ == "__main__":
    main()

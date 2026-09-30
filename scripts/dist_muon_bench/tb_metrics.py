# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extract full-precision loss and grad_norm from a TorchTitan TensorBoard dir."""
import json
import os
import sys


def read(tb_root):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    subdirs = sorted(
        d for d in os.listdir(tb_root) if os.path.isdir(os.path.join(tb_root, d))
    )
    if not subdirs:
        raise SystemExit(f"no TensorBoard run under {tb_root}")
    acc = EventAccumulator(os.path.join(tb_root, subdirs[-1]))
    acc.Reload()
    out = {}
    for name, tag in (
        ("loss", "loss_metrics/global_avg_loss"),
        ("grad_norm", "grad_norm"),
    ):
        scalar_tags = acc.Tags().get("scalars")
        if isinstance(scalar_tags, list) and tag in scalar_tags:
            out[name] = {str(s.step): s.value for s in acc.Scalars(tag)}
    return out


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        print(json.dumps(read(sys.argv[2]), indent=1))
        return
    base, test = read(sys.argv[1]), read(sys.argv[2])
    worst = 0.0
    identical = True
    print(
        f"{'step':>5s} {'base loss':>20s} {'test loss':>20s} {'diff':>12s}   "
        f"{'base grad_norm':>18s} {'test grad_norm':>18s} {'diff':>12s}"
    )
    for step in sorted(base["loss"], key=int):
        bl, tl = base["loss"][step], test["loss"].get(step)
        bg, tg = base.get("grad_norm", {}).get(step), test.get("grad_norm", {}).get(
            step
        )
        if tl is None:
            print(f"{step:>5s} missing in test")
            identical = False
            continue
        dl = tl - bl
        dg = (tg - bg) if bg is not None and tg is not None else float("nan")
        identical &= bl == tl and bg == tg
        worst = max(worst, abs(dl))
        print(
            f"{step:>5s} {bl:>20.10f} {tl:>20.10f} {dl:>12.3e}   {bg:>18.10f} {tg:>18.10f} {dg:>12.3e}"
        )
    print(f"bitwise identical metrics: {identical}; max |loss diff| = {worst:.3e}")
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()

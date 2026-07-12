#!/usr/bin/env python3
"""Model soup: weight-average two merged composite checkpoints.

The quick-specialist (blend) and medium-specialist (apex) occupy opposite
ends of the tier-Pareto frontier (C54). Data-interpolation between their
recipes (apex60) was strictly dominated, but weight-space averaging of the
two trained models is a different operation (model soups / task arithmetic)
that can land at a point dominating the data-interpolation hull — and it is a
single weight-set. This averages every tensor: out = a*A + (1-a)*B.
Run under the repo .venv.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True, help="checkpoint A dir")
    ap.add_argument("--b", type=Path, required=True, help="checkpoint B dir")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on A")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # copy config/tokenizer/etc from A; overwrite weights with the soup
    for item in args.a.iterdir():
        if item.suffix != ".safetensors" and item.name != "model.safetensors.index.json":
            if item.is_file():
                shutil.copy2(item, args.out / item.name)

    a_shards = sorted(args.a.glob("*.safetensors"))
    b_map = {}  # tensor name -> (shard path)
    for shard in sorted(args.b.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                b_map[k] = shard

    index = {"weight_map": {}, "metadata": {}}
    for shard in a_shards:
        with safe_open(str(shard), "pt") as fa:
            keys = list(fa.keys())
            out_tensors = {}
            for k in keys:
                ta = fa.get_tensor(k)
                bshard = b_map.get(k)
                if bshard is None:
                    out_tensors[k] = ta
                    continue
                with safe_open(str(bshard), "pt") as fb:
                    tb = fb.get_tensor(k)
                if ta.shape != tb.shape:
                    out_tensors[k] = ta
                    continue
                soup = (args.alpha * ta.float() + (1 - args.alpha) * tb.float()).to(ta.dtype)
                out_tensors[k] = soup
            out_name = shard.name
            save_file(out_tensors, str(args.out / out_name), metadata={"format": "pt"})
            for k in keys:
                index["weight_map"][k] = out_name
        print(f"[soup] {shard.name}: {len(keys)} tensors averaged (alpha={args.alpha})", flush=True)

    idx_src = args.a / "model.safetensors.index.json"
    if idx_src.exists():
        shutil.copy2(idx_src, args.out / "model.safetensors.index.json")
    print(f"[soup] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

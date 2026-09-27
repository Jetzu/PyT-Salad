"""Time the PyTorch denoiser. Run with any torch env.

    python scripts/bench_torch.py [--device cuda] [--lengths 100,200,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402

PARAMS = "/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


CONFIG = DotDict(
    augment_size=15, encoder_depth=2, local_size=128, pair_size=64,
    num_neighbours=48, resi_dual=False, factor=4, heads=8, key_size=32,
    multi_query=False, diffusion_depth=6, block_size=1,
    diffusion_kind="vpfixed", sigma_data=10.0, preconditioning="vp",
    diffusion_time_scale="cosine", diffuse_sequence=False,
    aa_decoder_kind="adm", aa_decoder_depth=3, time_embedding=True,
    local_neighbours=16, fape_neighbours=64,
)


def make_data(length, device, atoms=20):
    g = torch.Generator(device="cpu").manual_seed(0)
    t = torch.arange(length, dtype=torch.float32)[:, None]
    bb = torch.cat([2.3 * torch.cos(t * 1.7), 2.3 * torch.sin(t * 1.7), 1.5 * t], -1)
    pos = bb[:, None, :] + torch.randn(length, atoms, 3, generator=g) * 1.2
    z = torch.zeros(length, dtype=torch.long)
    data = dict(
        pos=pos, pos_noised=pos.clone(),
        aa_gt=torch.full((length,), 20, dtype=torch.long),
        seq=torch.full((length,), 20, dtype=torch.long),
        seq_noised=torch.full((length,), 20, dtype=torch.long),
        residue_index=torch.arange(length), chain_index=z, batch_index=z,
        mask=torch.ones(length, dtype=torch.bool),
        cyclic_mask=torch.zeros(length, dtype=torch.bool),
        t_pos=torch.full((length,), 0.5), t_seq=torch.ones(length),
    )
    prev = {"pos": torch.zeros_like(pos),
            "local": torch.zeros(length, CONFIG.local_size)}
    return ({k: v.to(device) for k, v in data.items()},
            {k: v.to(device) for k, v in prev.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lengths", default="50,100,200,400")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--distogram", action="store_true",
                    help="also run the (unused at inference) distogram head")
    ap.add_argument("--compile", default="",
                    help="torch.compile mode: default/reduce-overhead/max-autotune")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = StructureDiffusionPredict(CONFIG).eval().to(device)
    load_model(model, load_param_npz(PARAMS), CONFIG, strict=True)
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    if args.compile:
        # The model is launch-bound in eager mode (~75 ms regardless of L), so
        # compilation is where the win is. Compile the DiffusionBlocks rather
        # than the whole model: the outer wrappers shuffle python dicts, which
        # Dynamo does not trace faithfully (it drops the 'condition' key).
        blocks = model.diffusion.diffusion_stack.blocks
        for i in range(len(blocks)):
            blocks[i] = torch.compile(blocks[i], mode=args.compile, dynamic=False)
        model.diffusion.aa_decoder.aa_decoder_stack.blocks = torch.nn.ModuleList([
            torch.compile(b, mode=args.compile, dynamic=False)
            for b in model.diffusion.aa_decoder.aa_decoder_stack.blocks])
        print(f"torch.compile(mode={args.compile!r}) on {len(blocks)} diffusion "
              f"blocks + the aa-decoder blocks")

    results = {}
    with torch.no_grad():
        for length in [int(x) for x in args.lengths.split(",")]:
            data, prev = make_data(length, device)
            for _ in range(args.warmup):
                model(data, prev, compute_distogram=args.distogram)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                model(data, prev, compute_distogram=args.distogram)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            ms = np.array(times) * 1e3
            results[length] = dict(median_ms=float(np.median(ms)),
                                   min_ms=float(ms.min()),
                                   p90_ms=float(np.percentile(ms, 90)))
            print(f"L={length:<5} median {np.median(ms):8.2f} ms   "
                  f"min {ms.min():8.2f} ms   "
                  f"-> 500-step design {np.median(ms) * 500 / 1000:7.1f} s")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"backend": "torch", "device": str(device),
                       "distogram": args.distogram, "results": results}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

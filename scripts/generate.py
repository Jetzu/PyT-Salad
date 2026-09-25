"""De-novo protein generation - the real entry point.

    python scripts/generate.py --num_aa 100 --num_designs 10 --out_path out/

Mirrors the CLI surface of upstream's
`salad/training/eval_noise_schedule_benchmark.py` for the unconditional case.
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

from salad_torch.config import get_config  # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.sampling import SamplerConfig, make_initial_data, sample  # noqa: E402
from salad_torch.io.pdb import write_pdb  # noqa: E402
from salad_torch.modules.model import StructureDiffusionInference  # noqa: E402


def parse_num_aa(spec: str):
    """`"100"` or `"100:50"` (colon-separated chains) -> (resi, chain).

    Chains restart their residue numbering and are offset by +50, matching
    upstream's `parse_num_aa`.
    """
    lengths = [int(x) for x in str(spec).split(":")]
    resi, chain, offset = [], [], 0
    for idx, n in enumerate(lengths):
        resi.append(np.arange(n) + offset)
        chain.append(np.full(n, idx))
        offset += n + 50
    return (torch.from_numpy(np.concatenate(resi)).long(),
            torch.from_numpy(np.concatenate(chain)).long())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default_vp")
    ap.add_argument("--params", default="")
    ap.add_argument("--num_aa", default="100")
    ap.add_argument("--num_designs", type=int, default=1)
    ap.add_argument("--num_steps", type=int, default=500)
    ap.add_argument("--out_steps", default="400")
    ap.add_argument("--prev_threshold", type=float, default=0.8)
    ap.add_argument("--timescale_pos", default="cosine(t)")
    ap.add_argument("--cloud_std", default="")
    ap.add_argument("--out_path", default="outputs/designs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timing_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    config = get_config(args.config)
    params = args.params or f"/hnvme/workspace/v181bb10-salad/{args.config}-200k.npz"

    model = StructureDiffusionInference(config).eval().to(device)
    load_model(model.predict, load_param_npz(params), config, strict=True)
    if args.compile:
        blocks = model.predict.diffusion.diffusion_stack.blocks
        for i in range(len(blocks)):
            blocks[i] = torch.compile(blocks[i], mode=args.compile, dynamic=False)

    resi, chain = parse_num_aa(args.num_aa)
    num_aa = resi.shape[0]
    cfg = SamplerConfig(num_steps=args.num_steps,
                        out_steps=tuple(int(x) for x in args.out_steps.split(",")),
                        prev_threshold=args.prev_threshold,
                        timescale_pos=args.timescale_pos)
    os.makedirs(args.out_path, exist_ok=True)
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"config {args.config}  L={num_aa}  steps={args.num_steps} "
          f"out_steps={cfg.out_steps}")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    stats, times = [], []

    for design in range(args.num_designs):
        data, prev = make_initial_data(num_aa, resi, chain)
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
        prev = {k: v.to(device) for k, v in prev.items()}
        if args.cloud_std:
            data["cloud_std"] = torch.full((num_aa,), float(args.cloud_std), device=device)

        def step_fn(d, p, generator=None):
            with torch.no_grad():
                return model(d, p, generator=gen)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        snaps = sample(step_fn, data, prev, cfg, generator=gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        snap = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in snaps[-1].items()}
        path = os.path.join(args.out_path, f"design_{design}.pdb")
        write_pdb(path, snap)

        ca = snap["pos"][:, 1].numpy()
        d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
        rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
        stats.append(dict(ca_ca_mean=float(d.mean()), ca_ca_min=float(d.min()),
                          ca_ca_max=float(d.max()), rg=rg, seconds=elapsed))
        print(f"  design {design}: {elapsed:6.1f}s  CA-CA {d.mean():.3f} A "
              f"(min {d.min():.3f} max {d.max():.3f})  Rg {rg:.1f} A  -> {path}")

    t = np.array(times)
    print(f"\n{args.num_designs} designs of {num_aa} aa: "
          f"median {np.median(t):.1f}s, total {t.sum():.1f}s")
    if args.timing_json:
        with open(args.timing_json, "w") as fh:
            json.dump(dict(backend="torch", config=args.config, num_aa=num_aa,
                           num_steps=args.num_steps, device=str(device),
                           compile=args.compile, stats=stats), fh, indent=2)


if __name__ == "__main__":
    main()

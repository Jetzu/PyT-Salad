"""Where does the denoiser's time actually go?

    python scripts/profile_model.py --device cuda --length 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch.config import get_config  # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.geometry.neighbours import extract_neighbours  # noqa: E402
from salad_torch.modules.blocks import extract_condition_neighbours  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--config", default="default_vp")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = get_config(args.config)
    model = StructureDiffusionPredict(cfg).eval().to(device)
    load_model(model, load_param_npz(
        f"/hnvme/workspace/v181bb10-salad/{args.config}-200k.npz"), cfg, strict=True)

    L, A = args.length, 5 + cfg.augment_size
    z = torch.zeros(L, dtype=torch.long, device=device)
    pos = (torch.randn(L, A, 3, device=device) * 4)
    data = dict(pos=pos, pos_noised=pos, aa_gt=torch.full((L,), 20, dtype=torch.long, device=device),
                seq=torch.full((L,), 20, dtype=torch.long, device=device),
                seq_noised=torch.full((L,), 20, dtype=torch.long, device=device),
                residue_index=torch.arange(L, device=device), chain_index=z, batch_index=z,
                mask=torch.ones(L, dtype=torch.bool, device=device),
                cyclic_mask=torch.zeros(L, dtype=torch.bool, device=device),
                t_pos=torch.full((L,), 0.5, device=device), t_seq=torch.ones(L, device=device))
    prev = {"pos": torch.zeros(L, A, 3, device=device),
            "local": torch.zeros(L, cfg.local_size, device=device)}

    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}   L={L}")

    with torch.no_grad():
        full = timeit(lambda: model(data, prev, compute_distogram=False), device)
        full_d = timeit(lambda: model(data, prev, compute_distogram=True), device)

        cond = timeit(lambda: model.prepare_condition(data), device)

        pc = model.prepare_condition(data)
        d2 = dict(data); d2.update(pc)
        dif = model.diffusion

        nb_cur = timeit(lambda: extract_neighbours(16, 16, 32)(
            pos, data["residue_index"], z, z, data["mask"]), device)
        nb_cond = timeit(lambda: extract_condition_neighbours(16, 16, 16, 16)(
            prev["pos"], pc["pair_condition"], data["residue_index"], z, z, data["mask"]), device)

        feats = timeit(lambda: dif.prepare_features(d2, prev), device)

        f = dif.prepare_features(d2, prev)
        local, p, pp, condition, paircond, t_pos, t_seq, resi, chain, batch, mask = f
        cn = extract_neighbours(16, 16, 32)(p, resi, chain, batch, mask)
        dn = extract_condition_neighbours(16, 16, 16, 16)(pp, paircond, resi, chain, batch, mask)
        blk = dif.diffusion_stack.blocks[0]
        one_block = timeit(lambda: blk(local, p, pp, condition, paircond, t_pos,
                                       resi, chain, batch, mask, cn, dn), device)

        aa = timeit(lambda: dif.aa_decoder.train_forward(
            data["aa_gt"], local, p, resi, chain, batch, mask), device)
        ang = timeit(lambda: dif.angle_head(data["aa_gt"], local, p), device)

    depth = cfg.diffusion_depth
    rows = [
        ("FULL forward (no distogram)", full, 100.0),
        ("FULL forward (with distogram)", full_d, 100 * full_d / full),
        ("-- prepare_condition", cond, 100 * cond / full),
        ("-- prepare_features", feats, 100 * feats / full),
        (f"-- neighbours: current x{depth}", nb_cur * depth, 100 * nb_cur * depth / full),
        (f"-- neighbours: condition x{depth}", nb_cond * depth, 100 * nb_cond * depth / full),
        (f"-- DiffusionBlock x{depth} (no nbrs)", one_block * depth, 100 * one_block * depth / full),
        ("-- aa_decoder", aa, 100 * aa / full),
        ("-- angle head", ang, 100 * ang / full),
    ]
    print(f"\n{'component':<36}{'ms':>9}{'% of fwd':>10}")
    for name, ms, pct in rows:
        print(f"{name:<36}{ms:>9.2f}{pct:>9.1f}%")
    nb_total = (nb_cur + nb_cond) * depth
    print(f"\nneighbour selection alone: {nb_total:.2f} ms = {100 * nb_total / full:.1f}% of the forward")
    print(f"500-step design: {full * 500 / 1000:.1f} s   (JAX reference: ~1.6 s)")


if __name__ == "__main__":
    main()

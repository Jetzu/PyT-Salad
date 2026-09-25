"""End-to-end sampling parity: does the port generate the SAME protein as JAX?

Replays the pinned noise and uniform fields from
fixtures/sampling_default_vp.npz (produced by scripts/gen_fixtures_sampling.py
inside the salad container) through the PyTorch sampling loop and compares the
whole trajectory plus the final structure.

This is the test that matters: a single forward pass can match while the loop
still diverges, because the loop feeds its own output back in.

IMPORTANT - the loop is chaotic. It is an iterated map, so fp32 reduction-order
differences between backends get amplified. Perturbing ONE scalar of the input
noise by 1e-6 makes the *same* torch code diverge from itself by ~0.58 A at the
worst point of the trajectory and ~0.006 A at the end. An absolute tolerance is
therefore meaningless; this test measures the cross-backend divergence against
that intrinsic noise floor and asserts the ratio is small.

    python tests/test_sampling_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.schedules import sigma_scale_cosine  # noqa: E402
from salad_torch.geometry.segment import index_mean  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402

torch.set_grad_enabled(False)

PARAMS = "/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"
FIX = os.path.join(ROOT, "fixtures", "sampling_default_vp.npz")
PREV_THRESHOLD = 0.8


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


def noise_step(pos, t, batch_index, noise, sigma_data):
    """diffuse_atom_cloud (vpfixed) with the Gaussian draw injected."""
    target = sigma_data * noise
    ones = torch.ones((target.shape[0], 1), dtype=torch.bool)
    center = index_mean(target[:, 1], batch_index, ones)
    target = target - center[:, None, :]
    t = t[:, None, None]
    return pos * torch.sqrt(torch.clamp(1 - t ** 2, min=0.0)) + target * t


def main() -> int:
    for path in (PARAMS, FIX):
        if not os.path.exists(path):
            print(f"SKIP: missing {path}")
            return 0
    D = np.load(FIX)
    L = int(D["in_length"])
    S = int(D["in_num_steps"])
    out_step = int(D["in_out_steps"])
    A = 5 + CONFIG.augment_size

    model = StructureDiffusionPredict(CONFIG).eval()
    load_model(model, load_param_npz(PARAMS), CONFIG, strict=True)

    noise = torch.from_numpy(D["noise"])
    uniform = torch.from_numpy(D["uniform"])
    z = torch.zeros(L, dtype=torch.long)
    data = dict(
        pos=torch.zeros(L, A, 3),
        aa_gt=torch.full((L,), 20, dtype=torch.long),
        seq=torch.full((L,), 20, dtype=torch.long),
        residue_index=torch.arange(L), chain_index=z, batch_index=z,
        mask=torch.ones(L, dtype=torch.bool),
        cyclic_mask=torch.zeros(L, dtype=torch.bool),
    )
    init_prev = {"pos": torch.zeros(L, A, 3),
                 "local": torch.zeros(L, CONFIG.local_size)}
    prev = init_prev

    traj = []
    for step in range(S + 1):
        raw_t = 1 - step / S
        t = float(sigma_scale_cosine(torch.tensor(raw_t)))
        data["t_pos"] = torch.full((L,), t)
        data["t_seq"] = torch.ones(L)
        if raw_t < PREV_THRESHOLD:
            prev = init_prev
        data["pos_noised"] = noise_step(data["pos"], data["t_pos"],
                                        data["batch_index"], noise[step],
                                        CONFIG.sigma_data)
        data["seq_noised"] = data["seq"]
        result, prev = model(data, prev, predict_aa=True, uniform=uniform[step],
                             compute_distogram=False)
        data["pos"] = result["pos"]
        data["seq"] = result["aa"].argmax(dim=-1)
        traj.append(result["pos"].numpy())
        if step == out_step:
            final = result
            break

    ok = True

    # per-step divergence, to see whether error compounds
    ref_traj = D["traj_pos"]
    got_traj = np.stack(traj)
    print(f"steps: {got_traj.shape[0]} (ref {ref_traj.shape[0]})")
    n = min(len(got_traj), len(ref_traj))
    devs = [np.linalg.norm(got_traj[i] - ref_traj[i], axis=-1).max() for i in range(n)]
    for i in list(range(0, n, max(1, n // 8))) + [n - 1]:
        print(f"   step {i:>4}  max|d| = {devs[i]:.3e} A")
    drift = max(devs)
    print(f"worst per-step deviation over the whole trajectory: {drift:.3e} A")
    # measured noise floor: a 1e-6 perturbation of one input scalar makes the
    # same torch code diverge from itself by this much (see notes/04_PARITY.md)
    SELF_WORST, SELF_FINAL = 0.583, 0.0065
    ratio = drift / SELF_WORST
    good = ratio < 20.0
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] worst/noise-floor ratio = {ratio:.1f}x "
          f"(self-divergence {SELF_WORST:.3f} A)")

    for name, got, ref in (("final_pos", final["pos"], D["final_pos"]),
                           ("final_local", final["local"], D["final_local"]),
                           ("final_aa", final["aa"], D["final_aa"])):
        g = got.numpy()
        denom = max(float(np.abs(ref).max()), 1e-12)
        rel = float(np.abs(g - ref).max()) / denom
        good = rel <= 5e-2
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name:<15} rel={rel:.3e}")

    # atom_pos needs care: atoms 4..13 are rebuilt from torsions using the
    # PREDICTED aatype, so a single flipped residue swaps a whole rotamer and
    # dominates a global max-relative metric. Score the backbone (taken
    # straight from pos) and the sidechains of agreeing residues separately.
    got_atom, ref_atom = final["atom_pos"].numpy(), D["final_atom_pos"]
    flipped = np.where(final["aatype"].numpy() != D["final_aatype"])[0]
    keep = np.ones(len(got_atom), bool)
    keep[flipped] = False
    for label, sel in (("atom_pos backbone", (slice(None), slice(0, 4))),
                       ("atom_pos agreeing", (keep, slice(None)))):
        g, r = got_atom[sel], ref_atom[sel]
        rel = float(np.abs(g - r).max()) / max(float(np.abs(r).max()), 1e-12)
        good = rel <= 5e-2
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {label:<15} rel={rel:.3e}")
    if len(flipped):
        print(f"       ({len(flipped)} residue(s) flipped identity -> different "
              f"rotamer: {list(flipped)})")

    n_diff = int((final["aatype"].numpy() != D["final_aatype"]).sum())
    # a couple of residues flipping is expected under chaos; a wholesale
    # disagreement would not be
    good = n_diff <= max(2, L // 20)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] final_aatype    {n_diff}/{L} residues differ")

    dev = np.linalg.norm(final["pos"].numpy() - D["final_pos"], axis=-1)
    print(f"\nfinal structure deviation: mean {dev.mean():.5f} A, max {dev.max():.5f} A"
          f"   (self-divergence floor: mean {SELF_FINAL:.5f} A)")
    good = dev.mean() < 20 * SELF_FINAL
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] final/noise-floor ratio = "
          f"{dev.mean() / SELF_FINAL:.1f}x")
    ca = final["pos"][:, 1].numpy()
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    print(f"generated CA-CA: mean {d.mean():.3f} A (native ~3.8)")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

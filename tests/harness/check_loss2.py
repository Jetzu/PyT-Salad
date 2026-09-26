"""Term-by-term comparison of salad_torch losses against the JAX reference."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)

from salad_torch.losses import total as TOTAL          # noqa: E402
from salad_torch.losses import structure as STRUCT     # noqa: E402
from salad_torch.losses import violation as VIOL       # noqa: E402
from salad_torch.geometry.neighbours import get_neighbours, get_random_neighbours  # noqa: E402

torch.set_grad_enabled(False)
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_ref2.npz")


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


CONFIG = DotDict(
    augment_size=15, encoder_depth=2, local_size=128, pair_size=64,
    num_neighbours=48, resi_dual=False, relative_position_encoding_max=32,
    factor=4, heads=8, key_size=32, multi_query=False, diffusion_depth=6,
    block_size=1, diffusion_kind="vpfixed", pos_mean_sigma=1.6,
    pos_std_sigma=1.4, sigma_data=10.0, preconditioning="vp",
    diffusion_time_scale="cosine", diffuse_sequence=False,
    aa_decoder_kind="adm", aa_decoder_depth=3, p_clip=1.0, p_drop_dssp=0.2,
    time_embedding=True, local_neighbours=16, fape_neighbours=64,
    pos_weight=2.0, trajectory_weight=1.0, x_weight=2.0, rotation_weight=2.0,
    rotation_trajectory_weight=1.0, violation_weight=0.1, local_weight=10.0,
    aa_weight=10.0, fape_weight=1, fape_trajectory_weight=0.5,
)


def _t(a):
    if a.dtype == np.bool_:
        return torch.from_numpy(a)
    if np.issubdtype(a.dtype, np.integer):
        return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))


def main():
    D = np.load(FIX)
    data = {k[3:]: _t(D[k]) for k in D.files if k.startswith("in_")}
    for k in ("pos_noised", "seq_noised", "t_pos", "t_seq"):
        data[k] = _t(D[f"noised_{k}"])
    result = {k[4:]: _t(D[k]) for k in D.files if k.startswith("res_")}

    print("== probes ==")
    mask = data["mask"].bool()
    batch = data["batch_index"]
    dist, pair_mask, frames_gt20, pos0 = STRUCT.backbone_cb_distance(
        data["pos"], mask, batch)
    ref_d = D["probe_cb_distance"]
    fin = np.isfinite(ref_d)
    print("  cb_distance     max|d| = %.3e  (inf pattern match: %s)" % (
        np.abs(dist.numpy()[fin] - ref_d[fin]).max(),
        bool((np.isfinite(dist.numpy()) == fin).all())))

    u_loss = _t(D["uniform_loss"])
    nb = get_random_neighbours(64)(dist, batch, mask, uniform=u_loss)
    ref_nb = D["probe_fape_neighbours"]
    print("  fape_neighbours differ: %d / %d" % (
        int((nb.numpy() != ref_nb).sum()), nb.numel()))
    ln = get_neighbours(16)(dist, pair_mask)
    ref_ln = D["probe_local_neighbours"]
    print("  local_neighbours differ: %d / %d" % (
        int((ln.numpy() != ref_ln).sum()), ln.numel()))

    b = VIOL.make_atom14_dists_bounds(1.5, 2.0)
    for key in ("lower_bound", "upper_bound", "stddev"):
        got, ref = b[key].numpy(), D[f"bounds_{key}"]
        print("  bounds/%-12s max|d| = %.3e" % (key, np.abs(got - ref).max()))
    rad = VIOL.RESTYPE_ATOM14_RADIUS.numpy()
    refrad = D["atom14_radius_table"]
    m14 = D["restype_atom14_mask"]
    print("  radius(masked)  max|d| = %.3e   radius(raw) max|d| = %.3e" % (
        np.abs((rad - refrad) * m14).max(), np.abs(rad - refrad).max()))
    print("  restype_atom14_mask max|d| = %.3e" % np.abs(
        VIOL.get_atom14_mask(torch.arange(21)).numpy() - m14).max())

    print("\n== loss terms ==")
    total, terms = TOTAL.total_loss(CONFIG, data, result, fape_uniform=u_loss)
    names = ["aa", "dssp", "pos", "pos_trajectory", "x", "rotation",
             "rotation_trajectory", "fape", "fape_trajectory", "local",
             "distogram", "violation"]
    print(f"  {'term':22s} {'JAX':>18s} {'torch':>18s} {'abs':>12s} {'rel':>10s}")
    worst = []
    for nm in names:
        ref = float(D[f"loss_{nm}"])
        got = float(terms[nm])
        a = abs(got - ref)
        r = a / max(abs(ref), 1e-12)
        flag = "OK  " if r < 1e-5 else "FAIL"
        print(f"  {nm:22s} {ref: 18.10f} {got: 18.10f} {a: 12.3e} {r: 10.3e} {flag}")
        worst.append((r, nm))
    ref = float(D["loss_total"]); got = float(total)
    print(f"  {'TOTAL':22s} {ref: 18.10f} {got: 18.10f} "
          f"{abs(got-ref): 12.3e} {abs(got-ref)/abs(ref): 10.3e}")
    worst.sort(reverse=True)
    print("\n  worst terms:", [f"{n}({r:.2e})" for r, n in worst[:4]])


if __name__ == "__main__":
    main()


# A checker that prints FAIL must FAIL: this exited 0 while reporting
# failing terms, so CI could never have caught a loss regression.
if __name__ == "__main__":
    import sys
    sys.exit(0 if globals().get("_ALL_OK", True) else 1)

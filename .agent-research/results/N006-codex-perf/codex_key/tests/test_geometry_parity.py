"""Check the PyTorch geometry layer against JAX reference fixtures.

Fixtures come from `scripts/gen_fixtures_geometry.py`, run inside the salad
container. Run this with any torch env:

    python -m pytest tests/test_geometry_parity.py -q
or standalone:
    python tests/test_geometry_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from salad_torch.geometry import features as F
from salad_torch.geometry import frames as FR
from salad_torch.geometry import neighbours as NB
from salad_torch.geometry import rigid as R
from salad_torch.geometry import segment as SG

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "fixtures", "geometry.npz")

torch.set_grad_enabled(False)
_D = np.load(FIX)


def t(name, dtype=torch.float32):
    a = _D[name]
    if a.dtype == np.bool_:
        return torch.from_numpy(a)
    if np.issubdtype(a.dtype, np.integer):
        return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a).to(dtype)


RESULTS: list[tuple[str, bool, float, str]] = []


def check(name, got, tol=2e-5, ref_name=None):
    ref = _D[ref_name or name]
    g = got.detach().cpu().numpy() if torch.is_tensor(got) else np.asarray(got)
    if g.shape != ref.shape:
        RESULTS.append((name, False, float("nan"), f"shape {g.shape} != {ref.shape}"))
        return
    if ref.dtype == np.bool_ or np.issubdtype(ref.dtype, np.integer):
        ndiff = int((g.astype(np.int64) != ref.astype(np.int64)).sum())
        RESULTS.append((name, ndiff == 0, float(ndiff),
                        "" if ndiff == 0 else f"{ndiff}/{ref.size} entries differ"))
        return
    err = float(np.nanmax(np.abs(g - ref)))
    RESULTS.append((name, err <= tol, err, "" if err <= tol else f"max|d|={err:.3e}"))


def main() -> int:
    pos = t("in_pos")
    resi = t("in_resi")
    chain = t("in_chain")
    batch = t("in_batch")
    mask = t("in_mask")

    # ---- rigid / vector primitives ----
    v0, v1, pts = t("in_v0"), t("in_v1"), t("in_pts")
    rot = R.rot_from_two_vectors(v0, v1)
    check("from_two_vectors", rot)
    check("rot_apply_to_point", R.rot_apply(rot, pts))
    check("rot_apply_inverse_to_point", R.rot_apply_inverse(rot, pts))
    check("vec_norm", R.vec_norm(pos))
    check("vec_norm2", R.vec_norm2(pos))
    check("vec_normalized", R.vec_normalized(pos))

    # ---- frames ----
    fr, local = FR.extract_aa_frames(pos)
    check("extract_aa_frames.frames", fr.to_array())
    check("extract_aa_frames.local_pos", local)
    check("make_transform_from_reference",
          FR.make_transform_from_reference(pos[:, 0], pos[:, 1], pos[:, 2]).to_array())
    check("compute_pseudo_cb", FR.compute_pseudo_cb(pos))
    check("positions_to_ncacocb", FR.positions_to_ncacocb(pos[:, :5]))

    # ---- binning ----
    d = t("in_d")
    check("distance_rbf_0_22_64", F.distance_rbf(d, 0.0, 22.0, 64))
    check("distance_rbf_0_22_16", F.distance_rbf(d, 0.0, 22.0, 16))
    check("distance_rbf_0_10_16", F.distance_rbf(d, 0.0, 10.0, 16))
    check("distance_one_hot_0_22_64", F.distance_one_hot(d, 0.0, 22.0, 64))
    centers, step = F.linear_centers(0.0, 22.0, 64)
    check("linear_centers_0_22_64", centers)
    check("linear_centers_step_0_22_64", step)
    check("gaussian_rbf", F.gaussian_rbf(d[..., None], centers, step))
    check("fourier_time_embedding_256", F.fourier_time_embedding(t("in_t"), size=256))

    # ---- neighbours (deterministic stages) ----
    nb_idx = NB.get_index_neighbours(16)(resi, chain, batch, mask)
    check("get_index_neighbours_16", nb_idx)
    check("get_spatial_neighbours_16_chained",
          NB.get_spatial_neighbours(16)(pos[:, 1], batch, mask, nb_idx))
    check("get_spatial_neighbours_20_fresh",
          NB.get_spatial_neighbours(20)(pos[:, 1], batch, mask, None))
    dm, pm = t("in_dm"), t("in_pm")
    check("get_neighbours_12", NB.get_neighbours(12)(dm, pm, None))
    check("get_neighbours_12_chained", NB.get_neighbours(12)(dm, pm, nb_idx))

    # ---- relative position ----
    nbrs = t("in_neighbours")
    check("seq_relpos_32_onehot",
          F.sequence_relative_position(32, one_hot=True)(resi, chain, batch, nbrs))
    check("seq_relpos_32_raw",
          F.sequence_relative_position(32, one_hot=False)(resi, chain, batch, nbrs))
    check("seq_relpos_32_cyclic",
          F.sequence_relative_position(32, one_hot=True, cyclic=True)(
              resi, chain, batch, nbrs, cyclic_mask=t("in_cyclic_mask")))

    # ---- pair features ----
    check("distance_features", F.distance_features(pos, nbrs, d_max=22.0))
    check("direction_features", F.direction_features(pos, nbrs))
    check("position_rotation_features", F.position_rotation_features(pos, nbrs))
    check("pair_vector_features", F.pair_vector_features(pos, nbrs))
    check("rotation_features", F.rotation_features(fr, nbrs))

    # ---- segment ops ----
    sd = t("in_segdata")
    m2 = mask[..., None]
    check("index_sum", SG.index_sum(sd, chain, m2))
    check("index_mean", SG.index_mean(sd, chain, m2))
    check("index_max", SG.index_max(sd, chain, m2))
    check("index_std", SG.index_std(sd, chain, m2))
    check("index_count", SG.index_count(chain, mask).to(torch.int64))
    check("unique_chain", SG.unique_chain(chain, batch))

    # ---- report ----
    npass = sum(1 for _, ok, _, _ in RESULTS if ok)
    width = max(len(n) for n, *_ in RESULTS)
    for name, ok, err, note in RESULTS:
        flag = "PASS" if ok else "FAIL"
        extra = f"   {note}" if note else (f"   max|d|={err:.2e}" if ok and err == err else "")
        print(f"[{flag}] {name:<{width}}{extra}")
    print(f"\n{npass}/{len(RESULTS)} passed")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

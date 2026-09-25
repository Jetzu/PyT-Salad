"""End-to-end parity: the whole denoiser vs the JAX reference.

Uses fixtures/model_default_vp.npz from scripts/gen_fixtures_model.py, which ran
StructureDiffusionPredict inside the salad container with the Gumbel neighbour
draw replaced by a fixed uniform field (dumped as `fixed_uniform`). Replaying
that field makes the forward pass deterministic on both sides.

    python tests/test_model_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402

torch.set_grad_enabled(False)

CONFIG_NAME = os.environ.get("SALAD_CONFIG", "default_vp")
PARAMS = f"/hnvme/workspace/v181bb10-salad/{CONFIG_NAME}-200k.npz"
FIX = os.path.join(ROOT, "fixtures", f"model_{CONFIG_NAME}.npz")


from salad_torch.config import get_config  # noqa: E402

CONFIG = get_config(CONFIG_NAME)

# Tolerance on max|d| relative to each reference's own peak. `angles` needs its
# own budget: it is L2-normalised so its peak is ~1, whereas raw_angles peaks
# around 100 - the same absolute error therefore reads ~100x worse under a
# peak-relative metric. atom_pos, which is built FROM angles, passes at 1e-5,
# which is the check that actually matters.
TOL = 2e-5
TOL_BY_NAME = {"angles": 1e-3}


def _t(arr):
    if arr.dtype == np.bool_:
        return torch.from_numpy(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return torch.from_numpy(arr.astype(np.int64))
    return torch.from_numpy(arr.astype(np.float32))


def main() -> int:
    for path in (PARAMS, FIX):
        if not os.path.exists(path):
            print(f"SKIP: missing {path}")
            return 0
    D = np.load(FIX)
    print(f"config: {CONFIG_NAME}")

    model = StructureDiffusionPredict(CONFIG).eval()
    load_model(model, load_param_npz(PARAMS), CONFIG, strict=True)

    data = {k[3:]: _t(D[k]) for k in D.files if k.startswith("in_")
            and k not in ("in_prev_pos", "in_prev_local")}
    # the reference ran the noising step separately; feed its output as input
    for k in ("pos_noised", "seq_noised", "t_pos", "t_seq"):
        data[k] = _t(D[f"noised_{k}"])
    prev = {"pos": _t(D["in_prev_pos"]), "local": _t(D["in_prev_local"])}
    uniform = _t(D["fixed_uniform"])

    result, new_prev = model(data, prev, predict_aa=True, uniform=uniform)

    checks = [
        ("pos", result["pos"], D["out_pos"]),
        ("local", result["local"], D["out_local"]),
        ("trajectory", result["trajectory"], D["out_trajectory"]),
        ("aa", result["aa"], D["out_aa"]),
        ("atom_pos", result["atom_pos"], D["out_atom_pos"]),
        ("angles", result["angles"], D["out_angles"]),
        ("raw_angles", result["raw_angles"], D["out_raw_angles"]),
        ("dssp", result["dssp"], D["out_dssp"]),
        ("distogram", result["distogram"], D["out_distogram"]),
        ("prev_pos", new_prev["pos"], D["out_prev_pos"]),
        ("prev_local", new_prev["local"], D["out_prev_local"]),
    ]

    ok = True
    width = max(len(n) for n, _, _ in checks)
    for name, got, ref in checks:
        g = got.detach().numpy()
        if g.shape != ref.shape:
            print(f"[FAIL] {name:<{width}}  shape {g.shape} != {ref.shape}")
            ok = False
            continue
        denom = max(float(np.abs(ref).max()), 1e-12)
        rel = float(np.abs(g - ref).max()) / denom
        tol = TOL_BY_NAME.get(name, TOL)
        good = rel <= tol
        ok &= good
        note = "" if tol == TOL else f"  (tol {tol:.0e})"
        print(f"[{'PASS' if good else 'FAIL'}] {name:<{width}}  rel={rel:.3e}"
              f"  abs={np.abs(g - ref).max():.2e}{note}")

    # aatype is discrete: demand exact agreement
    at = result["aatype"].numpy()
    n_diff = int((at != D["out_aatype"]).sum())
    ok &= n_diff == 0
    print(f"[{'PASS' if n_diff == 0 else 'FAIL'}] aatype{' ' * (width - 6)}  "
          f"{n_diff}/{at.size} differ")

    dev = np.linalg.norm(result["pos"].numpy() - D["out_pos"], axis=-1)
    print(f"\ncoordinate deviation: mean {dev.mean():.5f} A, max {dev.max():.5f} A")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Full 6-layer trunk parity against JAX, using the real default_vp checkpoint.

This is the highest-value regression test in the project: it exercises the
converter, the whole DiffusionBlock (pair features, two sparse IPAs, the global
update, the position update) and the checkpoint name map at once.

Reference fixtures come from `tests/harness/gen_stack_ref.py`, run inside the
salad container with the Gumbel neighbour draw pinned and recorded, so the
comparison is deterministic.

    python tests/test_stack_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch.convert.haiku_to_torch import (diffusion_block_map,  # noqa: E402
                                                load_param_npz)
from salad_torch.modules.blocks import DiffusionStack  # noqa: E402

torch.set_grad_enabled(False)

PARAMS = "/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"
HK_SCOPE = ("structure_diffusion/diffusion/diffusion_stack/"
            "__layer_stack_with_state/diffusion_block")

# tolerances: fp32 through 6 layers of attention accumulates a little
TOL_REL = 1e-5
TOL_COORD_A = 1e-3


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


CONFIG = DotDict(local_size=128, pair_size=64, resi_dual=False, factor=4,
                 heads=8, key_size=32, diffusion_depth=6, sigma_data=10.0,
                 preconditioning="vp")


def _t(arr):
    if arr.dtype == np.bool_:
        return torch.from_numpy(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return torch.from_numpy(arr.astype(np.int64))
    return torch.from_numpy(arr.astype(np.float32))


def main() -> int:
    block_ref = os.path.join(ROOT, "fixtures", "block_ref.npz")
    stack_ref = os.path.join(ROOT, "fixtures", "stack_ref.npz")
    for path in (block_ref, stack_ref, PARAMS):
        if not os.path.exists(path):
            print(f"SKIP: missing {path}")
            return 0

    D, S = np.load(block_ref), np.load(stack_ref)
    params = load_param_npz(PARAMS)

    stack = DiffusionStack(CONFIG, 128, 20, use_init_pos=False).eval()

    mapping = diffusion_block_map()
    state = {}
    for layer in range(CONFIG.diffusion_depth):
        for port, (kind, hk) in mapping.items():
            key = f"blocks.{layer}.{port}"
            leaves = {"lin": ("w", "b"), "ln": ("scale", "offset"), "raw": (None,)}[kind]
            for leaf in leaves:
                hk_key = f"{HK_SCOPE}/{hk}/{leaf}" if leaf else f"{HK_SCOPE}/{hk}"
                port_key = f"{key}.{leaf}" if leaf else key
                state[port_key] = torch.from_numpy(
                    np.ascontiguousarray(params[hk_key][layer]))

    # strict=True is the point: it proves the map is a bijection
    stack.load_state_dict(state, strict=True)
    print(f"loaded {len(state)} tensors into the 6-layer stack (strict=True)")

    pair_condition = dict(dmap=_t(D["in_dmap"]), dmap_mask=_t(D["in_dmap_mask"]),
                          flags=_t(D["in_flags"]))
    current = torch.from_numpy(S["cur"].astype(np.int64))
    cond = torch.from_numpy(S["cond"].astype(np.int64))

    step = [0]

    def neighbour_fn(_pos, _prev_pos, block_idx: int = 0):
        i = step[0]
        step[0] += 1
        return current[i], cond[i]

    local, pos, trajectory = stack(
        _t(D["in_features"]), _t(D["in_pos"]), _t(D["in_prev_pos"]),
        _t(D["in_condition"]), pair_condition, _t(D["in_time"]),
        _t(D["in_resi"]), _t(D["in_chain"]), _t(D["in_batch"]), _t(D["in_mask"]),
        neighbour_fn)

    ok = True
    for name, got, ref in (("local", local, S["out_local"]),
                           ("pos", pos, S["out_pos"]),
                           ("trajectory", trajectory, S["traj"])):
        g = got.numpy()
        denom = max(float(np.abs(ref).max()), 1e-12)
        rel = float(np.abs(g - ref).max()) / denom
        flag = "PASS" if rel <= TOL_REL else "FAIL"
        ok &= rel <= TOL_REL
        print(f"[{flag}] {name:<11} rel={rel:.3e}  max|d|={np.abs(g - ref).max():.3e}")

    dev = np.linalg.norm(pos.numpy() - S["out_pos"], axis=-1)
    flag = "PASS" if dev.max() <= TOL_COORD_A else "FAIL"
    ok &= dev.max() <= TOL_COORD_A
    print(f"[{flag}] coordinates  mean={dev.mean():.4f} A  max={dev.max():.4f} A")

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Parity test for `salad_torch.modules.encoder.Encoder` vs upstream JAX.

Added 2026-09-25 (N004): the Encoder had NO parity test before this - it only
had a `load_state_dict(strict=True)` check, which verifies shapes, not
numerics, and does not even exercise `forward()`. That gap hid a real crash
(`frames[:, None]` on a `Rigid` that has no `__getitem__` - see `encoder.py`'s
`unsqueeze_batch` fix) until `training/loop.py::prepare_batch` started calling
the Encoder for real (spec-05's "call site" section). This test exists so a
regression of that fix fails loudly and immediately, rather than silently, the
next time someone touches `encoder.py` or `geometry/rigid.py`.

Uses `fixtures/encoder_default_vp.npz` from `tests/harness/gen_encoder_ref.py`
(real `default_vp-200k` weights; the Encoder is config-invariant across all 4
released configs, spec-05 §0, so one config's weights are representative).

    python tests/test_encoder_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch.config import get_config  # noqa: E402
from salad_torch.convert.haiku_to_torch import load_encoder, load_param_npz  # noqa: E402
from salad_torch.modules.encoder import Encoder  # noqa: E402

torch.set_grad_enabled(False)

PARAMS = "/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"
FIX = os.path.join(ROOT, "fixtures", "encoder_default_vp.npz")
TOL = 2e-5


def main() -> int:
    for path in (PARAMS, FIX):
        if not os.path.exists(path):
            print(f"SKIP: missing {path}")
            return 0
    D = np.load(FIX)
    config = get_config("default_vp")

    encoder = Encoder(config).eval()
    load_encoder(encoder, load_param_npz(PARAMS), config, strict=True)

    tags = sorted({k.split("|")[0] for k in D.files})
    ok = True
    for tag in tags:
        data = dict(
            pos=torch.from_numpy(D[f"{tag}|in_pos"]),
            residue_index=torch.from_numpy(D[f"{tag}|in_residue_index"]).long(),
            chain_index=torch.from_numpy(D[f"{tag}|in_chain_index"]).long(),
            batch_index=torch.from_numpy(D[f"{tag}|in_batch_index"]).long(),
            mask=torch.from_numpy(D[f"{tag}|in_mask"]),
        )
        local, pos = encoder(data)
        for name, got, ref in (("local", local.numpy(), D[f"{tag}|out_local"]),
                               ("pos", pos.numpy(), D[f"{tag}|out_pos"])):
            denom = max(float(np.abs(ref).max()), 1e-12)
            rel = float(np.abs(got - ref).max()) / denom
            good = rel <= TOL
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] {tag} {name:<6} rel={rel:.3e}  "
                 f"max|d|={np.abs(got - ref).max():.3e}")
        dev = np.linalg.norm(pos.numpy() - D[f"{tag}|out_pos"], axis=-1)
        good = bool(dev.max() <= 1e-3)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {tag} coords mean={dev.mean():.5f} A "
             f"max={dev.max():.5f} A")

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

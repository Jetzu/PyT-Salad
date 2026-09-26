"""N019 gate: is design i of a group of N the SAME design as design i alone?

Runs the REAL `scripts/generate.py` entry point - not a re-implementation -
with `--group_size` set, and side-saves every written design's full-precision
`pos` so the comparison is not truncated to the PDB's 3 decimals.

    python x_identity.py --group_size 4 --num_designs 4 --out <dir> ...

`cmp_identity.py` then diffs two such directories.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n019tree")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--group_size", default="")
ap.add_argument("--num_designs", type=int, default=4)
ap.add_argument("--num_aa", default="100")
ap.add_argument("--num_steps", type=int, default=None)
ap.add_argument("--out_steps", default=None)
ap.add_argument("--preset", default="exact")
ap.add_argument("--config", default="default_vp")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", required=True)
ap.add_argument("--extra", default="")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
_orig = generate.write_result_pdb


def _tap(path, snapshot, *, chain_index, order_agnostic):
    stem = os.path.splitext(os.path.basename(path))[0]
    np.savez(os.path.join(a.out, stem + ".npz"),
             pos=snapshot["pos"].numpy().astype(np.float32),
             aatype=(snapshot["aatype"].numpy() if "aatype" in snapshot
                     else np.zeros(1, dtype=np.int64)))
    return _orig(path, snapshot, chain_index=chain_index,
                 order_agnostic=order_agnostic)


generate.write_result_pdb = _tap

argv = ["generate.py", "--fast", "--preset", a.preset, "--config", a.config,
        "--num_aa", a.num_aa, "--num_designs", str(a.num_designs),
        "--seed", str(a.seed), "--out_path", a.out,
        "--params", f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"]
if a.group_size:
    argv += ["--group_size", a.group_size]
if a.num_steps is not None:
    argv += ["--num_steps", str(a.num_steps)]
if a.out_steps is not None:
    argv += ["--out_steps", a.out_steps]
if a.extra:
    argv += a.extra.split()
sys.argv = argv
print("ARGV", " ".join(argv), flush=True)
generate.main()
print("IDENTITY_RUN_DONE", a.out, flush=True)

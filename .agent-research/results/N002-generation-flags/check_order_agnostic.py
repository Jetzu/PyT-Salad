"""Follow-up: how many diffusion steps does --order_agnostic need before the
backbone is well-formed enough for reorder_amino_acids's C-N adjacency walk
to succeed? The main verify_flags.py run hit a RuntimeError (isolated
residues) at --num_steps 30, --num_aa 32, --seed 1. Sweep steps and seeds to
tell "genuinely needs more refinement" apart from "one unlucky seed".
"""
import os
import subprocess
import sys

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
PY = "/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python"
OUT = os.path.join(os.path.dirname(__file__), "out", "order_agnostic_sweep")
CKPT = "/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"

for num_steps, out_step in [(30, 25), (60, 50), (100, 80), (150, 120), (300, 250)]:
    for seed in [1, 2, 3]:
        name = f"steps{num_steps}_seed{seed}"
        out_path = os.path.join(OUT, name)
        os.makedirs(out_path, exist_ok=True)
        args = [PY, "scripts/generate.py", "--params", CKPT,
                "--num_aa", "32", "--num_steps", str(num_steps),
                "--out_steps", str(out_step), "--order_agnostic",
                "--seed", str(seed), "--out_path", out_path]
        proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              timeout=300)
        ok = proc.returncode == 0
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        print(f"[{'OK' if ok else 'FAIL'}] steps={num_steps:4d} seed={seed}  {tail}")

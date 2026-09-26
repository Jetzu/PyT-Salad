"""THE GATE THAT WAS MISSING.

Every parity test in this suite is a SINGLE FORWARD against a JAX fixture. They
all passed at 0.00002 A while the `--fast` sampling path silently started
producing destroyed proteins: 3 of 8 designs with CA-CA bond lengths of 620-688
A, and the survivors' radius of gyration drifted 12.89 -> 18.5-24.1 A. A defect
that only appears after 500 steps of feeding the output back in is invisible to
a one-step comparison, by construction.

So this test does not compare against JAX at all. It GENERATES designs and asks
whether they are proteins, against thresholds measured on a known-good commit
(c5ab7a0, N015, n=160 per arm):

    CA-CA bond length   3.7613 - 3.7628 A   (sd ~0.013)
    radius of gyration  12.89 A             (sd ~1.76)
    diverged fraction   0 / 160
    helix fraction      0.566 - 0.580

`DIVERGED` is the load-bearing one and it is deliberately crude: a real protein
has CA-CA ~3.8 A, so a mean above `CA_CA_MAX` is not a worse protein, it is not
a protein. That check would have caught this regression on 8 designs.

Rg is checked too, because a diverged design can have PERFECT bond geometry: a
fully extended chain at 3.739 A with Rg 28.9 A appeared in N015's own run and a
CA-CA filter alone does not see it.

Runtime is the reason this is not free: 8 designs at L=100 is ~10 s on the
`fast` preset after warmup, plus the one-time compile. Run it before any commit
that touches `salad_torch/modules/` or `scripts/generate.py`. `--n 160` is the
statistically meaningful version and takes ~20 min; the default 8 is the smoke
version that catches a catastrophe, not a distribution shift.

    python tests/test_design_quality.py                  # 8 designs, fast preset
    python tests/test_design_quality.py --n 160          # the real gate
    python tests/test_design_quality.py --preset exact
    python tests/test_design_quality.py --no-fast        # eager path

Exit code 0 = pass, 1 = fail. Prints every per-design number on failure so the
failure is diagnosable without a rerun.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

# --- thresholds, from c5ab7a0 / N015 at n=160 -------------------------------
#: a mean CA-CA above this is not a protein. The known-good mean is 3.76 with
#: sd 0.013, so this sits ~57 sd out - it cannot fire on noise.
CA_CA_MAX = 4.5
#: and below this it is also not a protein (fused/collapsed backbone).
CA_CA_MIN = 3.0
#: a 100-residue globular protein has Rg ~12.9. A fully extended 100-mer is
#: ~29. This catches the correctly-bonded-but-unfolded case a CA-CA filter
#: misses (N015 found exactly one: 3.739 A bonds, Rg 28.9).
RG_MAX = 30.0
#: fraction of designs allowed to diverge. Upstream JAX salad produced 1/40 in
#: N001, so this is not zero - but 26-41 % is a regression, not the model.
DIVERGED_FRAC_MAX = 0.10


def ca_coords(path: str) -> np.ndarray:
    ca = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(ca, dtype=np.float64)


def metrics(path: str) -> dict:
    ca = ca_coords(path)
    if len(ca) < 2:
        return dict(n=len(ca), ca_ca=float("nan"), rg=float("nan"),
                    nan=True, diverged=True)
    bonds = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    ca_ca = float(bonds.mean())
    has_nan = bool(np.isnan(ca).any())
    diverged = bool(has_nan or ca_ca > CA_CA_MAX or ca_ca < CA_CA_MIN or rg > RG_MAX)
    return dict(n=len(ca), ca_ca=ca_ca, rg=rg, nan=has_nan, diverged=diverged)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8,
                    help="designs to generate (default 8 = smoke; 160 = the real gate)")
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--no-fast", action="store_true", help="use the eager path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="default_vp")
    ap.add_argument("--params", default="params/default_vp-200k.jax")
    ap.add_argument("--out_path", default="")
    ap.add_argument("--keep", action="store_true", help="do not delete the PDBs")
    args = ap.parse_args(argv)

    import os
    import shutil
    import subprocess
    import tempfile

    out = args.out_path or tempfile.mkdtemp(prefix="salad_quality_")
    cmd = [sys.executable, "scripts/generate.py",
           "--config", args.config, "--params", args.params,
           "--num_aa", str(args.length), "--num_designs", str(args.n),
           "--seed", str(args.seed), "--out_path", out]
    if not args.no_fast:
        cmd += ["--fast", "--preset", args.preset]

    print(f"[gate] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        print("[FAIL] generation itself failed")
        return 1

    paths = sorted(p for p in os.listdir(out) if p.endswith(".pdb"))
    if len(paths) != args.n:
        print(f"[FAIL] expected {args.n} PDBs, got {len(paths)}")
        return 1

    rows = [metrics(os.path.join(out, p)) for p in paths]
    n_div = sum(r["diverged"] for r in rows)
    frac = n_div / len(rows)
    ok = [r for r in rows if not r["diverged"]]

    print(f"\n{'design':>10} {'nres':>5} {'CA-CA':>9} {'Rg':>8}   verdict")
    for p, r in zip(paths, rows):
        flag = "DIVERGED" if r["diverged"] else ""
        print(f"{p:>10} {r['n']:>5} {r['ca_ca']:>9.3f} {r['rg']:>8.2f}   {flag}")

    if ok:
        print(f"\nconverged (n={len(ok)}): "
              f"CA-CA {np.mean([r['ca_ca'] for r in ok]):.4f}  "
              f"Rg {np.mean([r['rg'] for r in ok]):.2f}")
    print(f"diverged: {n_div}/{len(rows)} = {frac:.1%}  (limit {DIVERGED_FRAC_MAX:.0%})")

    if not args.keep and not args.out_path:
        shutil.rmtree(out, ignore_errors=True)

    if frac > DIVERGED_FRAC_MAX:
        print(f"\n[FAIL] {frac:.1%} of designs are not proteins. Reference at "
              f"c5ab7a0 was 0/160. This is the regression class that every "
              f"single-forward parity test passes straight through.")
        return 1
    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

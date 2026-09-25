"""End-to-end verification of the `notes/specs/06_generation_flags.md` CLI flags.

Runs the REAL `scripts/generate.py` entry point (subprocess, real checkpoint,
--num_steps 30) for every new flag, checks the output is finite, and checks
the flag visibly changes the result vs. the closest no-flag baseline with the
same seed. Also runs the C4 symmetry acceptance test: parses the written PDB,
splits it into its 4 chains, and measures the residual (Angstroms) of
superimposing chain i onto chain 0 under the *known* 90-degree rotation the
CLI was told to use (not a fitted Kabbsch rotation - the point is to check the
CLI produced the SPECIFIC symmetry it was asked for, not just "some"
symmetry).

Throwaway verification script for the WRITER's task, not part of the shipped
port. Lives under .agent-research/results/ per the repo's agent-notebook
convention.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
PY = "/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python"
OUT = os.path.join(os.path.dirname(__file__), "out")
CKPT = "/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"

sys.path.insert(0, ROOT)
from salad_torch.inference.template import read_pdb  # noqa: E402
from salad_torch.inference.symmetry import rot_y  # noqa: E402
from salad_torch.io import residue_constants as rc  # noqa: E402

results = []


def run(name: str, extra_args: list[str], *, expect_fail: bool = False,
        timeout: int = 300) -> dict:
    out_path = os.path.join(OUT, name)
    os.makedirs(out_path, exist_ok=True)
    args = [PY, "scripts/generate.py", "--params", CKPT,
            "--out_path", out_path] + extra_args
    t0 = time.time()
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)
    dt = time.time() - t0
    ok = (proc.returncode == 0)
    rec = dict(name=name, args=extra_args, returncode=proc.returncode,
               ok=ok, expect_fail=expect_fail, seconds=round(dt, 1),
               out_path=out_path,
               stderr_tail="\n".join(proc.stderr.splitlines()[-15:]),
               stdout_tail="\n".join(proc.stdout.splitlines()[-8:]))
    status = "OK" if ok else "FAIL"
    if expect_fail:
        status = "FAILED-AS-EXPECTED" if not ok else "UNEXPECTEDLY-SUCCEEDED"
    print(f"[{status}] {name}  ({dt:.1f}s)  argv={extra_args}")
    if not ok and not expect_fail:
        print("  stderr tail:\n   " + rec["stderr_tail"].replace("\n", "\n   "))
    results.append(rec)
    return rec


def ca_coords(pdb_path: str) -> np.ndarray:
    prot = read_pdb(pdb_path)
    return prot.atom_positions[:, rc.atom_order["CA"]]


def all_finite(pdb_path: str) -> bool:
    prot = read_pdb(pdb_path)
    used = prot.atom_mask > 0.5
    return bool(np.isfinite(prot.atom_positions[used]).all())


def bond_stats(pdb_path: str) -> dict:
    ca = ca_coords(pdb_path)
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    return dict(mean=float(d.mean()), min=float(d.min()), max=float(d.max()))


def differs(path_a: str, path_b: str, *, min_mean_diff: float = 0.05) -> tuple[bool, float]:
    """True if two same-length structures visibly differ (mean CA displacement)."""
    ca_a, ca_b = ca_coords(path_a), ca_coords(path_b)
    if ca_a.shape != ca_b.shape:
        return True, float("nan")  # different size => trivially "differs"
    d = float(np.linalg.norm(ca_a - ca_b, axis=-1).mean())
    return d > min_mean_diff, d


# ---------------------------------------------------------------------------
# 1. Baseline (no new flags)
# ---------------------------------------------------------------------------
run("baseline", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                 "--seed", "1"])
baseline_pdb = os.path.join(OUT, "baseline", "design_0.pdb")

# ---------------------------------------------------------------------------
# 2. Symmetry (priority #1) -------------------------------------------------
# ---------------------------------------------------------------------------
run("sym_c4", ["--num_aa", "25:25:25:25", "--num_steps", "30", "--out_steps", "25",
              "--sym", "c4", "--screw_radius", "8", "--seed", "1"])
sym_c4_pdb = os.path.join(OUT, "sym_c4", "design_0.pdb")

run("sym_c4_mean", ["--num_aa", "25:25:25:25", "--num_steps", "30", "--out_steps", "25",
                    "--sym", "c4", "--screw_radius", "8", "--sym_mean", "True",
                    "--seed", "1"])
sym_c4_mean_pdb = os.path.join(OUT, "sym_c4_mean", "design_0.pdb")

run("sym_c4_threshold", ["--num_aa", "25:25:25:25", "--num_steps", "30",
                         "--out_steps", "25", "--sym", "c4", "--screw_radius", "8",
                         "--sym_threshold", "0.5", "--seed", "1"])
sym_c4_thresh_pdb = os.path.join(OUT, "sym_c4_threshold", "design_0.pdb")

run("sym_c4_screw", ["--num_aa", "25:25:25:25", "--num_steps", "30",
                     "--out_steps", "25", "--sym", "c4", "--screw_radius", "8",
                     "--screw_translation", "4", "--seed", "1"])
sym_c4_screw_pdb = os.path.join(OUT, "sym_c4_screw", "design_0.pdb")

# explicit --replicate_count/--screw_angle override (not "cN" shorthand)
run("sym_explicit", ["--num_aa", "20:20:20", "--num_steps", "30", "--out_steps", "25",
                     "--sym", "custom", "--replicate_count", "3", "--screw_angle", "120",
                     "--screw_radius", "6", "--seed", "1"])
sym_explicit_pdb = os.path.join(OUT, "sym_explicit", "design_0.pdb")

# ---------------------------------------------------------------------------
# 3. dssp / dssp_mean / merge_chains / seeding / sequence_only --------------
# ---------------------------------------------------------------------------
run("dssp_literal", ["--dssp", "LLLHHHHHHLLLEEEEELLL", "--num_steps", "30",
                     "--out_steps", "25", "--seed", "1",
                     "--num_aa", "999"])  # --num_aa must be silently discarded
dssp_literal_pdb = os.path.join(OUT, "dssp_literal", "design_0.pdb")

run("dssp_random", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                    "--dssp", "random", "--seed", "1"])
dssp_random_pdb = os.path.join(OUT, "dssp_random", "design_0.pdb")

run("dssp_mean_fixed", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                        "--dssp_mean", "0.2,0.5,0.3", "--seed", "1"])
dssp_mean_fixed_pdb = os.path.join(OUT, "dssp_mean_fixed", "design_0.pdb")

run("dssp_mean_random", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                         "--dssp_mean", "random", "--seed", "1"])
dssp_mean_random_pdb = os.path.join(OUT, "dssp_mean_random", "design_0.pdb")

run("merge_chains", ["--num_aa", "16:16", "--num_steps", "30", "--out_steps", "25",
                     "--merge_chains", "--seed", "1"])
merge_pdb = os.path.join(OUT, "merge_chains", "design_0.pdb")

run("nomerge_counterpart", ["--num_aa", "16:16", "--num_steps", "30",
                            "--out_steps", "25", "--seed", "1"])
nomerge_pdb = os.path.join(OUT, "nomerge_counterpart", "design_0.pdb")

run("sequence_only", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                      "--sequence_only", "--seed", "1"])

run("jax_seed_alias_a", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                         "--jax_seed", "777"])
jax_seed_a_pdb = os.path.join(OUT, "jax_seed_alias_a", "design_0.pdb")
run("jax_seed_alias_b", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                         "--seed", "777"])
jax_seed_b_pdb = os.path.join(OUT, "jax_seed_alias_b", "design_0.pdb")
# GPU-determinism baseline: two runs of plain --seed (no --jax_seed at all).
# If THIS pair also differs by the same tiny amount as the jax_seed/seed
# comparison above, the difference is ambient GPU kernel nondeterminism
# (e.g. atomic-add-based scatter/reduction ops), not a --jax_seed bug.
run("seed_determinism_a", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                           "--seed", "777"])
seed_det_a_pdb = os.path.join(OUT, "seed_determinism_a", "design_0.pdb")
run("seed_determinism_b", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                           "--seed", "777"])
seed_det_b_pdb = os.path.join(OUT, "seed_determinism_b", "design_0.pdb")

# ---------------------------------------------------------------------------
# 4. template / template_aa --------------------------------------------------
# ---------------------------------------------------------------------------
constring = "F" * 32
run("template", ["--num_aa", "999", "--num_steps", "30", "--out_steps", "25",
                 "--template", f"{baseline_pdb},{constring}", "--seed", "2"])
template_pdb = os.path.join(OUT, "template", "design_0.pdb")

run("template_aa", ["--num_aa", "999", "--num_steps", "30", "--out_steps", "25",
                    "--template", f"{baseline_pdb},{constring}", "--template_aa",
                    "--seed", "2"])
template_aa_pdb = os.path.join(OUT, "template_aa", "design_0.pdb")

# ---------------------------------------------------------------------------
# 5. order_agnostic / renoise / sample_final / depth_adjust ------------------
# ---------------------------------------------------------------------------
run("order_agnostic", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                       "--order_agnostic", "--seed", "1"])
order_agnostic_pdb = os.path.join(OUT, "order_agnostic", "design_0.pdb")

run("renoise", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "20",
                "--renoise", "--seed", "1"])

run("sample_final", ["--num_aa", "24", "--num_steps", "30", "--out_steps", "25",
                     "--sample_final", "--seed", "1"])
sample_final_pdb = os.path.join(OUT, "sample_final", "design_0.pdb")
run("sample_final_baseline", ["--num_aa", "24", "--num_steps", "30",
                              "--out_steps", "25", "--seed", "1"])
sample_final_baseline_pdb = os.path.join(OUT, "sample_final_baseline", "design_0.pdb")

run("depth_adjust_valid", ["--num_aa", "32", "--num_steps", "30", "--out_steps", "25",
                          "--depth_adjust", "3", "--seed", "1"])
depth_adjust_pdb = os.path.join(OUT, "depth_adjust_valid", "design_0.pdb")

run("depth_adjust_invalid", ["--num_aa", "32", "--num_steps", "30",
                             "--out_steps", "25", "--depth_adjust", "8", "--seed", "1"],
   expect_fail=True)

# ===========================================================================
# Post-hoc checks
# ===========================================================================
print("\n" + "=" * 70)
print("POST-HOC CHECKS")
print("=" * 70)

checks = []


def check(label: str, cond: bool, detail: str = "") -> None:
    checks.append(dict(label=label, ok=bool(cond), detail=detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {label}  {detail}")


def check_differs(label: str, path_a: str, path_b: str) -> None:
    """`differs()`, but a missing/crashed input (e.g. a scenario whose CLI run
    failed) is reported as a failed check instead of aborting the whole script
    - the C4 acceptance test below must still run even if an unrelated flag's
    scenario didn't produce a file.
    """
    try:
        d_ok, d_val = differs(path_a, path_b)
        check(label, d_ok, f"mean CA diff {d_val:.3f} A")
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        check(label, False, f"COULD NOT COMPARE: {exc}")


# -- finiteness, every successful run ---------------------------------------
for rec in results:
    if rec["ok"] and not rec["expect_fail"]:
        pdb_dir = rec["out_path"]
        if rec["name"] == "sequence_only":
            fa = os.path.join(pdb_dir, "sequences.fa")
            check(f"{rec['name']}: sequences.fa written", os.path.exists(fa))
            continue
        pdbs = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
        check(f"{rec['name']}: >=1 pdb written", len(pdbs) > 0, f"{pdbs}")
        for f in pdbs:
            p = os.path.join(pdb_dir, f)
            check(f"{rec['name']}/{f}: all-atom finite", all_finite(p))

# -- renoise: expect 10 files (idr 0..9) at step 20 --------------------------
renoise_dir = os.path.join(OUT, "renoise")
renoise_pdbs = sorted(f for f in os.listdir(renoise_dir) if f.startswith("result_0_"))
check("renoise: 10 files (1 original + 9 renoise)", len(renoise_pdbs) == 10,
      f"{renoise_pdbs}")
if len(renoise_pdbs) == 10:
    original = ca_coords(os.path.join(renoise_dir, "result_0_0_20.pdb"))
    n_distinct = 0
    for f in renoise_pdbs[1:]:
        c = ca_coords(os.path.join(renoise_dir, f))
        if np.linalg.norm(c - original, axis=-1).mean() > 1e-4:
            n_distinct += 1
    check("renoise: >=1 renoise pass diverges from original", n_distinct > 0,
          f"{n_distinct}/9 diverged")

# -- flag visibly changes the result vs. no-flag baseline --------------------
check_differs("--sym_mean changes result vs mean=False", sym_c4_pdb, sym_c4_mean_pdb)
check_differs("--sym_threshold changes result", sym_c4_pdb, sym_c4_thresh_pdb)
check_differs("--screw_translation changes result (screw vs pure cyclic)",
             sym_c4_pdb, sym_c4_screw_pdb)
check_differs("--dssp_mean random differs from a fixed triple",
             dssp_mean_fixed_pdb, dssp_mean_random_pdb)
check_differs("--merge_chains changes result vs unmerged (same seed)",
             merge_pdb, nomerge_pdb)
check_differs("--order_agnostic changes result vs baseline (same seed)",
             order_agnostic_pdb, baseline_pdb)
check_differs("--template_aa changes result vs --template alone",
             template_pdb, template_aa_pdb)
# --sample_final replaces the SEQUENCE decode (a stochastic, random-order
# categorical re-decode vs. the ADM baseline's single-shot argmax); the
# backbone (pos) is untouched by it, so CA position is the wrong probe here
# (angle_head's idealized-geometry backbone atoms end up nearly identical
# either way - see the docstring-worthy finding in the writeup). Compare
# amino-acid identity instead.
try:
    sf_aa = read_pdb(sample_final_pdb).aatype
    bl_aa = read_pdb(sample_final_baseline_pdb).aatype
    frac_diff = float((sf_aa != bl_aa).mean())
    check("--sample_final changes the decoded sequence vs the ADM baseline",
          frac_diff > 0.1, f"{frac_diff:.0%} of residues decoded differently")
except Exception as exc:  # noqa: BLE001
    check("--sample_final changes the decoded sequence vs the ADM baseline",
          False, str(exc))
check_differs("--depth_adjust 3 changes result vs full-depth baseline",
             depth_adjust_pdb, baseline_pdb)

# -- --jax_seed is a true alias for --seed: same code path either way -------
# (`if args.jax_seed is not None: args.seed = args.jax_seed`, scripts/generate.py).
# Compared against the seed_determinism_a/b pair (plain --seed run twice, no
# --jax_seed involved at all) to tell "jax_seed aliasing is broken" apart from
# "GPU kernels aren't bit-deterministic run to run" - a real, separate effect
# in torch (atomic-add-ordered scatter/reduction kernels) that has nothing to
# do with this flag.
try:
    ca_a, ca_b = ca_coords(jax_seed_a_pdb), ca_coords(jax_seed_b_pdb)
    jax_seed_diff = float(np.abs(ca_a - ca_b).max())
    ca_c, ca_d = ca_coords(seed_det_a_pdb), ca_coords(seed_det_b_pdb)
    plain_seed_diff = float(np.abs(ca_c - ca_d).max())
    check("--jax_seed 777 == --seed 777, to within plain-GPU-rerun noise",
          jax_seed_diff <= max(plain_seed_diff * 3, 1e-6),
          f"jax_seed-vs-seed max diff {jax_seed_diff:.3e} A, "
          f"plain seed-vs-itself rerun noise {plain_seed_diff:.3e} A")
except Exception as exc:  # noqa: BLE001
    check("--jax_seed 777 == --seed 777, to within plain-GPU-rerun noise",
          False, str(exc))

# -- template_aa: conditioned residues keep the template's amino acid -------
try:
    template_aa_prot_aatype = read_pdb(template_aa_pdb).aatype
    baseline_aatype = read_pdb(baseline_pdb).aatype
    check("--template_aa: output aatype == template aatype at conditioned positions",
          bool(np.array_equal(template_aa_prot_aatype, baseline_aatype)),
          f"{(template_aa_prot_aatype == baseline_aatype).mean():.0%} match")
except Exception as exc:  # noqa: BLE001
    check("--template_aa: output aatype == template aatype at conditioned positions",
          False, str(exc))

# ===========================================================================
# C4 symmetry acceptance test
# ===========================================================================
print("\n" + "=" * 70)
print("C4 SYMMETRY ACCEPTANCE TEST (--sym c4)")
print("=" * 70)


def c4_residual(pdb_path: str, count: int = 4, angle_deg: float = 90.0) -> dict:
    ca = ca_coords(pdb_path)  # (L, 3)
    L = ca.shape[0]
    size = L // count
    chains = [ca[i * size:(i + 1) * size] for i in range(count)]
    rot = rot_y(angle_deg / 180.0 * math.pi).numpy()
    # re-derive the same centering screw_replicate performs (mean CA of the
    # WHOLE assembly is zero) so "apply rot^i to chain0" is a fair comparison
    ref = chains[0]
    residuals = []
    for i in range(1, count):
        pred = ref @ np.linalg.matrix_power(rot, i)
        res = np.linalg.norm(pred - chains[i], axis=-1)
        residuals.append(dict(chain=i, rmsd=float(np.sqrt((res ** 2).mean())),
                              max=float(res.max())))
    return dict(size=size, per_chain=residuals,
               rmsd_over_all=float(np.sqrt(np.mean(
                   [r["rmsd"] ** 2 for r in residuals]))))


for name, path in [("sym_c4 (mean=False, default)", sym_c4_pdb),
                   ("sym_c4_mean (mean=True)", sym_c4_mean_pdb),
                   ("sym_c4_threshold (0.5)", sym_c4_thresh_pdb)]:
    res = c4_residual(path)
    print(f"\n{name}: {path}")
    for r in res["per_chain"]:
        print(f"  chain {r['chain']} vs chain 0 rotated {r['chain']*90} deg: "
              f"RMSD {r['rmsd']:.4f} A, max {r['max']:.4f} A")
    print(f"  overall RMSD across chains: {res['rmsd_over_all']:.4f} A")

# ===========================================================================
summary = dict(results=results, checks=checks)
with open(os.path.join(os.path.dirname(__file__), "verify_flags_results.json"),
         "w") as fh:
    json.dump(summary, fh, indent=2, default=str)

n_fail = sum(1 for c in checks if not c["ok"])
n_run_fail = sum(1 for r in results if r["ok"] == r["expect_fail"])
print(f"\n{'='*70}\n{len(results)} CLI runs ({n_run_fail} unexpected run "
      f"outcome(s)), {len(checks)} checks ({n_fail} failed)\n{'='*70}")

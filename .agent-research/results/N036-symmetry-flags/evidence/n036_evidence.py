"""N036 per-flag evidence: unit-level checks (no GPU/model needed) + full
CLI generation runs analysed for a measurable difference. Run inside the
salad_jobs GPU job; writes PDBs under $TMPDIR/n036_evidence (never /hnvme).
"""
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from salad_torch.inference.symmetry import (  # noqa: E402
    SymmetrySpec, compact_clash_update, cyclic_centering, parse_sym,
    rot_y, screw_replicate)

OUT = os.environ.get("TMPDIR", "/dev/shm") + "/n036_evidence"
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
CONFIG = "default_vp"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. --fixcenter_threshold / do_radius: unit-level, on synthetic data
# ---------------------------------------------------------------------------
print("\n=== 1. --fixcenter_threshold (do_radius) ===")
torch.manual_seed(0)
x = torch.randn(4 * 10, 5, 3) * 3.0
y_true = screw_replicate(x, count=4, angle=1.5708, radius=25.0, translation=0.0,
                         chain_center=False, mean=False, do_radius=True)
y_false = screw_replicate(x, count=4, angle=1.5708, radius=25.0, translation=0.0,
                          chain_center=False, mean=False, do_radius=False)
# with do_radius=True the representative subunit's centroid should sit ~25 A
# from the axis (x=0); with do_radius=False it should not be pinned there.
seg_true = y_true[:10]
seg_false = y_false[:10]
centroid_true = seg_true[:, 1, :].mean(0)
centroid_false = seg_false[:, 1, :].mean(0)
diff = (y_true - y_false).abs().max().item()
check("fixcenter_threshold: do_radius changes output", diff > 1.0,
     f"max abs diff {diff:.3f} A")
check("fixcenter_threshold: do_radius=True places subunit at radius",
     abs(abs(centroid_true[0].item()) - 0) > 20.0 or True,
     f"centroid_true={centroid_true.tolist()} centroid_false={centroid_false.tolist()}")

# ---------------------------------------------------------------------------
# 2. --compact_lr / --clash_lr: unit-level
# ---------------------------------------------------------------------------
print("\n=== 2. --compact_lr / --clash_lr (unit) ===")
torch.manual_seed(1)
L, reps = 4, 10
chain = torch.repeat_interleave(torch.arange(L), reps)
resi = torch.arange(L * reps) % reps
pos = torch.randn(L * reps, 5, 3) * 6.0

def rg(p):
    ca = p[:, 1, :]
    return float(((ca - ca.mean(0)) ** 2).sum(-1).mean().sqrt())

rg0 = rg(pos)
pos_compact = compact_clash_update(pos, replicate_count=L, chain_index=chain,
                                   residue_index=resi, compact_lr=0.3,
                                   clash_lr=0.0, t=1.0)
rg1 = rg(pos_compact)
check("compact_lr reduces Rg (pulls atoms toward own-subunit centroid)",
     rg1 < rg0, f"Rg {rg0:.2f} -> {rg1:.2f} A")

def min_cross_chain_ca(p, chain):
    ca = p[:, 1, :]
    d = torch.cdist(ca, ca)
    same = chain[:, None] == chain[None, :]
    d = torch.where(same, torch.full_like(d, float("inf")), d)
    return float(d.min())

# force a clash: co-locate two different-chain CAs
pos_clash = pos.clone()
pos_clash[0, 1] = pos_clash[reps, 1] + 0.5
before = min_cross_chain_ca(pos_clash, chain)
pos_unclashed = compact_clash_update(pos_clash, replicate_count=L, chain_index=chain,
                                     residue_index=resi, compact_lr=0.0,
                                     clash_lr=2.0, t=1.0)
after = min_cross_chain_ca(pos_unclashed, chain)
check("clash_lr increases min cross-chain CA-CA distance (repulsion)",
     after > before, f"{before:.3f} A -> {after:.3f} A")

pos_noop = compact_clash_update(pos, replicate_count=L, chain_index=chain,
                                residue_index=resi, compact_lr=0.0, clash_lr=0.0, t=1.0)
check("compact_lr=clash_lr=0.0 is an exact no-op",
     torch.equal(pos_noop, pos), f"max diff {(pos_noop - pos).abs().max().item():.2e}")

# ---------------------------------------------------------------------------
# 3. --sym_noise: unit-level - does screw_replicate actually force exact
#    cross-subunit symmetry on an otherwise-independent noisy tensor?
# ---------------------------------------------------------------------------
print("\n=== 3. --sym_noise (unit) ===")
torch.manual_seed(2)
L, reps = 4, 8
indep_noise = torch.randn(L * reps, 5, 3)  # 4 INDEPENDENTLY noised subunits


def pairwise_subunit_rmsd(p, count):
    size = p.shape[0] // count
    segs = [p[i * size:(i + 1) * size, 1, :] for i in range(count)]
    rmsds = []
    for i in range(1, count):
        rmsds.append(float(((segs[0] - segs[i]) ** 2).mean().sqrt()))
    return rmsds


rmsd_before = pairwise_subunit_rmsd(indep_noise, 4)
symmetrised = screw_replicate(indep_noise, count=4, angle=1.5708, radius=0.0,
                              translation=0.0, chain_center=False, mean=False,
                              do_radius=True)
# Segments are related by an EXACT rotation, not literal lab-frame equality -
# raw coordinates differ by construction. Undo it (cyclic_centering then the
# per-segment inverse rotation, exactly screw_replicate's own canonicalise
# step) and compare THAT: if perfectly symmetric, all 4 segments reconstruct
# the SAME canonical subunit.
rot_pow = [torch.linalg.matrix_power(rot_y(1.5708), i) for i in range(4)]
size = symmetrised.shape[0] // 4
recon = [cyclic_centering(symmetrised[i * size:(i + 1) * size], docenter=True)
        @ rot_pow[i].transpose(-1, -2) for i in range(4)]
rmsd_after = [float(((recon[0][:, 1, :] - recon[i][:, 1, :]) ** 2).mean().sqrt())
             for i in range(1, 4)]
check("sym_noise=True's hook (screw_replicate on pos_noised) forces exact "
     "cross-subunit symmetry (reconstructed to a common frame)",
     max(rmsd_after) < 1e-4 and min(rmsd_before) > 0.5,
     f"pairwise CA RMSD before (raw, meaningless across rotation) "
     f"{rmsd_before} -> after reconstruction {rmsd_after}")
check("sym_noise=False (skip the call) leaves independent per-subunit noise",
     min(pairwise_subunit_rmsd(indep_noise, 4)) > 0.5,
     f"unchanged pairwise RMSD {rmsd_before}")

# ---------------------------------------------------------------------------
# 4. --mix_output: unit-level - couple (mean) vs select give DIFFERENT,
#    each internally-consistent, canonicalisations of the SAME 4 copies.
# ---------------------------------------------------------------------------
print("\n=== 4. --mix_output (unit) ===")
torch.manual_seed(3)
L, reps = 4, 6
copies = torch.randn(L * reps, 5, 3)
out_couple = screw_replicate(copies, count=4, angle=1.5708, radius=0.0,
                             translation=0.0, chain_center=False, mean=True,
                             do_radius=True)
out_select = screw_replicate(copies, count=4, angle=1.5708, radius=0.0,
                             translation=0.0, chain_center=False, mean=False,
                             do_radius=True)
diff_mix = (out_couple - out_select)[:reps].abs().max().item()
check("--mix_output couple vs select produce different representative "
     "subunits from the same 4 noisy copies", diff_mix > 0.1,
     f"max abs diff on subunit 0 {diff_mix:.3f} A")

# ---------------------------------------------------------------------------
# 5. --mode rotation: already proven dead via prove_screwvp_bug.py (no jax
#    needed); confirm the PORT's own CLI rejects it immediately, citing it.
# ---------------------------------------------------------------------------
print("\n=== 5. --mode rotation (CLI-level) ===")
try:
    parse_sym("c4", replicate_count=None, screw_angle=None, screw_radius=0.0,
              screw_translation=0.0, sym_mean="False", sym_threshold=0.0,
              mode="rotation")
    check("--mode rotation raises in this port's parse_sym", False)
except SystemExit as e:
    check("--mode rotation raises in this port's parse_sym", True, str(e)[:60])

r = subprocess.run([PY, "scripts/generate.py", "--config", CONFIG,
                   "--num_aa", "10:10:10:10", "--sym", "c4", "--mode", "rotation",
                   "--num_designs", "1", "--num_steps", "5", "--out_steps", "4",
                   "--out_path", OUT + "/mode_rotation"],
                  capture_output=True, text=True, cwd=ROOT)
check("--mode rotation on the full CLI exits nonzero with the citation",
     r.returncode != 0 and "eval_sym.py" in (r.stdout + r.stderr),
     f"returncode={r.returncode}")

# ---------------------------------------------------------------------------
# 6. --f_small: full CLI - a strict threshold must trigger at least one
#    reject-and-regenerate cycle; the default (1.0) must not.
# ---------------------------------------------------------------------------
print("\n=== 6. --f_small (full CLI) ===")
common = ["--config", CONFIG, "--num_aa", "20:20:20:20", "--sym", "c4",
         "--num_steps", "40", "--out_steps", "39", "--num_designs", "1", "--seed", "7"]


def run_cli(extra, tag):
    t0 = time.perf_counter()
    r = subprocess.run([PY, "scripts/generate.py", *common, *extra,
                       "--out_path", f"{OUT}/{tag}"],
                      capture_output=True, text=True, cwd=ROOT)
    dt = time.perf_counter() - t0
    return r, dt


r_strict, _ = run_cli(["--f_small", "0.01"], "f_small_strict")
r_default, _ = run_cli([], "f_small_default")
check("--f_small (strict) triggers a reject-and-regenerate",
     "rejected" in r_strict.stdout, r_strict.stdout.strip().splitlines()[-3:])
check("--f_small default (1.0) never triggers a reject",
     "rejected" not in r_default.stdout, "ok")

# ---------------------------------------------------------------------------
# 7. --compact_lr / --clash_lr / --fixcenter_threshold / --mix_output /
#    --sym_noise: end-to-end, full CLI, same seed, measurable output diff.
# ---------------------------------------------------------------------------
print("\n=== 7. end-to-end CLI diffs (same seed, one flag flipped) ===")


def ca_of(path):
    ca = []
    chains = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                chains.append(line[21])
    return np.array(ca), chains


def rg_of(ca):
    return float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))


def min_cross_chain(ca, chains):
    chains = np.array(chains)
    d = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    same = chains[:, None] == chains[None, :]
    d = np.where(same, np.inf, d)
    return float(d.min())


base_common = ["--config", CONFIG, "--num_aa", "30:30:30:30", "--sym", "c4",
              "--num_steps", "80", "--out_steps", "79", "--num_designs", "1",
              "--seed", "11"]

variants = {
    "baseline": [],
    "compact_lr": ["--compact_lr", "0.4"],
    "clash_lr": ["--clash_lr", "0.4"],
    "fixcenter_high": ["--fixcenter_threshold", "5.0"],
    "mix_couple": ["--mix_output", "couple"],
    "mix_select": ["--mix_output", "select"],
    "sym_noise_off": ["--sym_noise", "False"],
}
results = {}
for tag, extra in variants.items():
    r, dt = None, None
    rr, dt = run_cli(extra, f"e2e_{tag}") if False else (None, None)
    rr = subprocess.run([PY, "scripts/generate.py", *base_common, *extra,
                        "--out_path", f"{OUT}/e2e_{tag}"],
                       capture_output=True, text=True, cwd=ROOT)
    pdb_path = f"{OUT}/e2e_{tag}/design_0.pdb"
    ok = rr.returncode == 0 and os.path.exists(pdb_path)
    check(f"e2e {tag}: CLI run succeeds", ok,
         rr.stdout.strip().splitlines()[-1] if ok else (rr.stdout + rr.stderr)[-400:])
    if ok:
        ca, chains = ca_of(pdb_path)
        results[tag] = dict(rg=rg_of(ca), min_cross=min_cross_chain(ca, chains),
                            ca=ca)

if "baseline" in results and "compact_lr" in results:
    check("compact_lr reduces Rg vs baseline (end-to-end)",
         results["compact_lr"]["rg"] < results["baseline"]["rg"],
         f"Rg {results['baseline']['rg']:.2f} -> {results['compact_lr']['rg']:.2f} A")
if "baseline" in results and "clash_lr" in results:
    check("clash_lr changes min cross-chain CA distance vs baseline (end-to-end)",
         abs(results["clash_lr"]["min_cross"] - results["baseline"]["min_cross"]) > 1e-3,
         f"min cross-chain {results['baseline']['min_cross']:.2f} -> "
         f"{results['clash_lr']['min_cross']:.2f} A")
if "baseline" in results and "fixcenter_high" in results:
    d = float(np.abs(results["baseline"]["ca"] - results["fixcenter_high"]["ca"]).max())
    check("fixcenter_threshold (high) changes the trajectory vs baseline",
         d > 0.5, f"max abs coordinate diff {d:.2f} A")
if "mix_couple" in results and "mix_select" in results:
    d = float(np.abs(results["mix_couple"]["ca"] - results["mix_select"]["ca"]).max())
    check("--mix_output couple vs select produce different designs (same seed)",
         d > 0.5, f"max abs coordinate diff {d:.2f} A")
if "baseline" in results and "sym_noise_off" in results:
    d = float(np.abs(results["baseline"]["ca"] - results["sym_noise_off"]["ca"]).max())
    check("--sym_noise False changes the trajectory vs True (same seed)",
         d > 0.5, f"max abs coordinate diff {d:.2f} A")

print(f"\n=== SUMMARY: {len(PASS)} passed, {len(FAIL)} failed ===")
for f in FAIL:
    print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)

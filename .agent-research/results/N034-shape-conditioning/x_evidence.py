"""N034 evidence: do --blobs designs actually follow the requested shape?

Not "the PDB exists" - per-blob centroid recovery + per-residue nearest-blob
classification, against a shuffled-centre control.

Usage:
    python x_evidence.py <out_dir> <blob_csv> <blob_scale> [--chunk 100]

Reads every design_*.pdb in <out_dir>, and for each:

  1. Splits CA coordinates into consecutive blocks of `chunk` (matching
     eval_ve_shape.py's hardcoded 100 - see notes/specs/08_shape_conditioning.md
     §2) and computes each block's centroid.
  2. Compares each block's centroid to that block's REQUESTED blob centre
     (row i of the CSV, scaled by --blob_scale) - "centroid recovery error".
  3. Classifies every individual residue's CA to its NEAREST of the K
     requested centres and reports the accuracy against the block it was
     actually generated for (row order = block order) - "residue placement
     accuracy" - plus the same classification against a RANDOMLY SHUFFLED
     assignment of centres to blocks, as a control: if the shuffled control's
     accuracy is near chance (1/K) while the real one is high, the real
     number reflects the design actually following the layout, not an
     artefact of the metric or of protein designs clustering near the origin
     in general.

Prints a table and a JSON summary (`<out_dir>/../evidence_<tag>.json`, next
to the PDBs, not inside them, so cleanup of `out_dir` does not delete it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np


def read_blob_centres(csv_path: str, blob_scale: float) -> np.ndarray:
    """Row order = block order (block i is the i-th 100-residue block)."""
    centres = []
    with open(csv_path) as fh:
        next(fh)   # header, unconditionally skipped - matches parse_blobs
        for line in fh:
            x, y, z, _num_aa = line.strip().split(",")
            centres.append([float(x), float(y), float(z)])
    return np.asarray(centres, dtype=np.float64) * blob_scale


def ca_coords(pdb_path: str) -> np.ndarray:
    ca = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(ca, dtype=np.float64)


def evaluate_design(ca: np.ndarray, centres: np.ndarray, chunk: int, *,
                    seed: int) -> dict:
    n = ca.shape[0]
    k = centres.shape[0]
    if n != k * chunk:
        raise ValueError(f"{n} CA atoms is not {k} blocks of {chunk} "
                         f"(k*chunk={k * chunk}) - chunk/CSV mismatch")

    blocks = ca.reshape(k, chunk, 3)
    block_centroids = blocks.mean(axis=1)                     # (k, 3)
    centroid_err = np.linalg.norm(block_centroids - centres, axis=-1)  # (k,)

    # per-residue nearest-centre classification against the REAL centres
    # (never permuted - permuting the ORDER of a fixed point set would not
    # change which one is spatially nearest, so that would be a no-op
    # control, not a real one).
    dists = np.linalg.norm(ca[:, None, :] - centres[None, :, :], axis=-1)  # (n, k)
    nearest = dists.argmin(axis=-1)                            # (n,)
    true_block = np.repeat(np.arange(k), chunk)
    true_acc = float((nearest == true_block).mean())

    # control: a WRONG (deranged) hypothesis for which centre each block
    # "should" belong to. If the design actually follows the CSV, the data
    # will support the true labelling far better than this wrong one - i.e.
    # `nearest` (computed once, above, from the real geometry) should almost
    # never equal the wrong label. This is a real negative control, unlike
    # comparing against a relabelled-but-otherwise-identical point set.
    rng = random.Random(seed)
    perm = list(range(k))
    if k > 1:
        is_derangement = False
        while not is_derangement:
            rng.shuffle(perm)
            is_derangement = all(perm[i] != i for i in range(k))
    wrong_block_label = np.asarray(perm)[true_block]
    shuf_acc = float((nearest == wrong_block_label).mean())

    return dict(n=n, k=k,
               centroid_err_mean=float(centroid_err.mean()),
               centroid_err_max=float(centroid_err.max()),
               centroid_err_per_block=[float(x) for x in centroid_err],
               placement_accuracy=true_acc,
               placement_accuracy_shuffled_control=shuf_acc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("blob_csv")
    ap.add_argument("blob_scale", type=float)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--summary_dir", default=None,
                    help="where evidence_<tag>.json goes (default: out_dir's "
                         "parent, so it survives an out_dir cleanup)")
    args = ap.parse_args(argv)

    centres = read_blob_centres(args.blob_csv, args.blob_scale)
    paths = sorted(glob.glob(os.path.join(args.out_dir, "design_*.pdb")))
    if not paths:
        print(f"[FAIL] no design_*.pdb in {args.out_dir}")
        return 1

    rows = []
    for i, p in enumerate(paths):
        ca = ca_coords(p)
        try:
            m = evaluate_design(ca, centres, args.chunk, seed=1000 + i)
        except ValueError as exc:
            print(f"[FAIL] {p}: {exc}")
            return 1
        m["path"] = p
        rows.append(m)

    print(f"\n{'design':>28} {'n':>5} {'centroid_err_mean':>18} "
          f"{'centroid_err_max':>17} {'placement_acc':>14} {'shuffled_ctrl':>14}")
    for r in rows:
        print(f"{os.path.basename(r['path']):>28} {r['n']:>5} "
              f"{r['centroid_err_mean']:>18.3f} {r['centroid_err_max']:>17.3f} "
              f"{r['placement_accuracy']:>14.3f} "
              f"{r['placement_accuracy_shuffled_control']:>14.3f}")

    agg = dict(
        n_designs=len(rows),
        centroid_err_mean=float(np.mean([r["centroid_err_mean"] for r in rows])),
        centroid_err_max=float(np.max([r["centroid_err_max"] for r in rows])),
        placement_accuracy_mean=float(np.mean([r["placement_accuracy"] for r in rows])),
        placement_accuracy_shuffled_control_mean=float(
            np.mean([r["placement_accuracy_shuffled_control"] for r in rows])),
        chance_accuracy=1.0 / centres.shape[0],
        rows=rows,
    )
    print(f"\naggregate over {agg['n_designs']} designs:")
    print(f"  centroid_err_mean = {agg['centroid_err_mean']:.3f} A "
          f"(max {agg['centroid_err_max']:.3f} A)")
    print(f"  placement_accuracy_mean = {agg['placement_accuracy_mean']:.3f} "
          f"vs shuffled-control {agg['placement_accuracy_shuffled_control_mean']:.3f} "
          f"vs chance {agg['chance_accuracy']:.3f}")

    summary_dir = args.summary_dir or os.path.dirname(os.path.normpath(args.out_dir))
    os.makedirs(summary_dir, exist_ok=True)
    out_json = os.path.join(summary_dir, f"evidence_{args.tag}.json")
    with open(out_json, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

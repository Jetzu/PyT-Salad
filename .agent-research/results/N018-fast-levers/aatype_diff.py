"""N018 - paired aatype divergence: lever vs `exact`, same seed, same design index.

The brief's sharpest signal: not a coordinate norm, a COUNT of residues that
take a different amino acid identity. This only works if the two runs are
PAIRED - same starting noise per design index - which holds here because
`FastDenoiser._blank_inputs` draws `uniform` as one `torch.rand(n_draws, L, L)`
call whose SHAPE depends only on the config (`uniform_draw_count`), never on
the neighbourhood size K: shrinking K changes which indices `topk`/`argsort`
keeps, not how many randoms are drawn or in what order. So `--seed 42` should
draw the SAME random field for `fast_halfnbr`/`fast_randhalf` as it did for
N015's `a_exact` (K=64), design index for design index.

This is verified empirically below, not just argued: report the per-design
diff count distribution. If pairing had silently broken (e.g. a hidden extra
draw whose count depends on K), the diff count would sit at the ~95% two
sequences of i.i.d. draws from 20 amino acids would show by chance - the
script flags that explicitly.

Usage:  python aatype_diff.py <lever_dir1>[,<lever_dir2>] <exact_dir1>[,<exact_dir2>]
Directories are matched by SORTED design_N.pdb filename across the comma list
(so halfnbr_s42+halfnbr_s1234 lines up with a_exact(s42)+a_exact(s1234) in
seed order, same convention as N015's stats.py pooling).
"""
from __future__ import annotations

import glob
import os
import sys

THREE_TO_ONE = dict(
    ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G",
    HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S",
    THR="T", TRP="W", TYR="Y", VAL="V",
)


def read_seq(path):
    seq, seen = [], set()
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            resseq = line[22:26]
            if resseq in seen:
                continue
            seen.add(resseq)
            resname = line[17:20].strip()
            seq.append(THREE_TO_ONE.get(resname, "X"))
    return "".join(seq)


def load_dirs(spec):
    paths = []
    for d in spec.split(","):
        paths += sorted(glob.glob(os.path.join(d, "design_*.pdb")),
                        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return paths


def main(lever_spec, exact_spec):
    lever_paths = load_dirs(lever_spec)
    exact_paths = load_dirs(exact_spec)
    n = min(len(lever_paths), len(exact_paths))
    if len(lever_paths) != len(exact_paths):
        print(f"WARNING: {len(lever_paths)} lever designs vs {len(exact_paths)} "
              f"exact designs - comparing the first {n} by index only")
    diffs, lens = [], []
    unknown_frac = []
    for lp, ep in zip(lever_paths[:n], exact_paths[:n]):
        ls, es = read_seq(lp), read_seq(ep)
        L = min(len(ls), len(es))
        if len(ls) != len(es):
            print(f"  {os.path.basename(lp)}: LENGTH MISMATCH {len(ls)} vs {len(es)} "
                  f"- design likely diverged/truncated, comparing first {L}")
        d = sum(1 for a, b in zip(ls[:L], es[:L]) if a != b)
        diffs.append(d)
        lens.append(L)
        unknown_frac.append((ls.count("X") + es.count("X")) / max(1, 2 * L))

    import statistics as st
    frac = [d / L for d, L in zip(diffs, lens) if L]
    print(f"\nn={len(diffs)} designs  mean diff {st.mean(diffs):.2f} / {st.mean(lens):.1f} "
          f"residues  ({st.mean(frac) * 100:.1f}% +- {st.stdev(frac) * 100:.1f}% if n>1)"
          if len(diffs) > 1 else "")
    print(f"per-design diff counts: {diffs}")
    print(f"per-design lengths:     {lens}")
    chance = 19 / 20  # a random re-draw of a categorical-20 residue differs 95% of the time
    print(f"\nchance level for an UNPAIRED (independent) redraw: ~{chance*100:.0f}% of residues differ.")
    print("If the mean fraction above is near that, the seeds are NOT actually paired "
          "and this comparison should be read as unpaired only.")
    if any(u > 0.01 for u in unknown_frac):
        print(f"WARNING: unparsed residue names present (mapped to 'X') in "
              f"{sum(1 for u in unknown_frac if u > 0.01)} design(s) - check THREE_TO_ONE coverage")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])

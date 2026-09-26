"""N013 - compare two `x_floor.py` rep-sets: is arm B's perturbation inside
arm A's own run-to-run spread?

Three questions, in increasing strength:

  1. is the CROSS distribution (every A_i vs every B_j) distinguishable from
     the WITHIN distributions?  - the bar the brief asks for
  2. is every B output BIT-IDENTICAL to some A output?  - much stronger, and
     meaningful because the noise is a DISCRETE tie-flip with few attractors
  3. does B visit any bit-pattern A never produced, and vice versa?
"""
import sys, torch

A = torch.load(sys.argv[1], weights_only=False)
B = torch.load(sys.argv[2], weights_only=False)
key = sys.argv[3] if len(sys.argv) > 3 else "pos"
print(f"A: {A['meta']['tag']} L={A['meta']['length']} IND={A['meta']['ind']!r} "
      f"DET={A['meta']['det']!r} input={A['meta']['input']}")
print(f"B: {B['meta']['tag']} L={B['meta']['length']} IND={B['meta']['ind']!r} "
      f"DET={B['meta']['det']!r} input={B['meta']['input']}")
xa = torch.stack([o[key] for o in A["outs"]]).double()
xb = torch.stack([o[key] for o in B["outs"]]).double()
na, nb = xa.shape[0], xb.shape[0]


def pw(x, y, same):
    d = []
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if same and j <= i:
                continue
            d.append((x[i] - y[j]).abs().max().item())
    return torch.tensor(d)


def desc(name, d):
    s, _ = d.sort()
    print(f"  {name:<14} n={len(d):<5} zero={int((d == 0).sum()):<5} "
          f"mean {d.mean():.3e}  median {d.median():.3e}  "
          f"p95 {s[int(0.95 * (len(s) - 1))]:.3e}  max {d.max():.3e}")
    return d


print(f"\n[{key}] pairwise max|d|")
da = desc("within-A", pw(xa, xa, True))
db = desc("within-B", pw(xb, xb, True))
dc = desc("cross A-B", pw(xa, xb, False))


def clusters(dump):
    out = {}
    for i, o in enumerate(dump["outs"]):
        out.setdefault(hash(o[key].cpu().numpy().tobytes()), []).append(i)
    return out


sa, sb = clusters(A), clusters(B)
shared = set(sa) & set(sb)
print(f"\n  distinct bit-patterns: A {len(sa)}  B {len(sb)}  shared {len(shared)}")
print(f"  A cluster sizes {sorted((len(v) for v in sa.values()), reverse=True)}")
print(f"  B cluster sizes {sorted((len(v) for v in sb.values()), reverse=True)}")
cov_b = sum(len(v) for k, v in sb.items() if k in shared)
cov_a = sum(len(v) for k, v in sa.items() if k in shared)
print(f"  B reps landing on an A pattern: {cov_b}/{nb}    "
      f"A reps landing on a B pattern: {cov_a}/{na}")
verdict = ("INSIDE the floor (cross <= within-A max)"
           if dc.max() <= max(da.max().item(), db.max().item())
           else "ABOVE the floor")
print(f"\n  cross max {dc.max():.3e}  vs within max {max(da.max().item(), db.max().item()):.3e}"
      f"  -> {verdict}")

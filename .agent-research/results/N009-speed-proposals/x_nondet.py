"""Is the CUDA denoiser reproducible AT ALL? (N009)

`e2e.sbatch` found that two runs of the SAME command with the SAME --seed
produce different proteins. That voids "the written PDB is unchanged" as a test
of any optimisation, so the source has to be named before anything else is
believed.

Prime suspect: `get_neighbours` chains its stages with
    distance[index[:, None], neighbours] = update
and a row that contains the -1 sentinel writes to column N-1 TWICE - once with
`+inf` (if N-1 is a genuine neighbour of that row) and once with the original
finite distance (from the -1 slot). That is notes/04 section 4's B1 bug. On
CUDA, `index_put_` with duplicate indices is documented NON-DETERMINISTIC, so
which of the two writes lands is decided by the scheduler.

    python x_nondet.py --device cuda
"""
import argparse, sys
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
import torch
from salad_torch.geometry.neighbours import (get_neighbours, get_index_neighbours,
                                             get_spatial_neighbours, pair_distance)
from salad_torch.geometry.segment import index_mean

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
ap.add_argument("--length", type=int, default=100); ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()
torch.set_grad_enabled(False)
dev = torch.device(a.device); L = a.length
g = torch.Generator().manual_seed(3)
pos = (torch.randn(L, 3, generator=g) * 12).to(dev)
batch = torch.zeros(L, dtype=torch.long, device=dev)
mask = torch.ones(L, dtype=torch.bool, device=dev)

print(f"device={dev}  L={L}  reps={a.reps}")

# 1. index_put_ with duplicate indices, which is what stage chaining does
d = pair_distance(pos)
idx = torch.arange(L, device=dev)
nbr = get_spatial_neighbours(16)(pos, batch, mask)
# force -1 sentinels into the row, as the all-inf block-contact stage does
nbr = torch.cat((nbr, torch.full((L, 16), -1, dtype=torch.long, device=dev)), dim=-1)
rows_with_last = int(((nbr == L - 1).any(-1) & (nbr == -1).any(-1)).sum())
print(f"rows that index BOTH a real N-1 and a -1 sentinel: {rows_with_last}/{L}"
      f"   (these are the duplicate-write rows)")
outs = []
for _ in range(a.reps):
    dd = torch.where(mask[:, None] & mask[None, :], d, torch.full_like(d, float("inf"))).clone()
    upd = torch.where(nbr != -1, torch.full_like(dd[idx[:, None], nbr], float("inf")),
                      dd[idx[:, None], nbr])
    dd[idx[:, None], nbr] = upd
    outs.append(dd.clone())
same = all(torch.equal(outs[0], o) for o in outs[1:])
dmaxv = max((outs[0] - o).abs().max().item() for o in outs[1:]) if len(outs) > 1 else 0.0
print(f"index_put_ chaining reproducible over {a.reps} reps: {same}   max|d|={dmaxv}")

# 2. does the SELECTED NEIGHBOUR SET change?
sel = [get_neighbours(32)(d, mask[:, None] & mask[None, :], nbr) for _ in range(a.reps)]
same_sel = all(torch.equal(sel[0], s) for s in sel[1:])
ndiff = max(int((sel[0] != s).sum()) for s in sel[1:]) if len(sel) > 1 else 0
print(f"selected neighbour indices reproducible: {same_sel}   worst differing slots: {ndiff}")

# 3. index_mean (scatter_add / atomicAdd), used by every Update
x = torch.randn(L, 512, device=dev)
ms = [index_mean(x, batch, mask[..., None]) for _ in range(a.reps)]
same_m = all(torch.equal(ms[0], v) for v in ms[1:])
dm = max((ms[0] - v).abs().max().item() for v in ms[1:]) if len(ms) > 1 else 0.0
print(f"index_mean reproducible: {same_m}   max|d|={dm:.3e}")

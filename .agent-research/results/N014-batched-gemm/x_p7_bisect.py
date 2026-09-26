"""N014 - bisect the batched prologue against six sequential `pair_2` calls.

`x_p7_equiv2.py` found a whole-forward divergence at chains=4 + batches=2 on
CPU, where everything else was bit-exact. This compares the PROLOGUE itself, so
the difference is attributed to a tensor rather than to the chaos below it.
"""
import argparse, sys
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
import torch                                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=64)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--batches", type=int, default=2)
ap.add_argument("--device", default="cpu")
ap.add_argument("--pair-cond", action="store_true")
a = ap.parse_args()
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
from salad_torch.config import get_config                          # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict    # noqa: E402
from salad_torch.geometry import features as GF                    # noqa: E402

dev = torch.device(a.device)
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/%s-200k.npz" % a.config),
           cfg, strict=True)
stack = m.diffusion.diffusion_stack
sp = stack.cond_pair_stack
L, A = a.length, 5 + cfg.augment_size
g = torch.Generator(device="cpu").manual_seed(11)
prev_pos = (torch.randn(L, A, 3, generator=g) * 4).to(dev)
resi = torch.arange(L, device=dev)
chain = (torch.arange(L, device=dev) * a.chains // L).long()
batch = (torch.arange(L, device=dev) * a.batches // L).long()
mask = torch.ones(L, dtype=torch.bool, device=dev)
print("L=%d chains=%d batches=%d depth=%d" % (L, a.chains, a.batches, sp.depth))
print("  chain ids %s   batch ids %s" % (chain.unique().tolist(), batch.unique().tolist()))

pc = None
if a.pair_cond:
    flags = torch.cat(((chain[:, None] != chain[None, :]).float()[..., None],
                       torch.zeros(L, L, 2, device=dev)), dim=-1)
    pc = dict(dmap=torch.zeros(L, L, device=dev),
              dmap_mask=torch.zeros(L, L, dtype=torch.bool, device=dev),
              flags=flags, omap=torch.zeros(L, L, 9, device=dev))

K = int(__import__("os").environ.get("BISECT_K", "48"))
nbr = torch.stack([torch.randint(-1, L, (L, K), generator=g).to(dev)
                   for _ in range(sp.depth)])

seq_pair, seq_mask = [], []
for i, blk in enumerate(stack.blocks):
    p, pm = blk.pair_2(prev_pos, pc, nbr[i], resi, chain, batch, mask)
    seq_pair.append(p); seq_mask.append(pm)
seq_pair = torch.stack(seq_pair); seq_mask = torch.stack(seq_mask)
bat_pair, bat_mask = sp(prev_pos, pc, nbr, resi, chain, batch, mask)

print("\nPROLOGUE OUTPUT  pair  equal=%s  max|d|=%.6e"
      % (torch.equal(seq_pair, bat_pair),
         (seq_pair.double() - bat_pair.double()).abs().max()))
print("PROLOGUE OUTPUT  mask  equal=%s" % torch.equal(seq_mask, bat_mask))
for i in range(sp.depth):
    d = (seq_pair[i].double() - bat_pair[i].double()).abs().max().item()
    km = seq_mask[i][..., None]
    dk = ((seq_pair[i].double() - bat_pair[i].double()).abs() * km).max().item()
    print("  block %d  all-lanes %.6e   kept-lanes %.6e   mask equal=%s"
          % (i, d, dk, torch.equal(seq_mask[i], bat_mask[i])))

b, n, k = nbr.shape
off = (torch.arange(b, device=dev) * n)[:, None, None]
nb = torch.where(nbr >= 0, nbr + off, off + (n - 1)).reshape(b * n, k)
resi_t, chain_t = resi.repeat(b), (chain[None] + off[:, :, 0]).reshape(-1)
batch_t = (batch[None] + off[:, :, 0]).reshape(-1)
rp_b = GF.sequence_relative_position(32, one_hot=False, cyclic=False)(
    resi_t, chain_t, batch_t, nb).reshape(b, n, k)
bad = 0
for i in range(b):
    rp_s = GF.sequence_relative_position(32, one_hot=False, cyclic=False)(
        resi, chain, batch, nbr[i])
    ne = int((rp_s != rp_b[i]).sum())
    bad += ne
    if ne:
        w = (rp_s != rp_b[i]).nonzero()[:5]
        print("  relpos block %d: %d/%d bins DIFFER, e.g. %s"
              % (i, ne, n * k,
                 [(int(r), int(c), int(rp_s[r, c]), int(rp_b[i][r, c])) for r, c in w]))
print("\nRELPOS BINS total differing: %d/%d" % (bad, b * n * k))
print("  tiled chain ids min/max %d/%d   tiled batch ids min/max %d/%d   n=%d  b*n=%d"
      % (int(chain_t.min()), int(chain_t.max()), int(batch_t.min()),
         int(batch_t.max()), n, b * n))

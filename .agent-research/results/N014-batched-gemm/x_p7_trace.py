"""N014 - where does the chains=4 + batches=2 whole-forward divergence enter?

The prologue itself is bit-exact for that shape (x_p7_bisect.py), so this taps
`DiffusionBlock.forward` and compares, per block and in order, the tensors the
two arms actually feed the attention: `cond_neighbours` and the `pair`/`pair_mask`
the block ends up using.
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
ap.add_argument("--mask-frac", type=float, default=1.0)
a = ap.parse_args()
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
from salad_torch import opt                                        # noqa: E402
from salad_torch.config import get_config                          # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict    # noqa: E402
from salad_torch.modules.diffusion import uniform_draw_count       # noqa: E402
import salad_torch.modules.blocks as B                             # noqa: E402

dev = torch.device("cpu")
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval()
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/%s-200k.npz" % a.config),
           cfg, strict=True)
D = m.diffusion
L, A = a.length, 5 + cfg.augment_size
g = torch.Generator(device="cpu").manual_seed(7)
rnd = lambda *sh: torch.randn(*sh, generator=g)
chain = (torch.arange(L) * a.chains // L).long()
bidx = (torch.arange(L) * a.batches // L).long()
mk = torch.ones(L, dtype=torch.bool)
if a.mask_frac < 1.0:
    mk[int(L * a.mask_frac):] = False
pos = rnd(L, A, 3) * 4
u = torch.rand(uniform_draw_count(cfg), L, L, generator=g).clamp_(1e-6, 1 - 1e-6)
flags = torch.cat(((chain[:, None] != chain[None, :]).float()[..., None],
                   torch.zeros(L, L, 2)), dim=-1)
ARGS = dict(pos_noised=pos, prev_pos=rnd(L, A, 3) * 4, prev_local=rnd(L, cfg.local_size),
            condition=rnd(L, cfg.local_size), dmap=torch.zeros(L, L),
            omap=torch.zeros(L, L, 9), dmap_mask=torch.zeros(L, L, dtype=torch.bool),
            flags=flags, t_pos=torch.full((L,), 0.5), t_seq=torch.ones(L),
            seq_noised=torch.full((L,), 20, dtype=torch.long),
            aa_gt=torch.full((L,), 20, dtype=torch.long), pos_gt=pos,
            resi=torch.arange(L), chain=chain, batch=bidx, mask=mk,
            cyclic_mask=None, uniform=u)
ORDER = ("pos_noised prev_pos prev_local condition dmap omap dmap_mask flags t_pos "
         "t_seq seq_noised aa_gt pos_gt resi chain batch mask cyclic_mask uniform").split()
pa = [ARGS[k] for k in ORDER]

REC = []
_orig = B.DiffusionBlock.forward


def traced(self, local, pos, prev_pos, condition, pair_condition, time, resi,
           chain, batch, mask, current_neighbours, cond_neighbours,
           cyclic_mask=None, init_pos=None, cond_pair=None):
    rec = dict(cur=current_neighbours.clone(), cond=cond_neighbours.clone(),
               local_in=local.clone(), pos_in=pos.clone())
    if cond_pair is not None:
        rec["pair"] = cond_pair[0].clone(); rec["pmask"] = cond_pair[1].clone()
    else:
        p, pm = self.pair_2(prev_pos, pair_condition, cond_neighbours, resi, chain,
                            batch, mask, cyclic_mask=cyclic_mask, init_pos=init_pos)
        rec["pair"] = p.clone(); rec["pmask"] = pm.clone()
    out = _orig(self, local, pos, prev_pos, condition, pair_condition, time, resi,
                chain, batch, mask, current_neighbours, cond_neighbours,
                cyclic_mask=cyclic_mask, init_pos=init_pos, cond_pair=cond_pair)
    rec["local_out"], rec["pos_out"] = out[0].clone(), out[1].clone()
    REC.append(rec)
    return out


B.DiffusionBlock.forward = traced


def run(flag):
    global REC
    REC = []
    opt.BATCH_COND = flag
    out = D.forward_core(*pa, True, False, True)
    return out, REC


(o0, r0) = run(False)
(o1, r1) = run(True)
opt.BATCH_COND = False
print("arm A blocks=%d  arm B blocks=%d" % (len(r0), len(r1)))
for i, (x, y) in enumerate(zip(r0, r1)):
    line = "block %d" % i
    for k in ("cur", "cond", "local_in", "pos_in", "pair", "pmask",
              "local_out", "pos_out"):
        eq = torch.equal(x[k], y[k])
        if x[k].dtype.is_floating_point:
            d = (x[k].double() - y[k].double()).abs().max().item()
            line += "  %s=%s(%.1e)" % (k, "T" if eq else "F", d)
        else:
            nd = int((x[k] != y[k]).sum())
            line += "  %s=%s(%d)" % (k, "T" if eq else "F", nd)
    print(line)
print("final pos equal=%s  max|d|=%.6e"
      % (torch.equal(o0[0], o1[0]), (o0[0].double() - o1[0].double()).abs().max()))

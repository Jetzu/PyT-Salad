"""P-R: the condition branch's pair features for all 6 blocks depend only on
prev_pos (loop-invariant) and each block's cond_neighbours, so they can be
computed in ONE batched prologue: 6x fewer kernel launches, weights taken
straight from the checkpoint's native (6, in, 64) layer-stack layout.

usage: x_batch6.py [--length 100]
"""
import argparse, sys, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure")
from common import build, make_data, bench
import salad_torch.geometry.features as GF
from salad_torch.modules.blocks import extract_condition_neighbours
from salad_torch.modules.basic import gelu

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=100)
a = ap.parse_args()
L = a.length
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
dev = torch.device("cuda")
m, cfg = build()
d, _ = make_data(cfg, L)
resi, chain, batch, mask = (d["residue_index"], d["chain_index"],
                            d["batch_index"], d["mask"])
PREV = torch.randn(L, 20, 3, device=dev) * 4
uni = torch.rand(20, L, L, device=dev).clamp_(1e-6, 1 - 1e-6)
pcz = {"dmap": torch.zeros(L, L, device=dev),
       "dmap_mask": torch.zeros(L, L, dtype=torch.bool, device=dev),
       "flags": torch.zeros(L, L, 3, device=dev)}
with torch.no_grad():
    NBR = [extract_condition_neighbours(16, 16, 16, 16)(
        PREV, pcz, resi, chain, batch, mask,
        uniform=uni[1 + 3 * b + 1], uniform2=uni[1 + 3 * b + 2]) for b in range(6)]
K = NBR[0].shape[1]
mods = [blk.pair_2 for blk in m.diffusion.diffusion_stack.blocks]
C = lambda f: torch.compile(f, mode="reduce-overhead", dynamic=False)


def timed(name, fn, ref=None):
    with torch.no_grad():
        t = bench(lambda: (torch.compiler.cudagraph_mark_step_begin(), fn()),
                  iters=50, warm=10)
        out = fn().clone()
    extra = ""
    if ref is not None:
        extra = (f"   max|d| {(out - ref).abs().max().item():.3e}"
                 f"   bitexact {bool(torch.equal(out, ref))}")
    print(f"  {name:<40} {t:7.3f} ms{extra}")
    return t, out


def v0():
    out = []
    for mod, nbr in zip(mods, NBR):
        idx = torch.arange(L, device=dev)
        g = {k: v[idx[:, None], nbr] for k, v in pcz.items()}
        pair = mod.linear_relpos(GF.sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, nbr))
        pair = pair + mod.linear_dmap(torch.where(
            g["dmap_mask"][..., None], GF.distance_rbf(g["dmap"]),
            torch.zeros((), device=dev)))
        pair = pair + mod.linear_flags(torch.where(
            g["flags"].any(-1)[..., None], g["flags"], torch.zeros_like(g["flags"])))
        pair = pair + mod.linear_dist(GF.distance_features(PREV, nbr, 0.0, 22.0))
        pair = pair + mod.linear_dir(GF.direction_features(PREV, nbr))
        pair = pair + mod.linear_rot(GF.position_rotation_features(PREV, nbr))
        pair = pair + mod.linear_pairvec(GF.pair_vector_features(PREV, nbr))
        out.append(mod.mlp(mod.layer_norm(pair)))
    return torch.stack(out)


# --- load-time stacked weights: exactly the checkpoint's native layout ---
def stack(attr, sub=None):
    ws = []
    for mod in mods:
        lin = getattr(mod, attr) if sub is None else getattr(mod, attr).layers[sub]
        ws.append(lin.w)
    return torch.stack(ws).detach().contiguous()


W = {k: stack(f"linear_{k}") for k in
     ("relpos", "dmap", "flags", "dist", "dir", "rot", "pairvec")}
W0 = stack("mlp", 0)
W1 = stack("mlp", 1)
B0 = torch.stack([mo.mlp.layers[0].b for mo in mods]).detach().contiguous()
B1 = torch.stack([mo.mlp.layers[1].b for mo in mods]).detach().contiguous()
LNS = torch.stack([mo.layer_norm.scale for mo in mods]).detach().contiguous()
LNO = torch.stack([mo.layer_norm.offset for mo in mods]).detach().contiguous()
# tiled positions so the unmodified feature builders see one flat residue axis
POS6 = PREV.repeat(6, 1, 1).contiguous()
RESI6 = resi.repeat(6)
CH6 = torch.cat([chain + b for b in range(6)])
BA6 = torch.cat([batch + b for b in range(6)])
off = (torch.arange(6, device=dev) * L)[:, None, None]
NBR6 = torch.stack(NBR)
NBR6F = torch.where(NBR6 >= 0, NBR6 + off, off + L - 1).reshape(6 * L, K)
IDX6 = torch.arange(6 * L, device=dev)
DMAP6 = torch.zeros(6 * L, 6 * L, device=dev)
DMASK6 = torch.zeros(6 * L, 6 * L, dtype=torch.bool, device=dev)
FLAGS6 = torch.zeros(6 * L, 6 * L, 3, device=dev)


def v1():
    g_dmap = DMAP6[IDX6[:, None], NBR6F]
    g_mask = DMASK6[IDX6[:, None], NBR6F]
    g_flags = FLAGS6[IDX6[:, None], NBR6F]
    f = dict(
        relpos=GF.sequence_relative_position(32, one_hot=True)(
            RESI6, CH6, BA6, NBR6F),
        dmap=torch.where(g_mask[..., None], GF.distance_rbf(g_dmap),
                         torch.zeros((), device=dev)),
        flags=torch.where(g_flags.any(-1)[..., None], g_flags,
                          torch.zeros_like(g_flags)),
        dist=GF.distance_features(POS6, NBR6F, 0.0, 22.0),
        dir=GF.direction_features(POS6, NBR6F),
        rot=GF.position_rotation_features(POS6, NBR6F),
        pairvec=GF.pair_vector_features(POS6, NBR6F))
    pair = None
    for k in ("relpos", "dmap", "flags", "dist", "dir", "rot", "pairvec"):
        x = f[k].reshape(6, L * K, -1)
        t = torch.bmm(x, W[k])
        pair = t if pair is None else pair + t
    mu = pair.mean(-1, keepdim=True)
    var = pair.var(-1, unbiased=False, keepdim=True)
    pair = (pair - mu) * torch.rsqrt(var + 1e-5) * LNS[:, None] + LNO[:, None]
    pair = gelu(torch.bmm(pair, W0) + B0[:, None])
    pair = torch.bmm(pair, W1) + B1[:, None]
    return pair.reshape(6, L, K, -1)


print(f"condition-branch pair features x6 blocks, L={L}, K={K}")
t0, r0 = timed("v0 sequential (current)", C(v0))
t1, r1 = timed("v1 batched bmm prologue", C(v1), ref=r0)
print(f"     -> {(t0-t1)/t0*100:+.1f}% vs v0   ({t0-t1:+.3f} ms/fwd)")

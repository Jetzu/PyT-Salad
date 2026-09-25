"""Where does the 2.0 ms of pair-feature work go? Feature construction vs GEMMs.

usage: x_pairsplit.py [--length 100]
"""
import argparse, sys, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure")
from common import build, make_data, bench, kernel_stats
import salad_torch.geometry.features as GF
from salad_torch.geometry.neighbours import extract_neighbours

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=100)
a = ap.parse_args()
L = a.length
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
dev = torch.device("cuda")
m, cfg = build()
d, p = make_data(cfg, L)
resi, chain, batch, mask = (d["residue_index"], d["chain_index"],
                            d["batch_index"], d["mask"])
POS = [torch.randn(L, 20, 3, device=dev) * 4 for _ in range(6)]
PREV = torch.zeros(L, 20, 3, device=dev)
uni = torch.rand(20, L, L, device=dev).clamp_(1e-6, 1 - 1e-6)
pcz = {"dmap": torch.zeros(L, L, device=dev),
       "dmap_mask": torch.zeros(L, L, dtype=torch.bool, device=dev),
       "flags": torch.zeros(L, L, 3, device=dev)}
mods = []
for b in m.diffusion.diffusion_stack.blocks:
    mods += [b.pair_1, b.pair_2]
with torch.no_grad():
    NBR = [extract_neighbours(16, 16, 32)(POS[i], resi, chain, batch, mask,
                                          uniform=uni[i + 1]) for i in range(6)]
nbrs, poss = [], []
for i in range(6):
    nbrs += [NBR[i], NBR[i]]
    poss += [POS[i], PREV]
C = lambda f: torch.compile(f, mode="reduce-overhead", dynamic=False)


def timed(name, fn, base=None):
    with torch.no_grad():
        t = bench(lambda: (torch.compiler.cudagraph_mark_step_begin(), fn()),
                  iters=50, warm=10)
    rel = f"   ({t/base*100:5.1f}% of full)" if base else ""
    print(f"  {name:<40} {t:7.3f} ms{rel}")
    return t


def full():
    out = []
    for mod, pos, nbr in zip(mods, poss, nbrs):
        idx = torch.arange(pos.shape[0], device=pos.device)
        g = {k: v[idx[:, None], nbr] for k, v in pcz.items()}
        pair = mod.linear_relpos(GF.sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, nbr))
        pair = pair + mod.linear_dmap(torch.where(
            g["dmap_mask"][..., None], GF.distance_rbf(g["dmap"]),
            torch.zeros((), device=dev)))
        pair = pair + mod.linear_flags(torch.where(
            g["flags"].any(-1)[..., None], g["flags"], torch.zeros_like(g["flags"])))
        pair = pair + mod.linear_dist(GF.distance_features(pos, nbr, 0.0, 22.0))
        pair = pair + mod.linear_dir(GF.direction_features(pos, nbr))
        pair = pair + mod.linear_rot(GF.position_rotation_features(pos, nbr))
        pair = pair + mod.linear_pairvec(GF.pair_vector_features(pos, nbr))
        out.append(mod.mlp(mod.layer_norm(pair)))
    return torch.stack(out)


# cached feature tensors -> isolates GEMM + add + LN + MLP
with torch.no_grad():
    CACHE = []
    for mod, pos, nbr in zip(mods, poss, nbrs):
        idx = torch.arange(pos.shape[0], device=pos.device)
        g = {k: v[idx[:, None], nbr] for k, v in pcz.items()}
        CACHE.append(dict(
            relpos=GF.sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, nbr),
            dmap=torch.where(g["dmap_mask"][..., None], GF.distance_rbf(g["dmap"]),
                             torch.zeros((), device=dev)),
            flags=torch.where(g["flags"].any(-1)[..., None], g["flags"],
                              torch.zeros_like(g["flags"])),
            dist=GF.distance_features(pos, nbr, 0.0, 22.0),
            dir=GF.direction_features(pos, nbr),
            rot=GF.position_rotation_features(pos, nbr),
            pairvec=GF.pair_vector_features(pos, nbr)))


def gemms_only():
    out = []
    for mod, f in zip(mods, CACHE):
        pair = mod.linear_relpos(f["relpos"])
        pair = pair + mod.linear_dmap(f["dmap"])
        pair = pair + mod.linear_flags(f["flags"])
        pair = pair + mod.linear_dist(f["dist"])
        pair = pair + mod.linear_dir(f["dir"])
        pair = pair + mod.linear_rot(f["rot"])
        pair = pair + mod.linear_pairvec(f["pairvec"])
        out.append(mod.mlp(mod.layer_norm(pair)))
    return torch.stack(out)


def gemms_geom_only():   # drop relpos/dmap/flags GEMMs, keep the 4 geometric ones
    out = []
    for mod, f in zip(mods, CACHE):
        pair = mod.linear_dist(f["dist"])
        pair = pair + mod.linear_dir(f["dir"])
        pair = pair + mod.linear_rot(f["rot"])
        pair = pair + mod.linear_pairvec(f["pairvec"])
        out.append(mod.mlp(mod.layer_norm(pair)))
    return torch.stack(out)


def mk_feat(which):
    def fn():
        acc = None
        for pos, nbr in zip(poss, nbrs):
            if which == "dist":
                x = GF.distance_features(pos, nbr, 0.0, 22.0)
            elif which == "dir":
                x = GF.direction_features(pos, nbr)
            elif which == "rot":
                x = GF.position_rotation_features(pos, nbr)
            elif which == "pairvec":
                x = GF.pair_vector_features(pos, nbr)
            elif which == "relpos":
                x = GF.sequence_relative_position(32, one_hot=True)(
                    resi, chain, batch, nbr)
            s = x.sum()
            acc = s if acc is None else acc + s
        return acc
    return fn


print(f"pair-feature block x12, L={L}, K=64")
tf = timed("full (features + GEMMs + MLP)", C(full))
tg = timed("GEMMs+LN+MLP on cached features", C(gemms_only), tf)
timed("  same, only the 4 geometric GEMMs", C(gemms_geom_only), tf)
print(f"  => feature construction ~= {tf - tg:.3f} ms "
      f"({(tf-tg)/tf*100:.0f}%), projections ~= {tg:.3f} ms ({tg/tf*100:.0f}%)")
for w in ("relpos", "dist", "dir", "rot", "pairvec"):
    timed(f"build {w} only x12", C(mk_feat(w)), tf)

with torch.no_grad():
    cf = C(full)
    us, n, rows = kernel_stats(lambda: (torch.compiler.cudagraph_mark_step_begin(),
                                        cf())[1])
print(f"  full: GPU busy {us:.0f} us in {n:.0f} kernels "
      f"({n/12:.1f} per pair block, {us/n:.2f} us mean)")
for t_us, cnt, key in rows[:12]:
    print(f"   {t_us:8.1f} us  x{cnt:6.1f}  {key[:66]}")

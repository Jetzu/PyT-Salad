"""Isolated microbenchmarks for the N007 restructure proposals.

usage: x_micro.py <diag|pairfuse|neigh|ipa> [--length 100]
"""
import argparse, math, sys, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure")
from common import build, make_data, bench
import salad_torch.geometry.features as GF
from salad_torch.geometry.neighbours import (get_index_neighbours,
                                             get_neighbours,
                                             get_random_neighbours,
                                             get_spatial_neighbours,
                                             extract_neighbours, pair_distance)
from salad_torch.modules.blocks import extract_condition_neighbours

ap = argparse.ArgumentParser()
ap.add_argument("exp")
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
pc = {"dmap": torch.zeros(L, L, device=dev),
      "dmap_mask": torch.zeros(L, L, dtype=torch.bool, device=dev),
      "flags": torch.zeros(L, L, 3, device=dev),
      "omap": torch.zeros(L, L, 9, device=dev)}
uni = torch.rand(20, L, L, device=dev).clamp_(1e-6, 1 - 1e-6)
# six "positions", one per block, all protein-scale
POS = [torch.randn(L, 20, 3, device=dev) * 4 for _ in range(6)]
PREV = torch.zeros(L, 20, 3, device=dev)

C = lambda f: torch.compile(f, mode="reduce-overhead", dynamic=False)


def timed(name, fn, ref=None):
    with torch.no_grad():
        t = bench(lambda: (torch.compiler.cudagraph_mark_step_begin(), fn()),
                  iters=50, warm=10)
        out = fn()
        out = out.clone() if torch.is_tensor(out) else tuple(o.clone() for o in out)
    extra = ""
    if ref is not None:
        o = out if torch.is_tensor(out) else out[0]
        r = ref if torch.is_tensor(ref) else ref[0]
        extra = f"   max|d| {(o.float()-r.float()).abs().max().item():.3e}"
    print(f"  {name:<34} {t:7.3f} ms{extra}")
    return t, out


# ------------------------------------------------------------------ diag
if a.exp == "diag":
    with torch.no_grad():
        cur = extract_neighbours(16, 16, 32)(POS[0], resi, chain, batch, mask,
                                             uniform=uni[1])
        cond = extract_condition_neighbours(16, 16, 16, 16)(
            PREV, pc, resi, chain, batch, mask, uniform=uni[2], uniform2=uni[3])
    print(f"L={L}: current_neighbours {tuple(cur.shape)}  "
          f"-1 entries {(cur == -1).sum().item()} / {cur.numel()}")
    print(f"L={L}: cond_neighbours    {tuple(cond.shape)}  "
          f"-1 entries {(cond == -1).sum().item()} / {cond.numel()}")
    for lo, hi, nm in ((0, 16, "index"), (16, 32, "spatial"),
                       (32, 48, "random"), (48, 64, "block_contact")):
        s = cond[:, lo:hi]
        print(f"   cond stage {nm:<14} -1: {(s == -1).sum().item():6d} / {s.numel()}")
    # duplicate columns per row (the B1 double-selection)
    dup = 0
    for row in range(L):
        v = cur[row][cur[row] >= 0]
        dup += v.numel() - torch.unique(v).numel()
    print(f"   duplicate neighbour slots in current: {dup}")
    dup = 0
    for row in range(L):
        v = cond[row][cond[row] >= 0]
        dup += v.numel() - torch.unique(v).numel()
    print(f"   duplicate neighbour slots in cond:    {dup}")
    sys.exit()

# ------------------------------------------------------------------ pairfuse
if a.exp == "pairfuse":
    mods = []
    for b in m.diffusion.diffusion_stack.blocks:
        mods += [b.pair_1, b.pair_2]
    with torch.no_grad():
        NBR = [extract_neighbours(16, 16, 32)(POS[i], resi, chain, batch, mask,
                                              uniform=uni[i + 1]) for i in range(6)]
    nbrs = []
    for i in range(6):
        nbrs += [NBR[i], NBR[i]]
    poss = []
    for i in range(6):
        poss += [POS[i], PREV]

    ORDER = ["relpos", "dmap", "flags", "dist", "dir", "rot", "pairvec"]
    WID = {"relpos": 66, "dmap": 64, "flags": 3, "dist": 400,
           "dir": 60, "rot": 9, "pairvec": 240}
    TOT = sum(WID.values())

    def feats(mod, pos, nbr):
        idx = torch.arange(pos.shape[0], device=pos.device)
        g = {k: v[idx[:, None], nbr] for k, v in pc.items() if k != "omap"}
        relpos = GF.sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, nbr)
        dmap_rbf = torch.where(g["dmap_mask"][..., None], GF.distance_rbf(g["dmap"]),
                               torch.zeros((), device=pos.device))
        flags = torch.where(g["flags"].any(-1)[..., None], g["flags"],
                            torch.zeros_like(g["flags"]))
        return dict(relpos=relpos, dmap=dmap_rbf, flags=flags,
                    dist=GF.distance_features(pos, nbr, 0.0, 22.0),
                    dir=GF.direction_features(pos, nbr),
                    rot=GF.position_rotation_features(pos, nbr),
                    pairvec=GF.pair_vector_features(pos, nbr))

    LIN = {"relpos": "linear_relpos", "dmap": "linear_dmap", "flags": "linear_flags",
           "dist": "linear_dist", "dir": "linear_dir", "rot": "linear_rot",
           "pairvec": "linear_pairvec"}

    def v0():                                  # current: 7 GEMMs + 6 adds
        out = []
        for mod, pos, nbr in zip(mods, poss, nbrs):
            f = feats(mod, pos, nbr)
            pair = getattr(mod, LIN["relpos"])(f["relpos"])
            for k in ORDER[1:]:
                pair = pair + getattr(mod, LIN[k])(f[k])
            out.append(mod.mlp(mod.layer_norm(pair)))
        return torch.stack(out)

    def v1():                                  # runtime cat of feats AND weights
        out = []
        for mod, pos, nbr in zip(mods, poss, nbrs):
            f = feats(mod, pos, nbr)
            x = torch.cat([f[k] for k in ORDER], dim=-1)
            w = torch.cat([getattr(mod, LIN[k]).w for k in ORDER], dim=0)
            out.append(mod.mlp(mod.layer_norm(x @ w)))
        return torch.stack(out)

    BIGW = [torch.cat([getattr(mod, LIN[k]).w for k in ORDER], dim=0).contiguous()
            for mod in mods]

    def v2():                                  # load-time concatenated weight
        out = []
        for mod, pos, nbr, w in zip(mods, poss, nbrs, BIGW):
            f = feats(mod, pos, nbr)
            x = torch.cat([f[k] for k in ORDER], dim=-1)
            out.append(mod.mlp(mod.layer_norm(x @ w)))
        return torch.stack(out)

    OFF = {}
    o = 0
    for k in ORDER:
        OFF[k] = (o, o + WID[k])
        o += WID[k]

    def v3():                                  # features written into one buffer
        out = []
        for mod, pos, nbr, w in zip(mods, poss, nbrs, BIGW):
            f = feats(mod, pos, nbr)
            x = torch.empty(pos.shape[0], nbr.shape[1], TOT,
                            device=pos.device, dtype=w.dtype)
            for k in ORDER:
                lo, hi = OFF[k]
                x[..., lo:hi] = f[k]
            out.append(mod.mlp(mod.layer_norm(x @ w)))
        return torch.stack(out)

    print(f"pair-feature block x12, L={L}, K=64, width {TOT}")
    c0 = C(v0)
    t0, r0 = timed("v0 current (7 GEMM + 6 add)", c0)
    for nm, fn in (("v1 runtime cat feats+weights", v1),
                   ("v2 load-time cat weight", v2),
                   ("v3 v2 + feats into one buffer", v3)):
        t, _ = timed(nm, C(fn), ref=r0)
        print(f"     -> {(t0-t)/t0*100:+.1f}% vs v0   ({t0-t:+.3f} ms/fwd)")
    sys.exit()

# ------------------------------------------------------------------ neigh
if a.exp == "neigh":
    def v0():
        outs = []
        for b in range(6):
            base = 1 + b * 3
            cur = extract_neighbours(16, 16, 32)(POS[b], resi, chain, batch, mask,
                                                 uniform=uni[base])
            cond = extract_condition_neighbours(16, 16, 16, 16)(
                PREV, pc, resi, chain, batch, mask,
                uniform=uni[base + 1], uniform2=uni[base + 2])
            outs += [cur, cond]
        return torch.stack(outs)

    def v1():
        """Hoist every loop-invariant stage out of the 6-block loop."""
        idx16 = get_index_neighbours(16)(resi, chain, batch, mask)
        same_batch = batch[:, None] == batch[None, :]
        pm = mask[:, None] & mask[None, :] & same_batch
        ca_prev = PREV[:, 1]
        dist_c = torch.sqrt(torch.clamp(
            ((ca_prev[:, None] - ca_prev[None, :]) ** 2).sum(-1), min=1e-12))
        dist_c = torch.where(pc["dmap_mask"], pc["dmap"], dist_c)
        cond_base = get_neighbours(16)(dist_c, pm, idx16)          # (N,32)
        flags_any = pc["flags"].any(-1)
        block_d = torch.where(flags_any, flags_any.to(dist_c.dtype),
                              torch.full_like(dist_c, float("inf")))
        outs = []
        for b in range(6):
            base = 1 + b * 3
            ca = POS[b][:, 1]
            dist_s = pair_distance(ca)                              # shared
            ds = torch.where(same_batch, dist_s, torch.full_like(dist_s, float("inf")))
            cur = get_neighbours(16)(ds, pm, idx16)
            cur = get_random_neighbours(32)(dist_s, batch, mask, cur,
                                            uniform=uni[base])
            cond = get_random_neighbours(16)(dist_c, batch, mask, cond_base,
                                             uniform=uni[base + 1])
            cond = get_random_neighbours(16)(block_d, batch, mask, cond,
                                             uniform=uni[base + 2])
            outs += [cur, cond]
        return torch.stack(outs)

    print(f"neighbour selection, 6 blocks, L={L}")
    c0 = C(v0)
    t0, r0 = timed("v0 current", c0)
    t1, r1 = timed("v1 hoisted + shared distance", C(v1))
    print(f"     -> {(t0-t1)/t0*100:+.1f}% vs v0   ({t0-t1:+.3f} ms/fwd)")
    print(f"     bit-identical indices: {bool(torch.equal(r0, r1))}")
    sys.exit()

# ------------------------------------------------------------------ ipa
if a.exp == "ipa":
    attns, locals_, poss, nbrs, pairs, pms = [], [], [], [], [], []
    with torch.no_grad():
        for i, b in enumerate(m.diffusion.diffusion_stack.blocks):
            nbr = extract_neighbours(16, 16, 32)(POS[i], resi, chain, batch, mask,
                                                 uniform=uni[i + 1])
            for at in (b.attn_1.ada_point_attention, b.attn_2.ada_point_attention):
                attns.append(at)
                locals_.append(torch.randn(L, cfg.local_size, device=dev))
                poss.append(POS[i] / cfg.sigma_data)
                nbrs.append(nbr)
                pairs.append(torch.randn(L, nbr.shape[1], cfg.pair_size, device=dev))
                pms.append(mask[:, None] & mask[nbr] & (nbr != -1))
    from salad_torch.geometry.frames import aa_frames_t, apply_inverse_t
    from salad_torch.geometry.rigid import vec_norm

    def run(fused):
        out = []
        for at, lo, pos, nbr, pair, pm in zip(attns, locals_, poss, nbrs, pairs, pms):
            rot, trans, _ = aa_frames_t(pos)
            local = at.layer_norm(lo)
            n, h, s = L, at.heads, at.size
            qp, vp = at.query_points, at.value_points
            q, k, v = torch.split(at.qkv(local).reshape(n, h, 3 * s), s, dim=-1)
            q = at.layer_norm_1(q)
            k = at.layer_norm_2(k)
            g = at.qkv_global(local, rot, trans).reshape(n, h, 2 * qp + vp, 3)
            q_g, k_g, v_g = g[:, :, :qp], g[:, :, qp:2 * qp], g[:, :, 2 * qp:]
            bias = at.bias(pair)
            w_c = math.sqrt(2.0 / (9.0 * qp))
            w_l = math.sqrt(1.0 / 3.0)
            dfac = torch.nn.functional.softplus(at.gamma).reshape(1, 1, h) * w_c / 2
            if not fused:
                dist = dfac * ((q_g[:, None] - k_g[nbr]) ** 2).sum(dim=(-1, -2))
                dot = math.sqrt(1.0 / s) * torch.einsum("ihc,ijhc->ijh", q, k[nbr])
                logits = w_l * (dot + bias - dist)
            else:
                df = (dfac * w_l).reshape(1, h, 1)
                qf = q_g.reshape(n, h, -1)
                kf = k_g.reshape(n, h, -1)
                q2 = (qf * qf).sum(-1, keepdim=True)
                k2 = (kf * kf).sum(-1, keepdim=True)
                fq = torch.cat([q * (w_l * math.sqrt(1.0 / s)), qf * (2 * df),
                                -q2 * df, -torch.ones_like(q2) * df], dim=-1)
                fk = torch.cat([k, kf, torch.ones_like(k2), k2], dim=-1)
                logits = torch.einsum("ihd,ijhd->ijh", fq, fk[nbr]) + w_l * bias
            logits = torch.where(pm[..., None], logits, torch.full_like(logits, -1e9))
            attn = torch.softmax(logits, dim=-2)
            attn = torch.where(pm[..., None], attn, torch.zeros_like(attn))
            op = torch.einsum("ijh,ijc->ihc", attn, pair)
            os_ = torch.einsum("ijh,ijhc->ihc", attn, v[nbr])
            opt = torch.einsum("ijh,ijhpc->ihpc", attn, v_g[nbr])
            opt = apply_inverse_t(rot, trans, opt, k=2)
            out.append(at.project_out(torch.cat((
                op.reshape(n, -1), os_.reshape(n, -1),
                opt.reshape(n, -1), vec_norm(opt).reshape(n, -1)), dim=-1)))
        return torch.stack(out)

    print(f"sparse IPA x12, L={L}, K=64")
    t0, r0 = timed("v0 current", C(lambda: run(False)))
    t1, r1 = timed("v1 flash-IPA algebra", C(lambda: run(True)), ref=r0)
    print(f"     -> {(t0-t1)/t0*100:+.1f}% vs v0   ({t0-t1:+.3f} ms/fwd)")
    sys.exit()

print("unknown exp")

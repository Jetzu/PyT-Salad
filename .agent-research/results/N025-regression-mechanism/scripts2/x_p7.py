"""N025 - is P7's batched condition prologue MIS-COMPILED, or just reassociated?

Four arms on ONE dumped, bit-identical mid-rollout state:
  eager  P7 on / off        compiled  P7 on / off
so `|compiled - eager|` can be read separately with and without P7. Run one arm
per PROCESS: `opt.BATCH_COND` is read at trace time and changing it in-process
would not force a recompile.

`--unit` additionally compares the prologue's own output against the six
sequential `pair_2` calls at the same state, which is the direct test of whether
P7's maths is right for DISTINCT per-block neighbour indices (the eager entry
point feeds it six IDENTICAL index sets, so it never exercises that).
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--state", required=True, help="dumped arg tuple (.pt)")
ap.add_argument("--dump", action="store_true", help="build the state and exit")
ap.add_argument("--unit", action="store_true", help="prologue vs sequential pair_2")
ap.add_argument("--compiled", action="store_true")
ap.add_argument("--out", default="")
ap.add_argument("--num_aa", type=int, default=100)
ap.add_argument("--warm_steps", type=int, default=30)
ap.add_argument("--num_steps", type=int, default=250)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="cuda")
ap.add_argument("--config", default="default_vp")
ap.add_argument("--params", default="/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
a = ap.parse_args()

sys.path.insert(0, a.root)
from salad_torch import opt as salad_opt                                       # noqa
from salad_torch.config import get_config                                      # noqa
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz      # noqa
from salad_torch.diffusion.sampling import make_initial_data                   # noqa
from salad_torch.diffusion.schedules import parse_timescale                    # noqa
from salad_torch.modules.diffusion import uniform_draw_count                   # noqa
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion  # noqa

# DETERMINISTIC segment reduction everywhere: without it the atomics jitter
# (2.2e-4 A between identical forwards) swamps what we are measuring.
salad_opt.DET_SEGMENT = True

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device(a.device)
cfg = get_config(a.config)
model = StructureDiffusionInference(cfg).eval().to(dev)
load_model(model.predict, load_param_npz(a.params), cfg, strict=True)
L = a.num_aa
nd = uniform_draw_count(cfg)
ORDER = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
         "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt",
         "pos_gt", "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")
NAMES = ("pos", "local", "trajectory", "aa", "aa_features", "dssp",
         "distogram_neighbours", "raw_angles", "angles", "angle_pos", "aatype")
print(f"[x] root={a.root} device={dev} BATCH_COND={salad_opt.BATCH_COND} "
      f"DET_SEGMENT={salad_opt.DET_SEGMENT} "
      f"tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True)

# ---------------------------------------------------------------- dump
if a.dump:
    resi = torch.arange(L, device=dev)
    chain = torch.zeros(L, dtype=torch.long, device=dev)
    data, init_prev = make_initial_data(L, resi, chain, None, device=dev)
    prev = init_prev
    ts = parse_timescale("cosine(t)")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    cond = model.predict.prepare_condition(data)
    pc = cond["pair_condition"]
    # roll forward EAGERLY with P7 OFF, so the state we probe is produced by the
    # path the port was validated on and is not itself contaminated by P7.
    salad_opt.BATCH_COND = False
    args = None
    for step in range(a.warm_steps + 1):
        raw_t = 1 - step / a.num_steps
        data["t_pos"] = ts(raw_t) * torch.ones_like(data["t_pos"])
        data["t_seq"] = torch.ones_like(data["t_seq"])
        if raw_t < 0.8:
            prev = init_prev
        with torch.no_grad():
            noised = apply_diffusion(cfg, data, generator=gen)
        d = dict(data, **noised)
        uni = torch.rand(nd, L, L, device=dev, generator=gen).clamp_(1e-6, 1 - 1e-6)
        pack = dict(pos_noised=d["pos_noised"], prev_pos=prev["pos"],
                    prev_local=prev["local"], condition=cond["condition"],
                    dmap=pc["dmap"], omap=pc["omap"], dmap_mask=pc["dmap_mask"],
                    flags=pc["flags"], t_pos=d["t_pos"], t_seq=d["t_seq"],
                    seq_noised=d["seq_noised"], aa_gt=d["aa_gt"], pos_gt=d["pos"],
                    resi=d["residue_index"], chain=d["chain_index"],
                    batch=d["batch_index"], mask=d["mask"], cyclic_mask=None,
                    uniform=uni)
        if step == a.warm_steps:
            args = pack
            break
        with torch.no_grad():
            out = model.predict.diffusion.forward_core(
                *[pack[k] for k in ORDER], True, True, False)
        data["pos"] = out[0]
        data["seq"] = out[3].argmax(-1)
        prev = {"pos": out[0], "local": out[1]}
    pp = args["prev_pos"]
    print(f"[dump] step={a.warm_steps} raw_t={1 - a.warm_steps / a.num_steps:.3f} "
          f"prev_pos |max|={pp.abs().max():.3f} nonzero={bool((pp != 0).any())} "
          f"pos_noised |max|={args['pos_noised'].abs().max():.3f}", flush=True)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in args.items()},
               a.state)
    print("[dump] wrote", a.state, flush=True)
    raise SystemExit(0)

st = torch.load(a.state, map_location="cpu")
args = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in st.items()}
pp = args["prev_pos"]
print(f"[x] state prev_pos |max|={pp.abs().max():.4f} nonzero={bool((pp != 0).any())}",
      flush=True)

# ---------------------------------------------------------------- unit
if a.unit:
    from salad_torch.modules.diffusion import (                               # noqa
        get_index_neighbours, condition_neighbour_prologue, condition_neighbour_tail)
    dm = model.predict.diffusion
    stack = dm.diffusion_stack
    if stack.cond_pair_stack is None:
        print("[unit] no cond_pair_stack for this config"); raise SystemExit(0)
    print(f"[unit] cache in_sync = {stack.cond_pair_stack.in_sync(stack.blocks)}",
          flush=True)
    draws = args["uniform"]
    resi, chain, batch, mask = (args["resi"], args["chain"], args["batch"],
                                args["mask"])
    prev_pos = args["prev_pos"]
    depth = len(stack.blocks)
    with torch.no_grad():
        idx_nb = get_index_neighbours(16)(resi, chain, batch, mask)
        cond_dist, cond_seed = condition_neighbour_prologue(16, 16)(
            prev_pos, None, resi, chain, batch, mask, index_neighbours=idx_nb)
        nbrs = [condition_neighbour_tail(16, 0)(
                    cond_dist, cond_seed, None, batch, mask,
                    uniform=draws[1 + i * 3 + 1], uniform2=draws[1 + i * 3 + 2])
                for i in range(depth)]
        same01 = int((nbrs[0] == nbrs[1]).all())
        print(f"[unit] per-block cond neighbour sets identical(0,1)? {bool(same01)} "
              f"shape={tuple(nbrs[0].shape)}", flush=True)
        nb_stack = torch.stack(nbrs)
        bp, bm = stack.cond_pair_stack(prev_pos, None, nb_stack, resi, chain,
                                       batch, mask, cyclic_mask=None, init_pos=None)
        print(f"[unit] {'blk':>4} {'max|dpair|':>12} {'rel':>10} {'mask eq':>8}")
        worst = 0.0
        for i, blk in enumerate(stack.blocks):
            sp, sm = blk.pair_2(prev_pos, None, nbrs[i], resi, chain, batch,
                                mask, cyclic_mask=None, init_pos=None)
            d = (bp[i] - sp).abs().max().item()
            scale = sp.abs().max().item() or 1.0
            worst = max(worst, d)
            print(f"[unit] {i:>4} {d:>12.3e} {d / scale:>10.3e} "
                  f"{bool((bm[i] == sm).all()):>8}", flush=True)
        print(f"[unit] WORST max|dpair| over {depth} blocks = {worst:.6e}", flush=True)
    raise SystemExit(0)

# ---------------------------------------------------------------- forward arm
fn = model.predict.diffusion.forward_core
if a.compiled:
    fn = torch.compile(fn, mode="default", dynamic=False)
argl = [args[k] for k in ORDER] + [True, True, False]
with torch.no_grad():
    for _ in range(3 if a.compiled else 1):
        out = fn(*argl)
res = {n: o.detach().cpu().numpy() for n, o in zip(NAMES, out) if torch.is_tensor(o)}
np.savez(a.out, **res)
print(f"[x] wrote {a.out}  pos|max|={np.abs(res['pos']).max():.4f}", flush=True)

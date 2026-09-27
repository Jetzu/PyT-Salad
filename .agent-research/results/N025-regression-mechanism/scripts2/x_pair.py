"""N025 - how fast does P7's ~1e-5 reassociation grow along the rollout?

One process = one arm. Rolls ONE design (or a few) through the COMPILED
`forward_core` with a fixed seed, dumping `pos` at every step. Two arms run with
`SALAD_OPT_BATCHCOND` 1 and 0 get bit-identical inputs at step 0, so the whole
difference between their dumps is P7 and its amplification.
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--num_aa", type=int, default=100)
ap.add_argument("--num_designs", type=int, default=3)
ap.add_argument("--num_steps", type=int, default=250)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--eager", action="store_true")
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

# deterministic segment reduction: removes the 2.2e-4 A atomics jitter, which
# would otherwise BE the perturbation we are trying to attribute to P7.
salad_opt.DET_SEGMENT = True
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
cfg = get_config(a.config)
model = StructureDiffusionInference(cfg).eval().to(dev)
load_model(model.predict, load_param_npz(a.params), cfg, strict=True)
L, nd = a.num_aa, uniform_draw_count(cfg)
ORDER = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
         "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt",
         "pos_gt", "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")
fn = model.predict.diffusion.forward_core
if not a.eager:
    fn = torch.compile(fn, mode="default", dynamic=False)
print(f"[p] root={a.root} BATCH_COND={salad_opt.BATCH_COND} "
      f"DET_SEGMENT={salad_opt.DET_SEGMENT} compiled={not a.eager}", flush=True)

resi = torch.arange(L, device=dev)
chain = torch.zeros(L, dtype=torch.long, device=dev)
ts = parse_timescale("cosine(t)")
allpos, allnbr = [], []
for design in range(a.num_designs):
    gen = torch.Generator(device=dev).manual_seed(a.seed * 1000003 + design)
    data, init_prev = make_initial_data(L, resi, chain, None, device=dev)
    prev = init_prev
    cond = model.predict.prepare_condition(data)
    pc = cond["pair_condition"]
    traj, nbrs = [], []
    for step in range(a.num_steps):
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
        with torch.no_grad():
            out = fn(*[pack[k] for k in ORDER], True, True, False)
        data["pos"] = out[0]
        data["seq"] = out[3].argmax(-1)
        prev = {"pos": out[0], "local": out[1]}
        traj.append(out[0][:, 1].detach().cpu().numpy())        # CA only
        nbrs.append(out[6].detach().cpu().numpy().astype(np.int32))
    allpos.append(np.stack(traj))
    allnbr.append(np.stack(nbrs))
    ca = allpos[-1][-1]
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
    cc = float(np.linalg.norm(np.diff(ca, axis=0), axis=1).mean())
    print(f"[p] design {design}: final CA-CA {cc:.3f} Rg {rg:.2f}", flush=True)
np.savez_compressed(a.out, pos=np.stack(allpos), nbr=np.stack(allnbr))
print("[p] wrote", a.out, flush=True)

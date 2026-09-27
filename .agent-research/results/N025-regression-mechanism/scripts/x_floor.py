"""N025 - the SELF-DIVERGENCE floor of the captured forward, measured properly.

A floor read from ONE pair of identical calls is a coin flip on this model: the
neighbour argsort turns ~1.6e-7 of atomics jitter into a DISCRETE tie flip
(notes/04 section 8), so repeats are bimodal - most agree to ~1e-5 and a few
land a flip and disagree by ~1e-2. So this runs K identical calls on ONE real
mid-rollout state and reports the whole distribution plus the number of
DISTINCT bit patterns, at a fixed state, so two commits are comparable.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np, torch

ROOT = os.environ["SALAD_ROOT"]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("N025_NOCSE") == "1":
    from n025_veto import veto_cse
    veto_cse()

from salad_torch import opt as salad_opt                                     # noqa
from salad_torch.config import get_config                                    # noqa
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz    # noqa
from salad_torch.diffusion.schedules import parse_timescale                  # noqa
from salad_torch.diffusion.sampling import make_initial_data                 # noqa
from salad_torch.modules.diffusion import uniform_draw_count                 # noqa
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--num_aa", type=int, default=100)
ap.add_argument("--preset", default="fast")
ap.add_argument("--eager", action="store_true")
ap.add_argument("--repeats", type=int, default=24)
ap.add_argument("--warm_steps", type=int, default=20)
ap.add_argument("--num_steps", type=int, default=250)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--params", default="/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
cfg = get_config(a.config)
model = StructureDiffusionInference(cfg).eval().to(dev)
load_model(model.predict, load_param_npz(a.params), cfg, strict=True)
L = a.num_aa
resi = torch.arange(L, device=dev)
chain = torch.zeros(L, dtype=torch.long, device=dev)
nd = uniform_draw_count(cfg)
fastd = None
if not a.eager:
    from salad_torch.modules.fast import FastDenoiser
    fastd = FastDenoiser(model.predict, L, device=dev, preset=a.preset).warmup(pair_cond=False)
print("[x] tf32 =", torch.backends.cuda.matmul.allow_tf32,
      "DETSEG =", getattr(salad_opt, "DET_SEGMENT", "absent"),
      "BATCH_COND =", getattr(salad_opt, "BATCH_COND", "absent"),
      "AGGR =", getattr(salad_opt, "AGGRESSIVE_FUSION", "absent"), flush=True)

gen = torch.Generator(device=dev).manual_seed(a.seed)
ts = parse_timescale("cosine(t)")
data, prev = make_initial_data(L, resi, chain, None, device=dev)


def call(dd, pp, uni, need_aa=False):
    with torch.no_grad():
        if fastd is not None:
            return fastd(dd, pp, need_aa=need_aa, uniform=uni, generator=gen)
        return model.predict(dd, pp, uniform=uni[0], compute_distogram=False)


for step in range(a.warm_steps):
    raw_t = 1 - step / a.num_steps
    data["t_pos"] = ts(raw_t) * torch.ones_like(data["t_pos"])
    data["t_seq"] = torch.ones_like(data["t_seq"])
    with torch.no_grad():
        noised = apply_diffusion(cfg, data, generator=gen)
    uni = torch.rand(nd, L, L, device=dev, generator=gen).clamp_(1e-6, 1 - 1e-6)
    upd, prev = call(dict(data, **noised), prev, uni)
    data["pos"] = upd["pos"]
with torch.no_grad():
    noised = apply_diffusion(cfg, data, generator=gen)
base_in = dict(data, **noised)
uni0 = torch.rand(nd, L, L, device=dev, generator=gen).clamp_(1e-6, 1 - 1e-6)

outs, hashes = [], []
for _ in range(a.repeats):
    r, _ = call(base_in, prev, uni0)
    p = r["pos"].detach().float().cpu().numpy()
    outs.append(p.astype(np.float64))
    hashes.append(hashlib.blake2b(p.tobytes(), digest_size=8).hexdigest())
ref = outs[0]
d = np.array([np.abs(o - ref).max() for o in outs[1:]])
res = dict(repeats=a.repeats, distinct_bit_patterns=len(set(hashes)),
           d_min=float(d.min()), d_median=float(np.median(d)),
           d_max=float(d.max()), d_mean=float(d.mean()),
           frac_above_1e3=float((d > 1e-3).mean()),
           frac_above_1e2=float((d > 1e-2).mean()),
           all_pairs_max=float(max(np.abs(outs[i] - outs[j]).max()
                                   for i in range(len(outs))
                                   for j in range(i + 1, len(outs)))),
           hashes_head=hashes[:6])
for k, v in res.items():
    print(f"  {k:24s} {v}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print("[x] wrote", a.out, flush=True)

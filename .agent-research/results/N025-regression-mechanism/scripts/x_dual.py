"""N025 - COMPILED vs EAGER, step by step, along a real rollout.

The rollout is driven by ONE leg (the compiled denoiser by default); at every
step the OTHER leg is run on the bitwise-identical inputs and the two outputs
are compared. Eager is the reference: it is what the four JAX parity fixtures
check. So this measures how far the compiled/captured path has drifted from the
model the port was validated against, AT EVERY STEP OF A REAL TRAJECTORY,
instead of at one synthetic forward.

`distogram_neighbours` is reported separately because it is a DISCRETE index
tensor: a disagreement there is not rounding, it is a different neighbourhood,
which is the quantity a rollout's tertiary contacts are built from.
"""
from __future__ import annotations
import argparse, json, os, sys
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
from salad_torch.modules.diffusion import uniform_draw_count                 # noqa
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion  # noqa
from salad_torch.diffusion.sampling import make_initial_data                 # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--num_aa", type=int, default=100)
ap.add_argument("--num_designs", type=int, default=4)
ap.add_argument("--num_steps", type=int, default=250)
ap.add_argument("--prev_threshold", type=float, default=0.8)
ap.add_argument("--preset", default="fast")
ap.add_argument("--drive", default="compiled", choices=["compiled", "eager"])
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

from salad_torch.modules.fast import FastDenoiser                            # noqa
fastd = FastDenoiser(model.predict, L, device=dev, preset=a.preset).warmup(pair_cond=False)
print("[x] tf32 =", torch.backends.cuda.matmul.allow_tf32, flush=True)
ts = parse_timescale("cosine(t)")
gen = torch.Generator(device=dev).manual_seed(a.seed)
rows = []

for design in range(a.num_designs):
    fastd.invalidate_condition_cache()
    data, init_prev = make_initial_data(L, resi, chain, None, device=dev)
    prev = init_prev
    for step in range(a.num_steps):
        raw_t = 1 - step / a.num_steps
        data["t_pos"] = ts(raw_t) * torch.ones_like(data["t_pos"])
        data["t_seq"] = torch.ones_like(data["t_seq"])
        if raw_t < a.prev_threshold:
            prev = init_prev
        with torch.no_grad():
            noised = apply_diffusion(cfg, data, generator=gen)
        uni = torch.rand(nd, L, L, device=dev, generator=gen).clamp_(1e-6, 1 - 1e-6)
        d_in = dict(data, **noised)
        with torch.no_grad():
            rc, pc_ = fastd(d_in, prev, need_aa=True, uniform=uni, generator=gen)
            # eager takes ONE (N, N) field - feed the compiled path's slice 0 so
            # at least the FIRST neighbour stage sees identical randomness; the
            # later stages differ by construction (upstream's eager path reuses
            # one field), which is why the DISCRETE column below is read against
            # a same-commit baseline, not as an absolute.
            re, pe_ = model.predict(d_in, prev, uniform=uni[0], compute_distogram=True)
        dp = float((rc["pos"].double() - re["pos"].double()).abs().max())
        dl = float((rc["local"].double() - re["local"].double()).abs().max())
        rec = dict(design=design, step=step, raw_t=raw_t, d_pos=dp, d_local=dl)
        for k in ("distogram_neighbours", "aatype"):
            if k in rc and k in re and torch.is_tensor(rc[k]):
                rec["disagree_" + k] = float((rc[k] != re[k]).float().mean())
        ca = rc["pos"][:, 1].double()
        rec["ca_ca"] = float((ca[1:] - ca[:-1]).norm(dim=-1).mean())
        rec["rg"] = float((ca - ca.mean(0)).pow(2).sum(1).mean().sqrt())
        rows.append(rec)
        data["pos"] = (rc if a.drive == "compiled" else re)["pos"]
        data["seq"] = (rc if a.drive == "compiled" else re)["aa"].argmax(-1)
        prev = pc_ if a.drive == "compiled" else pe_
    print(f"[design {design}] CA-CA {rows[-1]['ca_ca']:.3f} Rg {rows[-1]['rg']:.2f} "
          f"last d_pos {rows[-1]['d_pos']:.3e}", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rows, open(a.out, "w"))
print("[x] wrote", a.out, len(rows), flush=True)

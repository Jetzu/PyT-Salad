"""N025 - INPUT-SENSITIVITY probe of the compiled/captured forward.

Every existing parity test compares ONE forward against a JAX fixture with all
inputs fixed. That cannot see an input that has stopped mattering. This probe
asks the opposite question: with everything else held bitwise fixed, does each
input still CHANGE the output, and by how much relative to the run-to-run floor?

Dead inputs to look for:
  * `uniform`  - the pre-drawn Gumbel field. 1 + depth*3 = 19 (L, L) slices.
    If a compiled/captured graph froze the capture-time field, or a CSE pass
    collapsed two draws, some slice stops mattering while a single forward
    still matches JAX (the fixture pins `uniform` too).
  * `prev_pos` / `prev_local` - the self-conditioning feedback.

Run at HEAD and at the known-good commit and diff the columns.
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
from salad_torch.diffusion.sampling import SamplerConfig, make_initial_data  # noqa
from salad_torch.diffusion.schedules import parse_timescale                  # noqa
from salad_torch.modules.diffusion import uniform_draw_count                 # noqa
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion  # noqa


def d(a, b):
    return float((a.double() - b.double()).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_aa", type=int, default=100)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--warm_steps", type=int, default=20,
                    help="steps of real rollout before probing, so the state is real")
    ap.add_argument("--num_steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="default_vp")
    ap.add_argument("--params", default="/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    cfgm = get_config(a.config)
    model = StructureDiffusionInference(cfgm).eval().to(device)
    load_model(model.predict, load_param_npz(a.params), cfgm, strict=True)
    L = a.num_aa
    resi = torch.arange(L, device=device)
    chain = torch.zeros(L, dtype=torch.long, device=device)
    nd = uniform_draw_count(cfgm)

    fastd = None
    if not a.eager:
        from salad_torch.modules.fast import FastDenoiser
        fastd = FastDenoiser(model.predict, L, device=device,
                             preset=a.preset).warmup(pair_cond=False)
    print("[x] tf32 =", torch.backends.cuda.matmul.allow_tf32, "n_draws =", nd, flush=True)

    gen = torch.Generator(device=device).manual_seed(a.seed)
    data, prev = make_initial_data(L, resi, chain, None, device=device)
    ts = parse_timescale("cosine(t)")

    def call(dd, pp, uni, need_aa=False):
        # the EAGER Diffusion.forward takes ONE (N, N) field and reuses it for
        # every draw; only forward_core (the fast path) slices a
        # (n_draws, N, N) stack. So the eager arm is fed slice 0, which is the
        # same randomness it would have drawn itself, in the shape it wants.
        with torch.no_grad():
            if fastd is not None:
                return fastd(dd, pp, need_aa=need_aa, uniform=uni, generator=gen)
            return model.predict(dd, pp, uniform=uni[0], compute_distogram=False)

    # --- warm the rollout up to a REAL state --------------------------------
    for step in range(a.warm_steps):
        raw_t = 1 - step / a.num_steps
        data["t_pos"] = ts(raw_t) * torch.ones_like(data["t_pos"])
        data["t_seq"] = torch.ones_like(data["t_seq"])
        with torch.no_grad():
            noised = apply_diffusion(cfgm, data, generator=gen)
        uni = torch.rand(nd, L, L, device=device, generator=gen).clamp_(1e-6, 1 - 1e-6)
        upd, prev = call(dict(data, **noised), prev, uni)
        data["pos"] = upd["pos"]
    with torch.no_grad():
        noised = apply_diffusion(cfgm, data, generator=gen)
    base_in = dict(data, **noised)
    uni0 = torch.rand(nd, L, L, device=device, generator=gen).clamp_(1e-6, 1 - 1e-6)
    uni1 = torch.rand(nd, L, L, device=device, generator=gen).clamp_(1e-6, 1 - 1e-6)

    ref, _ = call(base_in, prev, uni0)
    ref = ref["pos"].clone()

    rows = {}
    # floor: the identical call, repeated
    r2, _ = call(base_in, prev, uni0)
    rows["floor_identical"] = d(ref, r2["pos"])
    # whole uniform field replaced
    r3, _ = call(base_in, prev, uni1)
    rows["uniform_all"] = d(ref, r3["pos"])
    # each slice on its own
    per = []
    for j in range(nd):
        u = uni0.clone(); u[j] = uni1[j]
        rj, _ = call(base_in, prev, u)
        per.append(d(ref, rj["pos"]))
    rows["uniform_per_slice"] = per
    # self-conditioning inputs
    pz = {"pos": torch.zeros_like(prev["pos"]), "local": prev["local"]}
    rj, _ = call(base_in, pz, uni0); rows["prev_pos_zeroed"] = d(ref, rj["pos"])
    pz = {"pos": prev["pos"], "local": torch.zeros_like(prev["local"])}
    rj, _ = call(base_in, pz, uni0); rows["prev_local_zeroed"] = d(ref, rj["pos"])
    pz = {"pos": torch.zeros_like(prev["pos"]), "local": torch.zeros_like(prev["local"])}
    rj, _ = call(base_in, pz, uni0); rows["prev_both_zeroed"] = d(ref, rj["pos"])
    # a control that must always matter
    bi = dict(base_in); bi["pos_noised"] = base_in["pos_noised"] + 0.01
    rj, _ = call(bi, prev, uni0); rows["pos_noised_plus_0.01"] = d(ref, rj["pos"])
    bi = dict(base_in); bi["t_pos"] = base_in["t_pos"] * 0.9
    rj, _ = call(bi, prev, uni0); rows["t_pos_x0.9"] = d(ref, rj["pos"])
    # `pos_gt` = data["pos"], which _pack_inputs also feeds in
    bi = dict(base_in); bi["pos"] = base_in["pos"] + 0.01
    rj, _ = call(bi, prev, uni0); rows["pos_gt_plus_0.01"] = d(ref, rj["pos"])

    rows["prev_pos_norm"] = float(prev["pos"].norm().item())
    rows["prev_local_norm"] = float(prev["local"].norm().item())
    rows["ref_absmax"] = float(ref.abs().max().item())
    for k, v in rows.items():
        print(f"  {k:24s} {v}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print("[x] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

"""N025 - per-step rollout instrumentation.

Reproduces scripts/generate.py's ORDINARY (non-grouped) loop for the plain
de-novo case, with a recorder around step_fn. Nothing about the schedule, the
RNG stream order or the denoiser call is changed: apply_diffusion is called
first (as in generate.py), then the (n_draws, L, L) uniform field is drawn off
the SAME generator with the SAME call FastDenoiser._prepare_inputs would make,
then passed in explicitly so it can be hashed.

Records per (design, step):
  raw_t, ca_ca (mean CA-CA of the x0 prediction), rg, max|pos|,
  ||prev_pos||, ||prev_local||, max|prev_local|,
  h_prev_in  = hash of prev["pos"] fed to this step
  h_pos_out  = hash of update["pos"] this step produced
  h_uniform  = hash of the Gumbel uniform field
  h_noise    = hash of pos_noised
  sc_live    = whether self-conditioning was live (raw_t >= prev_threshold)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np, torch

ROOT = os.environ["SALAD_ROOT"]
sys.path.insert(0, ROOT)
from salad_torch import opt as salad_opt                                    # noqa
from salad_torch.config import get_config                                   # noqa
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz   # noqa
from salad_torch.diffusion.sampling import SamplerConfig, make_initial_data, sample  # noqa
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion   # noqa
from salad_torch.modules.diffusion import uniform_draw_count                # noqa


def h(t):
    if t is None:
        return ""
    a = t.detach().to(torch.float32).cpu().contiguous().numpy()
    return hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest()


def geom(pos):
    ca = pos[:, 1].detach().to(torch.float64).cpu().numpy()
    if not np.isfinite(ca).all():
        return float("nan"), float("nan")
    b = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    return float(b.mean()), rg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_aa", type=int, default=100)
    ap.add_argument("--num_designs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--num_steps", type=int, default=250)
    ap.add_argument("--out_step", type=int, default=200)
    ap.add_argument("--prev_threshold", type=float, default=0.8)
    ap.add_argument("--config", default="default_vp")
    ap.add_argument("--params", default="/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_cse", action="store_true",
                    help="set torch._inductor.config.joint_custom_post_pass = None")
    ap.add_argument("--tf32", default="", help="force torch tf32 on/off on the eager path")
    ap.add_argument("--no_capture", action="store_true",
                    help="compile but do NOT capture a CUDA graph (separates "
                         "inductor compilation from graph capture)")
    args = ap.parse_args()

    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    config = get_config(args.config)
    model = StructureDiffusionInference(config).eval().to(device)
    load_model(model.predict, load_param_npz(args.params), config, strict=True)

    L = args.num_aa
    resi = torch.arange(L, device=device)
    chain = torch.zeros(L, dtype=torch.long, device=device)

    fastd = None
    if args.fast:
        if args.no_cse:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from n025_veto import veto_cse
            veto_cse()
        from salad_torch.modules.fast import FastDenoiser
        fastd = FastDenoiser(model.predict, L, device=device,
                             preset=args.preset,
                             # mode="default" with capture=False is a PLAIN
                             # compile: no manual graph and no inductor-managed
                             # cudagraphs either (the "reduce-overhead" default
                             # would still capture internally, which is the
                             # thing this arm is trying to remove).
                             capture=not args.no_capture,
                             manual_graph=not args.no_capture,
                             mode=("default" if args.no_capture
                                   else "reduce-overhead"),
                             ).warmup(pair_cond=False)
    if args.tf32:
        on = args.tf32 not in ("0", "off", "false")
        torch.backends.cuda.matmul.allow_tf32 = on
        torch.backends.cudnn.allow_tf32 = on
    print("[x] tf32 =", torch.backends.cuda.matmul.allow_tf32, flush=True)

    cfg = SamplerConfig(num_steps=args.num_steps, out_steps=(args.out_step,),
                        prev_threshold=args.prev_threshold,
                        lean_steps=bool(args.fast and not config.diffuse_sequence
                                        and salad_opt.LEAN_STEPS))
    n_draws = uniform_draw_count(config)
    print("[x] n_draws =", n_draws, flush=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    state = {"design": 0, "step": 0}

    def step_fn(d, p, generator=None, *, need_aa=True):
        with torch.no_grad():
            noised = apply_diffusion(config, d, generator=generator)
            rec = dict(design=state["design"], step=state["step"],
                       raw_t=float(d["t_pos"][0].item()),
                       prev_pos_norm=float(p["pos"].norm().item()),
                       prev_local_norm=float(p["local"].norm().item()),
                       prev_local_absmax=float(p["local"].abs().max().item()),
                       h_prev_in=h(p["pos"]), h_prevloc_in=h(p["local"]),
                       h_noise=h(noised["pos_noised"]))
            # the SAME call FastDenoiser._prepare_inputs makes, in the same
            # stream position (after apply_diffusion), so the RNG order is
            # unchanged and the field can be hashed. Passed explicitly to the
            # eager path too, which otherwise draws off the process-global RNG
            # (K007) and would not be comparable at all.
            uni = torch.rand(n_draws, L, L, device=device, dtype=torch.float32,
                             generator=generator).clamp_(1e-6, 1 - 1e-6)
            rec["h_uniform"] = h(uni)
            if fastd is not None:
                update, newprev = fastd(dict(d, **noised), p, need_aa=need_aa,
                                        uniform=uni, generator=generator)
            else:
                # eager takes ONE (N, N) field, not the (n_draws, N, N) stack
                update, newprev = model.predict(dict(d, **noised), p, uniform=uni[0],
                                                compute_distogram=False)
            pos = update["pos"]
            cc, rg = geom(pos)
            rec.update(ca_ca=cc, rg=rg,
                       posmax=float(pos.abs().max().item()),
                       h_pos_out=h(pos), h_local_out=h(update["local"]),
                       h_newprev=h(newprev["pos"]))
            rows.append(rec)
            state["step"] += 1
            return update, newprev

    for design in range(args.num_designs):
        if fastd is not None:
            fastd.invalidate_condition_cache()
        state["design"], state["step"] = design, 0
        data, prev = make_initial_data(L, resi, chain, None, device=device)
        snaps = sample(step_fn, data, prev, cfg, generator=gen)
        s = snaps[-1]
        cc, rg = geom(s["pos"])
        print(f"[design {design}] final CA-CA {cc:.3f} Rg {rg:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh)
    print("[x] wrote", args.out, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()

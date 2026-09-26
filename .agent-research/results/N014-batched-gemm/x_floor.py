"""N013 - quantify the CUDA run-to-run noise floor of the `exact` denoiser.

N009 section 7.5 / notes/04 section 8 report two SAMPLES of it (1.755e-02 and
2.21e-04 A). A sample is not a floor. This runs the production path - real
weights, `preset="exact"`, compiled + manually CUDA-graphed - on ONE frozen
input, `--reps` times, and reports the whole distribution of pairwise
max|d| over the returned `pos`.

The input is frozen bit for bit (same static buffers, same pre-drawn
`uniform`), so every difference between two replies is the backend.

    python x_floor.py --length 100 --reps 40 --save floor_L100.pt

`--input real` first advances the real sampler `--upto` steps so the frozen
state is a mid-trajectory protein rather than a Gaussian blob - tie-flipping
in the neighbour argsort is a property of the geometry, so this matters.
"""
import argparse, os, sys, time, json
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--preset", default="exact")
ap.add_argument("--input", default="synth", choices=("synth", "real"))
ap.add_argument("--upto", type=int, default=250)
ap.add_argument("--save", default="")
ap.add_argument("--state-save", default="")
ap.add_argument("--state-load", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--time", action="store_true", help="also time the step")
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()


def _apply_inductor_env():
    spec = os.environ.get("SALAD_IND", "")
    if not spec:
        return ""
    import torch._inductor.config as ind
    for item in spec.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        cur = getattr(ind, k)
        setattr(ind, k, type(cur)(int(v)) if isinstance(cur, bool) else
                (int(v) if isinstance(cur, int) else v))
    return spec


torch.set_float32_matmul_precision("highest")
_IND = _apply_inductor_env()
_DET = os.environ.get("SALAD_TORCH_DET", "")
if _DET:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

from salad_torch import opt                                     # noqa: E402
from salad_torch.config import get_config                       # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import (StructureDiffusionPredict,           # noqa: E402
                                       StructureDiffusionInference, apply_diffusion)
from salad_torch.modules.fast import FastDenoiser                # noqa: E402
from salad_torch.diffusion.sampling import make_initial_data     # noqa: E402
from salad_torch.diffusion.schedules import parse_timescale      # noqa: E402

dev = torch.device("cuda")
cfg = get_config(a.config)
PARAMS = f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"
L = a.length
A = 5 + cfg.augment_size

inf = StructureDiffusionInference(cfg).eval().to(dev)
load_model(inf.predict, load_param_npz(PARAMS), cfg, strict=True)
m = inf.predict

g = torch.Generator(device="cpu").manual_seed(11)
z = torch.zeros(L, dtype=torch.long, device=dev)
resi = torch.arange(L, device=dev)

fast = FastDenoiser(m, L, device=dev, preset=a.preset).warmup(pair_cond=False)
print(f"gate: pair_cond_active={fast._pc_active} graphs={sorted(fast._graphs)}")

if a.state_load:
    _st = torch.load(a.state_load, weights_only=False)
    data = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in _st["data"].items()}
    prev = {k: v.to(dev) for k, v in _st["prev"].items()}
    a.input = "loaded@%d" % _st["upto"]
    print(f"loaded frozen state from {a.state_load}")
elif a.input == "synth":
    pos = (torch.randn(L, A, 3, generator=g) * 4).to(dev)
    data = dict(pos=pos, pos_noised=pos,
                aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
                seq=torch.full((L,), 20, dtype=torch.long, device=dev),
                seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
                residue_index=resi, chain_index=z, batch_index=z,
                mask=torch.ones(L, dtype=torch.bool, device=dev),
                t_pos=torch.full((L,), 0.5, device=dev),
                t_seq=torch.ones(L, device=dev))
    prev = {"pos": torch.zeros(L, A, 3, device=dev),
            "local": torch.zeros(L, cfg.local_size, device=dev)}
else:
    # advance the REAL sampler `--upto` steps, then freeze that state.
    gen = torch.Generator(device=dev).manual_seed(42)
    ts = parse_timescale("cosine(t)")
    data, prev = make_initial_data(L, resi, z, None, atom_count=A,
                                   local_size=cfg.local_size, device=dev)
    init_prev = prev
    with torch.no_grad():
        for step in range(a.upto):
            raw_t = 1 - step / 500
            data["t_pos"] = ts(raw_t) * torch.ones_like(data["t_pos"])
            data["t_seq"] = torch.ones_like(data["t_seq"])
            if raw_t < 0.8:
                prev = init_prev
            noised = apply_diffusion(cfg, data, generator=gen)
            upd, prev = fast(dict(data, **noised), prev, need_aa=False,
                             generator=gen)
            data["pos"] = upd["pos"]
    noised = apply_diffusion(cfg, data, generator=gen)
    ca = data["pos"][:, 1]
    print(f"real state at step {a.upto}: raw_t={1 - a.upto/500:.3f}  "
          f"CA-CA {torch.linalg.norm(ca[1:] - ca[:-1], dim=-1).mean():.3f} A  "
          f"Rg {(ca - ca.mean(0)).pow(2).sum(-1).mean().sqrt():.2f} A")
    data = dict(data, **noised)
    prev = {k: v.clone() for k, v in prev.items()}

if a.state_save:
    torch.save({"data": {k: (v.cpu() if torch.is_tensor(v) else v)
                         for k, v in data.items()},
                "prev": {k: v.cpu() for k, v in prev.items()},
                "upto": a.upto}, a.state_save)
    print(f"saved frozen state to {a.state_save}")

# freeze the Gumbel draws too: they are an INPUT of the captured graph
uniform = torch.rand(fast.n_draws, L, L, device=dev, generator=torch.Generator(
    device=dev).manual_seed(5)).clamp_(1e-6, 1 - 1e-6)

fast.invalidate_condition_cache()
outs = []
with torch.no_grad():
    for _ in range(a.reps):
        r, _ = fast(data, prev, need_aa=False, uniform=uniform)
        outs.append({k: r[k].clone() for k in ("pos", "local", "trajectory")
                     if k in r})

def stats(key):
    xs = torch.stack([o[key] for o in outs]).double()
    n = xs.shape[0]
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append((xs[i] - xs[j]).abs().max().item())
    d = torch.tensor(d)
    uniq = {}
    for i, o in enumerate(outs):
        b = o[key].cpu().numpy().tobytes()
        uniq.setdefault(hash(b), []).append(i)
    ds, _ = d.sort()
    return dict(key=key, n_reps=n, n_pairs=len(d),
                distinct=len(uniq),
                cluster_sizes=sorted((len(v) for v in uniq.values()), reverse=True),
                zero_pairs=int((d == 0).sum()),
                mean=d.mean().item(), median=d.median().item(),
                p95=ds[int(0.95 * (len(ds) - 1))].item(), max=d.max().item(),
                nonzero_min=(d[d > 0].min().item() if (d > 0).any() else 0.0),
                nonzero_median=(d[d > 0].median().item() if (d > 0).any() else 0.0))

res = {"length": L, "reps": a.reps, "input": a.input, "preset": a.preset,
       "ind": _IND, "det": _DET, "tag": a.tag,
       "flags": dict(GATHER=int(opt.GATHER_ONEHOT), HOIST=int(opt.HOIST_NEIGHBOURS),
                     NOPC=int(opt.SKIP_PAIR_COND), AUX=int(opt.NEED_AUX),
                     BATCHCOND=int(opt.BATCH_COND), DETSEG=int(opt.DET_SEGMENT))}
for k in ("pos", "local", "trajectory"):
    if k in outs[0]:
        s = stats(k)
        res[k] = s
        print(f"FLOOR {a.tag:<10} L={L:<4} {a.input:<5} {k:<11} "
              f"distinct={s['distinct']:<3}/{s['n_reps']} zero_pairs="
              f"{s['zero_pairs']}/{s['n_pairs']}  mean {s['mean']:.3e}  "
              f"median {s['median']:.3e}  p95 {s['p95']:.3e}  max {s['max']:.3e}"
              f"   nonzero: min {s['nonzero_min']:.3e} med {s['nonzero_median']:.3e}")
        print(f"      clusters {s['cluster_sizes']}")

if a.time:
    def bench(fn, it, warm=8):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e3
    with torch.no_grad():
        tf = bench(lambda: fast(data, prev, uniform=uniform), a.iters)
        tl = bench(lambda: fast(data, prev, need_aa=False, uniform=uniform), a.iters)
    res["full_ms"], res["lean_ms"] = tf, tl
    print(f"TIME  {a.tag:<10} L={L:<4} IND=[{_IND}] DET=[{_DET}] "
          f"full {tf:6.3f} ms  lean {tl:6.3f} ms")

if a.save:
    # `pos` and `local` only: `trajectory` is 6x `pos` and nothing is gated on
    # it, and the notebook's rule is nothing over 5 MB inside .agent-research.
    torch.save({"meta": res,
                "outs": [{k: v.cpu() for k, v in o.items() if k != "trajectory"}
                         for o in outs]}, a.save)
    print(f"saved {a.save}")
print("JSON " + json.dumps(res))

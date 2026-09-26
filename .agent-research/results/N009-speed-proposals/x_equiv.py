"""N009 equivalence harness for the proposals the parity fixtures cannot cover.

`tests/test_model_parity.py` drives the EAGER `Diffusion.forward`, so it covers
P3 and P6 (which live in the shared modules) but not P2/P4/P5, which are flags
on `forward_core`. This compares `forward_core` against itself with the flags
off, on real weights.

  P2 (need_aa=False)   -> pos/local/trajectory must be torch.equal            (a skip)
  P4+P5 (pair_cond=False) -> must be <= the 2e-5 parity floor, and P4 alone must be 0.0
  plus: the fourth condition-neighbour stage really is all -1 when flags are off
  plus: with two chains (flags set) the gate must NOT fire

    python .agent-research/results/N009-speed-proposals/x_equiv.py [--device cuda]
"""
import argparse, os, sys
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
import torch
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.blocks import extract_condition_neighbours
from salad_torch.modules.diffusion import uniform_draw_count

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
a = ap.parse_args()
torch.set_grad_enabled(False)

def _apply_inductor_env():
    """`SALAD_IND="combo_kernels=1,max_fusion_size=256"` - inductor knobs.

    Benchmark-only, and only for knobs that merge or enlarge kernels without
    changing any element's arithmetic (horizontal/vertical fusion). Anything
    that swaps a GEMM algorithm is NOT in this category and belongs to the
    fast preset, not the exact one.
    """
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
dev = torch.device(a.device)
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)
D = m.diffusion

L, A = a.length, 5 + cfg.augment_size
g = torch.Generator(device="cpu").manual_seed(7)
def rnd(*sh): return torch.randn(*sh, generator=g).to(dev)

def make(chains: int):
    z = torch.zeros(L, dtype=torch.long, device=dev)
    chain = (torch.arange(L, device=dev) * chains // L).long()
    pos = rnd(L, A, 3) * 4
    n_draws = uniform_draw_count(cfg)
    u = torch.rand(n_draws, L, L, generator=g).clamp_(1e-6, 1 - 1e-6).to(dev)
    return dict(
        pos_noised=pos, prev_pos=rnd(L, A, 3) * 4, prev_local=rnd(L, cfg.local_size),
        condition=rnd(L, cfg.local_size),
        dmap=torch.zeros(L, L, device=dev), omap=torch.zeros(L, L, 9, device=dev),
        dmap_mask=torch.zeros(L, L, dtype=torch.bool, device=dev),
        flags=torch.cat((((chain[:, None] != chain[None, :])).float()[..., None],
                         torch.zeros(L, L, 2, device=dev)), dim=-1),
        t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev),
        seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
        aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
        pos_gt=pos, resi=torch.arange(L, device=dev), chain=chain, batch=z,
        mask=torch.ones(L, dtype=torch.bool, device=dev),
        cyclic_mask=None, uniform=u)

ARGS = ("pos_noised prev_pos prev_local condition dmap omap dmap_mask flags t_pos "
        "t_seq seq_noised aa_gt pos_gt resi chain batch mask cyclic_mask uniform").split()
def run(d, **kw):
    return D.forward_core(*[d[k] for k in ARGS], **kw)

fails = []
FLOOR = 0.0          # set below from an identical-input rerun

def dmax(ref, got):
    return max((r.float() - h.float()).abs().max().item() for r, h in zip(ref, got))

def report(name, ref, got, exact):
    dev_max = dmax(ref, got)
    eq = all(torch.equal(r, h) for r, h in zip(ref, got))
    # A CHANGE is only a change if it exceeds the backend's own run-to-run
    # noise on identical input. On CUDA the neighbour chaining's `index_put_`
    # with duplicate indices and the segment reductions' atomicAdd are both
    # documented non-deterministic, so an absolute tolerance here would be
    # measuring the GPU, not the proposal.
    ok = (eq or dev_max <= max(FLOOR, 1e-12)) if exact else \
         (dev_max <= max(2e-5, FLOOR))
    fails.append(name) if not ok else None
    ratio = "" if FLOOR <= 0 else f"  = {dev_max / FLOOR:.2f}x the noise floor"
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<46} torch.equal={eq!s:<5} "
          f"max|d|={dev_max:.3e}  ({'bit-exact' if exact else 'floor 2e-5'}){ratio}")

d1 = make(1)                     # single chain: flags all false
print(f"single chain: flags.any()={bool(d1['flags'].any())}  "
      f"dmap_mask.any()={bool(d1['dmap_mask'].any())}")

# -- the structural claim P5 rests on -------------------------------------
pc = {"dmap": d1["dmap"], "omap": d1["omap"],
      "dmap_mask": d1["dmap_mask"], "flags": d1["flags"]}
nb = extract_condition_neighbours(16, 16, 16, 16)(
    d1["prev_pos"], pc, d1["resi"], d1["chain"], d1["batch"], d1["mask"],
    uniform=d1["uniform"][2], uniform2=d1["uniform"][3])
print(f"cond_neighbours {tuple(nb.shape)}: -1 in cols 0:48 = "
      f"{int((nb[:, :48] == -1).sum())}, in cols 48:64 = {int((nb[:, 48:] == -1).sum())} "
      f"of {nb[:, 48:].numel()}")
assert bool((nb[:, 48:] == -1).all()), "P5's premise is FALSE: block stage selected something"
nb48 = extract_condition_neighbours(16, 16, 16, 0)(
    d1["prev_pos"], None, d1["resi"], d1["chain"], d1["batch"], d1["mask"],
    uniform=d1["uniform"][2], uniform2=d1["uniform"][3])
print(f"K=48 selector reproduces the first 48 columns exactly: "
      f"{torch.equal(nb48, nb[:, :48])}")

# -- the control that has to come FIRST -----------------------------------
ref = run(d1)
FLOOR = dmax(ref[:3], run(d1)[:3])
print(f"\nNOISE FLOOR: the identical call run twice differs by {FLOOR:.3e} "
      f"on {a.device}")

report("P2  need_aa=False: pos/local/trajectory", ref[:3], run(d1, need_aa=False), True)

# P4 alone, at the site it lives: the dmap/flags terms of one pair block.
# (End to end it cannot be isolated from P5 - they share one runtime gate - so
# it is checked where the skip actually happens.)
blk = D.diffusion_stack.blocks[0]
nbr = extract_condition_neighbours(16, 16, 16, 16)(
    d1["prev_pos"], pc, d1["resi"], d1["chain"], d1["batch"], d1["mask"],
    uniform=d1["uniform"][2], uniform2=d1["uniform"][3])
with_pc = blk.pair_2(d1["prev_pos"], pc, nbr, d1["resi"], d1["chain"],
                     d1["batch"], d1["mask"])
without = blk.pair_2(d1["prev_pos"], None, nbr, d1["resi"], d1["chain"],
                     d1["batch"], d1["mask"])
report("P4  pair block: dmap/flags terms dropped", with_pc, without, True)

report("P4+P5 pair_cond=False (K 64->48)", ref[:3], run(d1, pair_cond=False)[:3], False)

# -- the gate must not fire when there IS pair conditioning ----------------
d2 = make(2)                     # two chains -> flags channel 0 is set
print(f"\ntwo chains: flags.any()={bool(d2['flags'].any())}")
nb2 = extract_condition_neighbours(16, 16, 16, 16)(
    d2["prev_pos"], {"dmap": d2["dmap"], "omap": d2["omap"],
                     "dmap_mask": d2["dmap_mask"], "flags": d2["flags"]},
    d2["resi"], d2["chain"], d2["batch"], d2["mask"],
    uniform=d2["uniform"][2], uniform2=d2["uniform"][3])
sel = int((nb2[:, 48:] != -1).sum())
print(f"block-contact stage selects {sel} real neighbours of {nb2[:, 48:].numel()} "
      f"-> the skip would be WRONG here, and the runtime gate refuses it")
assert sel > 0
ref2, alt2 = run(d2), run(d2, pair_cond=False)
dm = max((r - h).abs().max().item() for r, h in zip(ref2[:3], alt2[:3]))
print(f"[INFO] with flags set, taking the skip anyway moves pos by {dm:.3e} "
      f"-- which is why FastDenoiser derives the gate from the tensors")
report("P2  need_aa=False, two chains", ref2[:3], run(d2, need_aa=False)[:3], True)

# -- and the path that actually ships: two captured CUDA graphs ------------
if dev.type == "cuda":
    from salad_torch.modules.fast import FastDenoiser
    fd = FastDenoiser(m, L, device=dev, preset="exact").warmup(pair_cond=False)
    fdata = dict(pos=d1["pos_gt"], pos_noised=d1["pos_noised"], aa_gt=d1["aa_gt"],
                 seq=d1["seq_noised"], seq_noised=d1["seq_noised"],
                 residue_index=d1["resi"], chain_index=d1["chain"],
                 batch_index=d1["batch"], mask=d1["mask"],
                 t_pos=d1["t_pos"], t_seq=d1["t_seq"])
    fprev = {"pos": d1["prev_pos"], "local": d1["prev_local"]}
    u = d1["uniform"]
    with torch.no_grad():
        r1, _ = fd(fdata, fprev, uniform=u)
        r1b, _ = fd(fdata, fprev, uniform=u)
        r2, _ = fd(fdata, fprev, uniform=u, need_aa=False)
    g_floor = (r1["pos"] - r1b["pos"]).abs().max().item()
    g_lean = (r1["pos"] - r2["pos"]).abs().max().item()
    print(f"\nSHIPPING PATH (compiled + CUDA-graph, preset=exact):")
    print(f"  graph replay floor (same graph twice)   {g_floor:.3e}")
    print(f"  full vs lean graph, same inputs         {g_lean:.3e}   "
          f"torch.equal={torch.equal(r1['pos'], r2['pos'])}")
    if not torch.equal(r1["pos"], r2["pos"]):
        fails.append("P2 shipping path: lean graph's pos differs from full")

print("\n" + ("FAILED: " + ", ".join(fails) if fails else "OK"))
sys.exit(1 if fails else 0)

"""N011 - measure one FAST-preset candidate, in its own process.

One variant, one length, one process. Compiling several shapes or several
graph variants in one process degrades cudagraphs and makes every number look
the same (scripts/bench_one.py carries the same warning).

The harness deliberately does NOT go through `FastDenoiser`: that class is
being rewritten in parallel by the exact-path work (N009). It reproduces its
mechanism instead - `torch.compile(mode="default")` + one explicit whole-forward
`torch.cuda.CUDAGraph` over `Diffusion.forward_core`, static input buffers,
replay - so the numbers are comparable to the 4.95 ms baseline while being
immune to that file changing underneath.

Every variant is a monkeypatch applied BEFORE the compile, so Dynamo traces the
patched code. The unpatched fp32 eager `forward_core` is run first and kept as
the deviation reference.

Usage:  python x_fast.py --variant EXACT --length 100
"""
from __future__ import annotations

import argparse, os, sys, time
import torch

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)

from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.geometry import neighbours as NB
from salad_torch.modules import blocks as BL
from salad_torch.modules import diffusion as DF

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="EXACT")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--device", default="cuda")
A = ap.parse_args()

dev = torch.device(A.device)
cfg = get_config(A.config)
model = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(model, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{A.config}-200k.npz"),
           cfg, strict=True)
dif = model.diffusion

L = A.length
ATOMS = dif.atom_count
LS = cfg.local_size
g = torch.Generator(device=dev).manual_seed(0)


def make_inputs(dtype=torch.float32):
    # reseed so the reference call and the benchmarked call see the SAME input
    g.manual_seed(0)
    z = torch.zeros(L, dtype=torch.long, device=dev)
    # a plausible mid-trajectory state: a random compact cloud, sigma_data-scaled
    pos = torch.randn(L, ATOMS, 3, generator=g, device=dev) * 4.0
    return dict(
        pos_noised=pos.to(dtype), prev_pos=(pos * 0.9).to(dtype),
        prev_local=torch.zeros(L, LS, device=dev, dtype=dtype),
        condition=torch.zeros(L, LS, device=dev, dtype=dtype),
        dmap=torch.zeros(L, L, device=dev, dtype=dtype),
        omap=torch.zeros(L, L, 9, device=dev, dtype=dtype),
        dmap_mask=torch.zeros(L, L, device=dev, dtype=torch.bool),
        flags=torch.zeros(L, L, 3, device=dev, dtype=dtype),
        t_pos=torch.full((L,), 0.5, device=dev, dtype=dtype),
        t_seq=torch.ones(L, device=dev, dtype=dtype),
        seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
        aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
        pos_gt=pos.to(dtype),
        resi=torch.arange(L, device=dev), chain=z, batch=z,
        mask=torch.ones(L, dtype=torch.bool, device=dev),
        cyclic_mask=None,
        uniform=torch.rand(DF.uniform_draw_count(cfg), L, L, device=dev,
                           generator=g).clamp_(1e-6, 1 - 1e-6).to(dtype),
    )


ARGS = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
        "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt", "pos_gt",
        "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")

torch.manual_seed(0)
ref_in = make_inputs()
with torch.no_grad():
    REF = dif.forward_core(*[ref_in[k] for k in ARGS],
                           need_aux=True, need_aa=True, pair_cond=True)[0].float()

# ----------------------------------------------------------------- patches
NEED_AUX, NEED_AA, PAIR_COND = True, False, False   # the EXACT-preset defaults
DTYPE = torch.float32


def patch_topk():
    """Stage selection by `topk` instead of a full stable argsort.

    NOT bit-exact: argsort(stable=True) breaks ties by ascending index and
    `topk` does not specify its tie order, and ties are everywhere here (the
    sequence-index stage's distances are integers, so |i-j| ties between the
    two directions on every row). The SET of selected distances is the same;
    which of two equidistant residues is taken can differ.
    """
    def get_neighbours(count: int):
        def inner(distance, mask, neighbours=None):
            n = distance.shape[0]
            index = torch.arange(n, device=distance.device)
            distance = torch.where(mask.bool(), distance,
                                   torch.full_like(distance, float("inf")))
            if neighbours is not None:
                update = torch.where(
                    neighbours != -1,
                    torch.full_like(distance[index[:, None], neighbours], float("inf")),
                    distance[index[:, None], neighbours])
                distance = distance.clone()
                distance[index[:, None], neighbours] = update
            k = min(count, n)   # argsort+slice yields min(n, count) columns
            if k <= 0:
                knn = torch.zeros((n, 0), dtype=torch.long, device=distance.device)
            else:
                picked, knn = torch.topk(distance, k, dim=-1, largest=False,
                                         sorted=True)
                knn = torch.where(picked < float("inf"), knn,
                                  torch.full_like(knn, -1))
            if neighbours is not None:
                knn = torch.cat((neighbours, knn), dim=-1)
            return knn
        return inner
    NB.get_neighbours = get_neighbours
    BL.get_neighbours = get_neighbours


def patch_counts(cur=None, cond_spatial=None, cond_random=None):
    """Override the neighbour stage counts inside `forward_core`."""
    if cur is not None:
        orig = DF.extract_neighbours
        DF.extract_neighbours = lambda a, b, c: orig(*cur)
    if cond_spatial is not None:
        origp = DF.condition_neighbour_prologue
        DF.condition_neighbour_prologue = lambda a, b: origp(a, cond_spatial)
        orige = DF.extract_condition_neighbours
        DF.extract_condition_neighbours = (
            lambda a, b, c, d: orige(a, cond_spatial, cond_random or c, d))
    if cond_random is not None:
        origt = DF.condition_neighbour_tail
        DF.condition_neighbour_tail = lambda a, b: origt(cond_random, b)


def patch_stride(stride: int):
    """Reuse the current-branch neighbour list for `stride` consecutive blocks.

    The self-conditioning branch is already loop-invariant; this one is not -
    block b's list is built from block b-1's positions - so reusing it is a
    genuine approximation, and the whole point of the fast preset.
    """
    orig = DF.extract_neighbours
    state = {"n": 0, "val": None}

    def factory(a, b, c):
        inner = orig(a, b, c)

        def wrapped(*args, **kw):
            if state["n"] % stride == 0 or state["val"] is None:
                state["val"] = inner(*args, **kw)
            state["n"] += 1
            return state["val"]
        return wrapped
    DF.extract_neighbours = factory


def patch_condsame():
    """Give every block the SAME Gumbel draw on the self-conditioning branch.

    After the P6 hoist the condition branch's distance matrix, index stage and
    spatial stage are already shared across the six blocks; only the random
    stage's draw differs. Pinning that draw makes `cond_neighbours` one tensor
    for the whole trunk - and then `pair_2`'s SEVEN geometric feature tensors
    are built from identical inputs (`prev_pos`, the same indices) in all six
    blocks, so inductor's CSE should build them once instead of six times.
    Only the per-block weight matrices still differ.

    This is exactly P7's observation ("the condition branch is a prologue")
    obtained without P7's bmm refactor - at the price of six blocks sampling
    the same 16 random neighbours instead of six independent sets.

    `forward_core` indexes the pre-drawn stack as base = 1 + 3*block, so the
    current branch takes i = 1 mod 3 and the condition branch i = 2 mod 3
    (random) and i = 0 mod 3 (block contact). Only the latter two are pinned.
    """
    orig = DF._slice

    def _slice(stack, i):
        if i > 0 and i % 3 != 1:
            i = 2 if i % 3 == 2 else 3
        return orig(stack, i)
    DF._slice = _slice


V = A.variant
if V == "BASE":            # what the port did before N009: everything on
    NEED_AUX, NEED_AA, PAIR_COND = True, True, True
elif V == "EXACT":         # exact preset, non-output step (the fair baseline)
    pass
elif V == "EXACT_AA":      # exact preset, output step
    NEED_AA = True
elif V == "noaux":
    NEED_AUX = False
elif V == "topk":
    patch_topk()
elif V == "cur48":
    patch_counts(cur=(16, 16, 16))
elif V == "cur32":
    patch_counts(cur=(16, 8, 8))
elif V == "cond32":
    patch_counts(cond_spatial=8, cond_random=8)
elif V == "both32":
    patch_counts(cur=(16, 8, 8), cond_spatial=8, cond_random=8)
elif V == "both48":
    patch_counts(cur=(16, 16, 16), cond_spatial=16, cond_random=8)
elif V == "stride2":
    patch_stride(2)
elif V == "stride3":
    patch_stride(3)
elif V == "stride6":
    patch_stride(6)
elif V == "tf32":
    torch.set_float32_matmul_precision("high")
elif V == "fp16":
    DTYPE = torch.float16
elif V == "bf16":
    DTYPE = torch.bfloat16
elif V == "condsame":
    patch_condsame()
elif V == "condsame48":
    patch_condsame()
    patch_counts(cur=(16, 16, 16))
elif V == "condsame_s2":
    patch_condsame()
    patch_stride(2)
elif V == "ndraw":
    pass
elif V == "COMBO":
    NEED_AUX = False
    patch_topk()
    patch_counts(cur=(16, 16, 16))
    torch.set_float32_matmul_precision("high")
elif V == "COMBO32":
    NEED_AUX = False
    patch_topk()
    patch_counts(cur=(16, 8, 8), cond_spatial=8, cond_random=8)
    torch.set_float32_matmul_precision("high")
elif V == "tf32med":
    torch.set_float32_matmul_precision("medium")
elif V == "tf32_cur48":
    torch.set_float32_matmul_precision("high")
    patch_counts(cur=(16, 16, 16))
elif V == "tf32_both32":
    torch.set_float32_matmul_precision("high")
    patch_counts(cur=(16, 8, 8), cond_spatial=8, cond_random=8)
elif V == "tf32_condsame":
    torch.set_float32_matmul_precision("high")
    patch_condsame()
elif V == "tf32_condsame32":
    torch.set_float32_matmul_precision("high")
    patch_condsame()
    patch_counts(cur=(16, 8, 8), cond_spatial=8, cond_random=8)
elif V == "COMBO2":       # the candidate fast preset
    NEED_AUX = False
    patch_condsame()
    patch_topk()
    torch.set_float32_matmul_precision("high")
elif V == "COMBO3":       # COMBO2 + the current branch at K=48
    NEED_AUX = False
    patch_condsame()
    patch_topk()
    patch_counts(cur=(16, 16, 16))
    torch.set_float32_matmul_precision("high")
else:
    raise SystemExit(f"unknown variant {V}")

if DTYPE is not torch.float32:
    model.to(DTYPE)

# ----------------------------------------------------------------- run
torch.manual_seed(0)
static = make_inputs(DTYPE)
args = [static[k] for k in ARGS] + [NEED_AUX, NEED_AA, PAIR_COND]
compiled = torch.compile(dif.forward_core, mode="default", dynamic=False)

if dev.type != "cuda":
    # smoke mode: no compile, no graph - just exercise every patch eagerly so a
    # typo in a monkeypatch costs a login-node second and not a GPU job
    with torch.no_grad():
        static_out = dif.forward_core(*args)
    t_compile, ms = 0.0, float("nan")
else:
    t_compile = time.perf_counter()
    with torch.no_grad():
        for _ in range(3):
            out = compiled(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = compiled(*args)
    t_compile = time.perf_counter() - t_compile

    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(A.iters):
        graph.replay()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / A.iters * 1e3

pos = static_out[0].float()
dev_max = float((pos - REF).abs().max())
dev_rms = float(((pos - REF) ** 2).sum(-1).sqrt().mean())
nan = bool(torch.isnan(pos).any())
print(f"RESULT variant={V:<9} L={L:<4} dtype={str(DTYPE).split('.')[-1]:<8} "
      f"need_aux={int(NEED_AUX)} need_aa={int(NEED_AA)} pair_cond={int(PAIR_COND)} "
      f"ms={ms:7.3f}  dev_max={dev_max:.3e} A  dev_rms={dev_rms:.3e} A  "
      f"nan={nan}  compile={t_compile:.0f}s", flush=True)

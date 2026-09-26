"""N014 - is the BATCHED prologue itself run-to-run deterministic on CUDA?

N013's arms job measured a ~5.7x larger within-arm spread for the P7 forward
than for the baseline. Two explanations: (a) P7's fixed reassociation moves the
neighbour argsort nearer to ties, so the EXISTING scatter_add noise flips more
of them, or (b) `bmm` itself is non-deterministic (a split-K with atomics)
where six `mm`s were not, i.e. P7 ADDS a noise source.

This calls the prologue and the six sequential `pair_2`s on ONE frozen CUDA
input, `--reps` times each, and reports how many distinct bit-patterns each
produces. (b) predicts >1 pattern for the batched arm only.
"""
import argparse, sys
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
import torch                                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--k", type=int, default=48)
ap.add_argument("--pair-cond", action="store_true")
a = ap.parse_args()
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
from salad_torch.config import get_config                          # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict    # noqa: E402

dev = torch.device("cuda")
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/%s-200k.npz" % a.config),
           cfg, strict=True)
stack = m.diffusion.diffusion_stack
sp = stack.cond_pair_stack
L, A = a.length, 5 + cfg.augment_size
g = torch.Generator(device="cpu").manual_seed(3)
prev_pos = (torch.randn(L, A, 3, generator=g) * 4).to(dev)
resi = torch.arange(L, device=dev)
chain = torch.zeros(L, dtype=torch.long, device=dev)
batch = torch.zeros(L, dtype=torch.long, device=dev)
mask = torch.ones(L, dtype=torch.bool, device=dev)
pc = None
if a.pair_cond:
    pc = dict(dmap=torch.zeros(L, L, device=dev),
              dmap_mask=torch.zeros(L, L, dtype=torch.bool, device=dev),
              flags=torch.zeros(L, L, 3, device=dev),
              omap=torch.zeros(L, L, 9, device=dev))
nbr = torch.stack([torch.randint(-1, L, (L, a.k), generator=g).to(dev)
                   for _ in range(sp.depth)])
print("L=%d K=%d depth=%d reps=%d pair_cond=%s" % (L, a.k, sp.depth, a.reps, a.pair_cond))


def sigs(fn):
    seen = {}
    ref = None
    worst = 0.0
    for r in range(a.reps):
        out = fn()
        if ref is None:
            ref = out
        else:
            worst = max(worst, (ref.double() - out.double()).abs().max().item())
        seen.setdefault(hash(out.cpu().numpy().tobytes()), []).append(r)
    return len(seen), sorted((len(v) for v in seen.values()), reverse=True), worst


def seq():
    return torch.stack([stack.blocks[i].pair_2(prev_pos, pc, nbr[i], resi, chain,
                                               batch, mask)[0]
                        for i in range(sp.depth)])


def bat():
    return sp(prev_pos, pc, nbr, resi, chain, batch, mask)[0]


for name, fn in (("six sequential mm", seq), ("one batched bmm", bat)):
    n, cl, worst = sigs(fn)
    print("  %-20s distinct bit-patterns %d/%d  clusters %s  max|d| vs rep0 %.3e"
          % (name, n, a.reps, cl[:6], worst))

s, b = seq(), bat()
print("  batched vs sequential (same process): equal=%s  max|d|=%.6e"
      % (torch.equal(s, b), (s.double() - b.double()).abs().max()))

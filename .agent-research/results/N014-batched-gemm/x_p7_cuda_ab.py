"""N013/P7 - the batched condition-branch prologue against the sequential path.

Runs `forward_core` twice in ONE process with real weights, once with
`opt.BATCH_COND` off and once on (the flag is read at import, so it is poked
directly here - legitimate because nothing is compiled in this harness).

On CPU both arms are deterministic, so the number printed is the whole effect
of the change. On CUDA the identical-input rerun comes first, as the control.
"""
import argparse, sys
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--length", type=int, default=48)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--chains", type=int, default=1)
ap.add_argument("--pair-cond", action="store_true")
ap.add_argument("--cyclic", action="store_true")
ap.add_argument("--batches", type=int, default=1)
ap.add_argument("--mask-frac", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--reps", type=int, default=8, help="identical-rerun reps for the floor")
a = ap.parse_args()
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
from salad_torch import opt                                       # noqa: E402
from salad_torch.config import get_config                         # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict   # noqa: E402
from salad_torch.modules.diffusion import uniform_draw_count      # noqa: E402

dev = torch.device(a.device)
cfg = get_config(a.config)
if a.cyclic:
    cfg["cyclic"] = True
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)
D = m.diffusion
stack = D.diffusion_stack
print(f"ARM config={a.config} cyclic={bool(cfg.cyclic)} chains={a.chains} batches={a.batches} mask_frac={a.mask_frac} pair_cond={a.pair_cond} L={a.length} dev={a.device}")
print(f"config={a.config} multi_motif={bool(cfg.multi_motif)} "
      f"cond_pair_stack={'present' if stack.cond_pair_stack is not None else 'NONE'}")
if stack.cond_pair_stack is not None:
    print(f"weight cache in sync after load_model: {stack.cond_pair_stack.in_sync(stack.blocks)}")

L, A = a.length, 5 + cfg.augment_size
g = torch.Generator(device="cpu").manual_seed(a.seed)
rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
chain = (torch.arange(L, device=dev) * a.chains // L).long()
z = torch.zeros(L, dtype=torch.long, device=dev)
bidx = (torch.arange(L, device=dev) * a.batches // L).long()
mk = torch.ones(L, dtype=torch.bool, device=dev)
if a.mask_frac < 1.0:
    mk[int(L * a.mask_frac):] = False
pos = rnd(L, A, 3) * 4
u = torch.rand(uniform_draw_count(cfg), L, L, generator=g).clamp_(1e-6, 1 - 1e-6).to(dev)
flags = torch.cat(((chain[:, None] != chain[None, :]).float()[..., None],
                   torch.zeros(L, L, 2, device=dev)), dim=-1)
dmap_mask = torch.zeros(L, L, dtype=torch.bool, device=dev)
dmap = torch.zeros(L, L, device=dev)
if a.pair_cond:
    dmap = (rnd(L, L).abs() * 4 + 4)
    dmap = 0.5 * (dmap + dmap.T)
    dmap_mask = (rnd(L, L) > 1.0)
    dmap_mask = dmap_mask | dmap_mask.T
ARGS = dict(pos_noised=pos, prev_pos=rnd(L, A, 3) * 4, prev_local=rnd(L, cfg.local_size),
            condition=rnd(L, cfg.local_size), dmap=dmap,
            omap=torch.zeros(L, L, 9, device=dev), dmap_mask=dmap_mask, flags=flags,
            t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev),
            seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
            aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
            pos_gt=pos, resi=torch.arange(L, device=dev), chain=chain, batch=bidx,
            mask=mk,
            cyclic_mask=(torch.ones(L, dtype=torch.bool, device=dev) if cfg.cyclic else None),
            uniform=u)
ORDER = ("pos_noised prev_pos prev_local condition dmap omap dmap_mask flags t_pos "
         "t_seq seq_noised aa_gt pos_gt resi chain batch mask cyclic_mask uniform").split()
pa = [ARGS[k] for k in ORDER]

pc_off = not (a.pair_cond or a.chains > 1)
print(f"chains={a.chains} pair_cond_tensors={'set' if not pc_off else 'empty'} "
      f"-> running forward_core(pair_cond={not pc_off})")


def run(flag, need_aa):
    opt.BATCH_COND = flag
    import salad_torch.modules.blocks as B
    return D.forward_core(*pa, True, need_aa, not pc_off)


for need_aa in (False, True):
    opt.BATCH_COND = False
    ref = run(False, need_aa)
    ctls = [run(False, need_aa) for _ in range(a.reps)]
    ctl = ctls[0]
    assert stack.batched_cond_pair_available() is False
    opt.BATCH_COND = True
    assert stack.batched_cond_pair_available() is (not cfg.multi_motif)
    got = run(True, need_aa)
    opt.BATCH_COND = False
    names = ("pos", "local", "trajectory") if not need_aa else (
        "pos", "local", "trajectory", "aa", "aa_features", "dssp",
        "dnbr", "raw_angles", "angles", "angle_pos", "aatype")
    print(f"\n--- need_aa={need_aa} ---")
    for i, nm in enumerate(names):
        r, c_, h = ref[i], ctl[i], got[i]
        if not torch.is_tensor(r) or not r.dtype.is_floating_point:
            print(f"  {nm:<12} int/other  equal(ref,batched)={torch.equal(r, h)}")
            continue
        noise = max((r.double() - c[i].double()).abs().max().item() for c in ctls)
        d = (r.double() - h.double()).abs().max().item()
        print(f"  {nm:<12} identical-rerun {noise:.3e}   batched-vs-sequential "
              f"{d:.3e}   equal={torch.equal(r, h)}")

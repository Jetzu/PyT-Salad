"""R03: adversarial tests of salad_torch/modules/fast.py (FastDenoiser).

Reference = the UNCOMPILED `Diffusion.forward_core` fed exactly the same
tensors, so compiled-vs-eager is an apples-to-apples comparison (the ergonomic
`Diffusion.forward` cannot be pinned: it leaves `uniform2` undrawn).
"""
import os, sys, torch, numpy as np
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"; sys.path.insert(0, ROOT)
torch.set_float32_matmul_precision("highest")
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser, _CORE_ARGS
from salad_torch.modules.diffusion import uniform_draw_count

dev = torch.device("cuda"); cfg = get_config("default_vp")
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
A = 5 + cfg.augment_size; ND = uniform_draw_count(cfg)
OUT = ("pos","local","trajectory","aa","aa_features","dssp","distogram_neighbours",
       "raw_angles","angles","angle_pos","aatype")

def mkdata(L, chains=1, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.zeros(L, dtype=torch.long, device=dev)
    pos = torch.randn(L, A, 3, device=dev, generator=g) * 4
    ch = (torch.arange(L, device=dev) // max(1, L // chains)).clamp(max=chains-1)
    d = dict(pos=pos, pos_noised=pos.clone(),
             aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
             seq=torch.full((L,), 20, dtype=torch.long, device=dev),
             seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
             residue_index=torch.arange(L, device=dev), chain_index=ch, batch_index=z,
             mask=torch.ones(L, dtype=torch.bool, device=dev),
             t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
    p = {"pos": torch.zeros(L, A, 3, device=dev), "local": torch.zeros(L, cfg.local_size, device=dev)}
    return d, p

def ref(d, p, U):
    """Uncompiled forward_core with the condition computed FRESH from `d`."""
    with torch.no_grad():
        c = m.prepare_condition(d); pc = c["pair_condition"]
        o = m.diffusion.forward_core(
            d["pos_noised"], p["pos"], p["local"], c["condition"], pc["dmap"], pc["omap"],
            pc["dmap_mask"], pc["flags"], d["t_pos"], d["t_seq"], d["seq_noised"], d["aa_gt"],
            d["pos"], d["residue_index"], d["chain_index"], d["batch_index"], d["mask"],
            d.get("cyclic_mask") if cfg.cyclic else None, U)
    return dict(zip(OUT, o))

def rep(t, ok, msg): print(f"[{'ok  ' if ok else 'FAIL'}] {t:<36} {msg}", flush=True)
def dp(a, b): return float((a - b).norm(dim=-1).max())

L = 64
g = torch.Generator(device=dev).manual_seed(0)
U = torch.rand(ND, L, L, device=dev, generator=g).clamp_(1e-6, 1-1e-6)
d0, p0 = mkdata(L)


print("\n=== C. static input buffers alias/mutate the CALLER's tensors ===")
f = FastDenoiser(m, L, device=dev).warmup()
dA, pA = mkdata(L, seed=1); dB, pB = mkdata(L, seed=2)
with torch.no_grad(): _ = f(dA, pA, uniform=U)[0]
snap = dA["pos_noised"].clone()
with torch.no_grad(): _ = f(dB, pB, uniform=U)[0]
rep("C1 caller data untouched (same L)", torch.equal(dA["pos_noised"], snap),
    f"max|d| = {float((dA['pos_noised']-snap).abs().max()):.3e}")
# the None-valued static slot: cyclic_mask
cfgc = get_config("default_vp"); cfgc.cyclic = True
mc = StructureDiffusionPredict(cfgc).eval().to(dev)
load_model(mc, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfgc, strict=True)
fc = FastDenoiser(mc, L, device=dev).warmup()
dC, _ = mkdata(L); dC["cyclic_mask"] = torch.zeros(L, dtype=torch.bool, device=dev)
dD, _ = mkdata(L); dD["cyclic_mask"] = torch.ones(L, dtype=torch.bool, device=dev)
with torch.no_grad():
    _ = fc(dC, p0, uniform=U)[0]; _ = fc(dD, p0, uniform=U)[0]
rep("C2 caller cyclic_mask untouched", int(dC["cyclic_mask"].sum()) == 0,
    f"dC['cyclic_mask'].sum() = {int(dC['cyclic_mask'].sum())} of {L} (expected 0)")

print("\n=== D. holding a returned tensor across the next call ===")
f = FastDenoiser(m, L, device=dev).warmup()
with torch.no_grad(): r1 = f(dA, pA, uniform=U)[0]
held = r1["pos"]; hc = held.clone()
with torch.no_grad(): _ = f(dB, pB, uniform=U)[0]
rep("D1 held output not overwritten", torch.equal(held, hc),
    f"max|d| = {float((held-hc).abs().max()):.3e} A")

print("\n=== E. predict_aa=False: is atom_pos built from the returned aatype? ===")
f = FastDenoiser(m, L, device=dev).warmup()
with torch.no_grad(): rf = f(dA, pA, uniform=U, predict_aa=False)[0]
rr = ref(dA, pA, U)
_, _, want_apos = m.diffusion.angle_head(dA["aa_gt"], rr["local"], rr["pos"])
rep("E1 aatype == aa_gt", bool((rf["aatype"] == dA["aa_gt"]).all()), "")
rep("E2 atom_pos rebuilt from aa_gt", dp(rf["atom_pos"], want_apos) < 1e-4,
    f"|fast - angle_head(aa_gt)| = {dp(rf['atom_pos'], want_apos):.3f} A   "
    f"|fast - angle_head(predicted)| = {dp(rf['atom_pos'], rr['angle_pos']):.2e} A")

print("\n=== F. aa_condition (--template_aa) ===")
d_aac = {**dA, "aa_condition": torch.zeros(L, dtype=torch.long, device=dev)}   # all ALA
f = FastDenoiser(m, L, device=dev).warmup()
with torch.no_grad(): rf = f(d_aac, pA, uniform=U)[0]
rep("F1 aatype forced to aa_condition", bool((rf["aatype"] == 0).all()),
    f"{int((rf['aatype'] != 0).sum())}/{L} residues ignore aa_condition")

print("\n=== G. length differing from construction ===")
f = FastDenoiser(m, L, device=dev).warmup()
dL, pL = mkdata(L + 13)
try:
    with torch.no_grad(): r = f(dL, pL)[0]
    rep("G1 mismatched L", False, f"NO ERROR; pos shape {tuple(r['pos'].shape)} for L={L+13}")
except Exception as e:
    rep("G1 mismatched L (crash)", True, f"{type(e).__name__}: {str(e)[:100]}")
# same, with an explicitly sized uniform so the default-draw shape is not the blocker
try:
    U2 = torch.rand(ND, L+13, L+13, device=dev).clamp_(1e-6, 1-1e-6)
    with torch.no_grad(): r = f(dL, pL, uniform=U2)[0]
    ok = tuple(r["pos"].shape) == (L+13, A, 3)
    rep("G2 mismatched L + sized uniform", False,
        f"NO ERROR; pos shape {tuple(r['pos'].shape)} (expected {(L+13, A, 3)})")
except Exception as e:
    rep("G2 mismatched L + sized uniform", True, f"{type(e).__name__}: {str(e)[:100]}")

print("\n=== H. interleaving two FastDenoiser instances ===")
f1 = FastDenoiser(m, L, device=dev).warmup(); f2 = FastDenoiser(m, L, device=dev).warmup()
with torch.no_grad():
    a1 = f1(dA, pA, uniform=U)[0]["pos"].clone(); b1 = f2(dB, pB, uniform=U)[0]["pos"].clone()
    a2 = f1(dA, pA, uniform=U)[0]["pos"].clone(); b2 = f2(dB, pB, uniform=U)[0]["pos"].clone()
rep("H1 f1 reproducible", float((a1-a2).abs().max()) < 1e-5, f"max|d| = {float((a1-a2).abs().max()):.3e}")
rep("H2 f2 reproducible", float((b1-b2).abs().max()) < 1e-5, f"max|d| = {float((b1-b2).abs().max()):.3e}")
rep("H3 f1 == eager", dp(a1, ref(dA, pA, U)["pos"]) < 1e-4, f"max|d| = {dp(a1, ref(dA,pA,U)['pos']):.3e}")

print("\n=== I. compute_distogram=True is silently dropped ===")
with torch.no_grad(): r = f1(dA, pA, uniform=U, compute_distogram=True)[0]
rep("I1 distogram present", "distogram" in r, f"keys: {'distogram' in r}")

print("\n=== J. did CUDA graphs actually engage? ===")
import torch._inductor.cudagraph_trees as ct
try:
    cont = ct.get_container(0)
    print("     cudagraph container:", cont, " tree roots:",
          None if cont is None else len(getattr(cont, 'tree_manager', None).roots) if getattr(cont,'tree_manager',None) else "no manager")
except Exception as e:
    print("     probe failed:", e)
print("\ndone")

"""N021 - is design `j` of a PACKED batch the design it would be alone?

The controlled A/B the question needs: both arms get the SAME pre-drawn
randomness, design for design, over the WHOLE production schedule, so the only
free variable left is a design's offset in the flat residue axis.

* every design owns a `torch.Generator`; at every step it draws its own
  ``noise`` (L, A, 3), ``seq_noise`` (L,) and ``uniform`` (draws, L, L) from it,
  in that fixed order;
* the UNPACKED arm feeds those straight in;
* the PACKED arm concatenates the noise and assembles the uniform fields
  BLOCK-DIAGONALLY into (draws, B*L, B*L), filling the off-diagonal blocks with
  0.5 (any legal uniform does: every cross-member pair is masked to +inf by
  `same_item` before selection, which is precisely what this checks).

Run under ``SALAD_OPT_DETSEG=1``: without it an identical command run twice
already differs in ~48 % of residues over a full rollout (N009/N019), so an
atomics-path comparison cannot discriminate and it would be dishonest to lean
on one.

Prints `DIV ...` lines, one per design.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n021tree")
sys.path.insert(0, ROOT)

from salad_torch.config import get_config                                # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.sampling import (SamplerConfig, make_initial_data,  # noqa: E402
                                            pack_designs, sample_group,
                                            split_packed_snapshot)
from salad_torch.diffusion.schedules import parse_num_aa                 # noqa: E402
from salad_torch.modules.diffusion import uniform_draw_count             # noqa: E402
from salad_torch.modules.fast import GroupFastDenoiser, PRESETS          # noqa: E402
from salad_torch.diffusion.forward import (diffuse_atom_cloud,           # noqa: E402
                                           diffuse_sequence)
from salad_torch.modules.model import StructureDiffusionPredict          # noqa: E402


def apply_diffusion_std(cfg, data, std_normal, seq_uniform):
    """`model.apply_diffusion` with the randomness INJECTED AS STANDARD NORMALS.

    `apply_diffusion(noise=...)` takes its argument as the already-SCALED
    perturbation, so feeding it a raw `randn` would run the rollout at a
    cloud std of 1 instead of the per-design `position_std` (~10-20 A) - a
    different, and wrong, trajectory. `noise_std_normal` is the hook for the
    unscaled draw, and it is not exposed through `apply_diffusion`, so the
    two scaling lines are reproduced here. `vp` (the shipped `default_vp`)
    takes `cloud_std=None`, i.e. `position_std` per batch segment.
    """
    if cfg.diffusion_kind == "edm":
        raise NotImplementedError("this probe is the vp/vpfixed/flow path")
    cloud_std = data.get("cloud_std")
    if cloud_std is None:
        cloud_std = (cfg.sigma_data
                     if cfg.diffusion_kind in ("vpfixed", "flow") else None)
    pos_noised = diffuse_atom_cloud(
        data["pos"], data["mask"], data["batch_index"], data["t_pos"],
        cloud_std=cloud_std, flow=cfg.diffusion_kind == "flow",
        noise_std_normal=std_normal)
    seq_noised, corrupt_aa = diffuse_sequence(data["seq"], data["t_seq"],
                                              uniform=seq_uniform)
    return {"pos_noised": pos_noised, "seq_noised": seq_noised,
            "t_pos": data["t_pos"], "t_seq": data["t_seq"],
            "corrupt_aa": corrupt_aa}

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="fast")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--seed", type=int, default=4242)
a = ap.parse_args()

dev = torch.device("cuda")
cfg_model = get_config(a.config)
L, B = a.length, a.batch_size
model = StructureDiffusionPredict(cfg_model).eval().to(dev)
load_model(model, load_param_npz(
    f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg_model, strict=True)
A = model.diffusion.atom_count
n_draws = uniform_draw_count(cfg_model)
scale = PRESETS[a.preset].get("steps_scale", 1.0)
num_steps = int(round(500 * scale))
out_at = int(round(400 * scale))
scfg = SamplerConfig(num_steps=num_steps, out_steps=(out_at,), lean_steps=True)
num_aa, resi, chain, _, _ = parse_num_aa(str(L))


def fresh_gens():
    gs = []
    for j in range(B):
        g = torch.Generator(device=dev)
        g.manual_seed((a.seed * 1_000_003 + j) & 0x7FFF_FFFF_FFFF_FFFF)
        gs.append(g)
    return gs


def draw(g):
    """One design's randomness for ONE step, in a fixed order."""
    noise = torch.randn(L, A, 3, device=dev, generator=g)
    seq_noise = torch.rand(L, device=dev, generator=g)
    uniform = torch.rand(n_draws, L, L, device=dev,
                         generator=g).clamp_(1e-6, 1 - 1e-6)
    return noise, seq_noise, uniform


def run(batch_size: int):
    """Generate all B designs with batch_size `batch_size` (B or 1)."""
    members = B // batch_size
    group = GroupFastDenoiser(model, L, group_size=members, device=dev,
                              preset=a.preset,
                              batch_size=batch_size).warmup(pair_cond=False)
    gens = fresh_gens()
    datas, prevs, lens = [], [], []
    for m in range(members):
        per = [make_initial_data(num_aa, resi, chain, None, device=dev)
               for _ in range(batch_size)]
        pd, pp, ln = pack_designs([d for d, _ in per], [p for _, p in per])
        datas.append(pd)
        prevs.append(pp)
        lens.append(ln)
    gate = group.pin_conditions(datas)

    def step_fn_group(ds, ps, gsx, *, need_aa=True):
        with torch.no_grad():
            fed, unis = [], []
            for m, d in enumerate(ds):
                base = m * batch_size
                draws = [draw(gens[base + j]) for j in range(batch_size)]
                noise = torch.cat([x[0] for x in draws], 0)
                seqn = torch.cat([x[1] for x in draws], 0)
                N = L * batch_size
                if batch_size == 1:
                    uni = draws[0][2]
                else:
                    uni = torch.full((n_draws, N, N), 0.5, device=dev)
                    for j, (_, _, u) in enumerate(draws):
                        uni[:, j * L:(j + 1) * L, j * L:(j + 1) * L] = u
                nz = apply_diffusion_std(cfg_model, d, noise, seqn)
                fed.append(dict(d, **nz))
                unis.append(uni)
            outs, newp = [], []
            for m, mem in enumerate(group.members):
                inputs, _g = mem._prepare_inputs(fed[m], ps[m], uniform=unis[m])
                mem._copy_in(inputs)
            key = (need_aa, bool(gate))
            if key not in group._graphs:
                group._build(need_aa, bool(gate))
            graph, gouts = group._graphs[key]
            graph.replay()
            for m, mem in enumerate(group.members):
                out = tuple(o.clone() if torch.is_tensor(o) else o
                            for o in gouts[m])
                r, p = mem._postprocess(out, fed[m], need_aa=need_aa,
                                        predict_aa=True)
                outs.append(r)
                newp.append(p)
            return outs, newp

    snaps = sample_group(step_fn_group, datas, prevs, scfg,
                         generators=[None] * members)
    designs = []
    for m in range(members):
        snap = dict(snaps[m][-1])
        designs += split_packed_snapshot(snap, lens[m])
    del group
    torch.cuda.empty_cache()
    return designs


print(f"# preset={a.preset} L={L} B={B} steps={num_steps} out={out_at} "
      f"DETSEG={os.environ.get('SALAD_OPT_DETSEG')}", flush=True)
alone = run(1)
packed = run(B)
for j in range(B):
    dp = (alone[j]["pos"] - packed[j]["pos"]).abs().max().item()
    dca = (alone[j]["pos"][:, 1] - packed[j]["pos"][:, 1]).abs().max().item()
    daa = int((alone[j]["aatype"] != packed[j]["aatype"]).sum().item())
    bits = bool(torch.equal(alone[j]["pos"], packed[j]["pos"]))
    print(f"DIV L={L} B={B} design={j} max_dpos={dp:.6e} max_dca={dca:.6e} "
          f"aatype_differ={daa}/{L} bit_identical={bits}", flush=True)
# a within-arm control: the same command twice. Under DETSEG this must be 0.
again = run(1)
for j in range(B):
    dp = (alone[j]["pos"] - again[j]["pos"]).abs().max().item()
    daa = int((alone[j]["aatype"] != again[j]["aatype"]).sum().item())
    print(f"FLOOR L={L} B=1 design={j} max_dpos={dp:.6e} "
          f"aatype_differ={daa}/{L}", flush=True)

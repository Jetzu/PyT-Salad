"""N021 - the (batch_size x group_size x L) trade-off surface.

ONE (preset, L, batch_size, group_size) per PROCESS, deliberately: N009/N016
learned that several shapes - and N019 that several graph captures - in one
process degrade cudagraphs and produce flat, false curves. The inductor cache
is shared through TORCHINDUCTOR_CACHE_DIR so only the first process at a given
(L, batch_size) pays the compile.

Everything timed here is timed INSIDE a captured graph replay or around a real
`sample_group` loop - never around an eager op.

Prints one `CELL ...` line of key=value pairs, or `CELL_OOM`/`CELL_FAIL`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n021tree")
sys.path.insert(0, ROOT)

from salad_torch.config import get_config                                # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.sampling import (SamplerConfig, make_initial_data,  # noqa: E402
                                            pack_designs, sample_group)
from salad_torch.diffusion.schedules import parse_num_aa                 # noqa: E402
from salad_torch.modules.fast import (GroupFastDenoiser, PRESETS,        # noqa: E402
                                      memory_bound_group)
from salad_torch.modules.model import (StructureDiffusionPredict,        # noqa: E402
                                       apply_diffusion)

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="fast")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--batch_size", type=int, default=1)
ap.add_argument("--group_size", default="1")
ap.add_argument("--config", default="default_vp")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--host_steps", type=int, default=30)
a = ap.parse_args()

dev = torch.device("cuda")
cfg_model = get_config(a.config)
L, B = a.length, a.batch_size
props = torch.cuda.get_device_properties(dev)
tag = (f"preset={a.preset} L={L} B={B} Greq={a.group_size}")

try:
    model = StructureDiffusionPredict(cfg_model).eval().to(dev)
    load_model(model, load_param_npz(
        f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg_model,
        strict=True)

    torch.cuda.reset_peak_memory_stats(dev)
    t_w = time.perf_counter()
    gs = a.group_size if a.group_size == "auto" else int(a.group_size)
    group = GroupFastDenoiser(model, L, group_size=gs, device=dev,
                              preset=a.preset, batch_size=B).warmup(pair_cond=False)
    warm_s = time.perf_counter() - t_w
    G = group.group_size
    per_replay = group.designs_per_replay

    def ev(fn, reps, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / reps

    lean_g = group._graphs.get((False, False)) or group._graphs.get((False, True))
    full_g = group._graphs.get((True, False)) or group._graphs.get((True, True))
    replay_lean = ev(lean_g[0].replay, a.reps) if lean_g else float("nan")
    replay_full = ev(full_g[0].replay, a.reps) if full_g else float("nan")

    # ---- the real production loop, host work included -----------------
    num_aa, resi, chain, is_cyclic, cyc = parse_num_aa(str(L))
    datas, prevs, gens = [], [], []
    for m in range(G):
        per = [make_initial_data(num_aa, resi, chain, None, device=dev)
               for _ in range(B)]
        pd, pp, _lens = pack_designs([d for d, _ in per], [p for _, p in per])
        datas.append(pd)
        prevs.append(pp)
        g = torch.Generator(device=dev)
        g.manual_seed(1000 + m)
        gens.append(g)
    gate = group.pin_conditions(datas)

    def step_fn_group(ds, ps, gsx, *, need_aa=True):
        with torch.no_grad():
            noised = [apply_diffusion(cfg_model, d, generator=gg)
                      for d, gg in zip(ds, gsx)]
            fed = [dict(d, **nz) for d, nz in zip(ds, noised)]
            return group(fed, ps, need_aa=need_aa, pair_cond=gate, generators=gsx)

    scfg = SamplerConfig(num_steps=a.host_steps, out_steps=(a.host_steps - 1,),
                         lean_steps=True)
    sample_group(step_fn_group, datas, prevs, scfg, generators=gens)   # warm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sample_group(step_fn_group, datas, prevs, scfg, generators=gens)
    torch.cuda.synchronize()
    host_ms = (time.perf_counter() - t0) * 1000.0 / a.host_steps

    peak_res = torch.cuda.max_memory_reserved(dev)
    steps = int(round(500 * PRESETS[a.preset].get("steps_scale", 1.0)))
    out_at = int(round(400 * PRESETS[a.preset].get("steps_scale", 1.0)))
    sec_per_design = ((out_at * host_ms + (replay_full - replay_lean))
                      / 1000.0) / per_replay
    print("CELL "
          f"{tag} B_eff={B} G={G} designs_per_replay={per_replay} "
          f"replay_lean_ms={replay_lean:.4f} replay_full_ms={replay_full:.4f} "
          f"replay_lean_per_design_ms={replay_lean/per_replay:.4f} "
          f"host_step_ms={host_ms:.4f} "
          f"host_step_per_design_ms={host_ms/per_replay:.4f} "
          f"designs_per_s={per_replay/(host_ms/1000.0):.4f} "
          f"sec_per_design={sec_per_design:.4f} "
          f"peak_reserved_MiB={peak_res/2**20:.1f} "
          f"peak_reserved_MiB_per_design={peak_res/2**20/per_replay:.1f} "
          f"per_member_MiB={(group.per_member_bytes or 0)/2**20:.1f} "
          f"warmup_s={warm_s:.1f} sm={props.multi_processor_count} "
          f"mem_bound_members={memory_bound_group(group.per_member_bytes, dev)}",
          flush=True)
except torch.cuda.OutOfMemoryError as exc:
    print(f"CELL_OOM {tag} {str(exc)[:120]}", flush=True)
except RuntimeError as exc:
    kind = "CELL_OOM" if "out of memory" in str(exc).lower() else "CELL_FAIL"
    print(f"{kind} {tag} {str(exc)[:200]}", flush=True)

"""N019 - speed AND memory vs group size N, one (preset, L, N) per process.

One process per point, deliberately: N009/N016 learned that putting several
shapes (and here, several graph captures) in one process degrades cudagraphs
and produces flat, false curves. The inductor cache is shared through
TORCHINDUCTOR_CACHE_DIR, so only the first process at a given (preset, L)
pays the compile.

Everything timed here is timed INSIDE a captured graph replay or around a
real `sample_group` loop - never around an eager op, which is how a previous
agent measured CPU dispatch and called it kernel time.

Prints one `CURVE ...` line of key=value pairs.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n019tree")
sys.path.insert(0, ROOT)

from salad_torch.config import get_config                                # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.sampling import (SamplerConfig, make_initial_data,  # noqa: E402
                                            sample_group)
from salad_torch.diffusion.schedules import parse_num_aa                 # noqa: E402
from salad_torch.modules.fast import (GroupFastDenoiser, PRESETS,        # noqa: E402
                                      memory_bound_group)
from salad_torch.modules.model import (StructureDiffusionPredict,        # noqa: E402
                                       apply_diffusion)

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="exact")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--group_size", default="1")
ap.add_argument("--config", default="default_vp")
ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--host_steps", type=int, default=40)
a = ap.parse_args()

dev = torch.device("cuda")
cfg_model = get_config(a.config)
L = a.length
model = StructureDiffusionPredict(cfg_model).eval().to(dev)
load_model(model, load_param_npz(
    f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg_model, strict=True)

torch.cuda.reset_peak_memory_stats(dev)
base_reserved = torch.cuda.memory_reserved(dev)
t_w = time.perf_counter()
gs = a.group_size if a.group_size == "auto" else int(a.group_size)
group = GroupFastDenoiser(model, L, group_size=gs, device=dev,
                          preset=a.preset).warmup(pair_cond=False)
warm_s = time.perf_counter() - t_w
n = group.group_size
props = torch.cuda.get_device_properties(dev)


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


# ---- pure graph replay, N designs per replay --------------------------
lean_g = group._graphs.get((False, False)) or group._graphs.get((False, True))
full_g = group._graphs.get((True, False)) or group._graphs.get((True, True))
replay_lean = ev(lean_g[0].replay, a.reps) if lean_g else float("nan")
replay_full = ev(full_g[0].replay, a.reps) if full_g else float("nan")

# ---- the real production loop, host work included ---------------------
num_aa, resi, chain, is_cyclic, cyc = parse_num_aa(str(L))
datas, prevs, gens = [], [], []
for i in range(n):
    d, p = make_initial_data(num_aa, resi, chain, None, device=dev)
    datas.append(d)
    prevs.append(p)
    g = torch.Generator(device=dev)
    g.manual_seed(1000 + i)
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
# the run stops at out_steps[-1], so it is out_at+1 steps of which one is full
sec_per_design = ((out_at * (host_ms - 0) + (replay_full - replay_lean))
                  / 1000.0) / n

print("CURVE "
      f"preset={a.preset} L={L} N={n} requested={a.group_size} "
      f"replay_lean_ms={replay_lean:.4f} replay_lean_per_design_ms={replay_lean/n:.4f} "
      f"replay_full_ms={replay_full:.4f} "
      f"host_step_ms={host_ms:.4f} host_step_per_design_ms={host_ms/n:.4f} "
      f"sec_per_design={sec_per_design:.4f} "
      f"peak_reserved_MiB={peak_res/2**20:.1f} "
      f"per_member_MiB={(group.per_member_bytes or 0)/2**20:.1f} "
      f"warmup_s={warm_s:.1f} "
      f"sm={props.multi_processor_count} totmem_GiB={props.total_memory/2**30:.2f} "
      f"mem_bound_N={memory_bound_group(group.per_member_bytes, dev)}",
      flush=True)

"""N030 whole-forward benchmark + kernel census. ONE arm, ONE length, ONE process.

Compiling several shapes or configs in one process degrades cudagraphs and once
produced a completely false "flat 17.7 ms" (N009/N012 method note), so the
sbatch puts every (arm, length) in its own python. Derived from N020's
x_bench.py, plus a full per-kernel-name GEMM census (the "are all 301 cuBLAS
launches necessary" question) and the exact-filter report.

The arm is applied by MUTATING THE PRESET, because `fast.py` resolves
`max_autotune_gemm` as `preset AND env` and every shipped preset says False -
setting only `SALAD_OPT_GEMMTPL=1` would leave templates OFF and the two arms
would time identically (the trap that has already cost this project a job).
The resolved value is asserted and printed.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The preset, NOT the environment, selects the template mode in this harness.
# `opt.resolve_gemm_templates` lets the environment NARROW the mode, and with
# `SALAD_OPT_GEMMTPL=1` still set from an earlier version of the sbatch the
# `exact` arm resolved to `all` and `FastDenoiser.__init__` UNINSTALLED the
# filter that had just been installed - the arm timed -14.2 % with a filter
# report of `shapes: 0`, i.e. it never ran. Exactly the trap the brief warns
# about. Popped here, before anything imports `salad_torch.opt` (which reads
# the environment once, at import).
import os as _os
_os.environ.pop("SALAD_OPT_GEMMTPL", None)

import torch
import x_arm
from torch.profiler import ProfilerActivity, profile

from x_common import build  # noqa: E402

torch.set_grad_enabled(False)
TAG = os.environ.get("N030_TAG", "?")
L = int(os.environ.get("N030_LEN", "100"))
PRESET = os.environ.get("N030_PRESET", "exact")
ARM = os.environ.get("N030_ARM", "off")
REPS = int(os.environ.get("N030_REPS", "3"))

m, cfg, _ = build(L, device="cuda")
from salad_torch import opt                                       # noqa: E402
from salad_torch.modules.fast import PRESETS, FastDenoiser        # noqa: E402

MODE = "off"
for _f in ARM.split("+"):
    if _f == "all":
        MODE = "all"
    elif _f.startswith("exact"):
        MODE = "exact"
    elif _f.startswith("stable"):
        MODE = "stable"
WANT_TPL = MODE != "off"
PRESETS[PRESET]["gemm_templates"] = MODE if WANT_TPL else False
# N030: `decompose_small_bmm` is now ALSO resolved inside `FastDenoiser.__init__`
# from `PRESETS[PRESET]`, constructed AFTER `x_arm.apply` below - so without
# this, `FastDenoiser.__init__` would silently CLOBBER whatever
# `x_arm.apply("decomp32")` had just set back to off (the preset's default),
# exactly the "arm that looks on and is not" trap this project keeps hitting.
DEC = 0
for _f in ARM.split("+"):
    if _f.startswith("decomp"):
        DEC = int(_f[6:]) if _f[6:] else 32
PRESETS[PRESET]["decompose_small_bmm"] = DEC if DEC else False
ARM_INFO = x_arm.apply(ARM)
print("[arm] " + json.dumps(ARM_INFO), flush=True)

dev = torch.device("cuda")
A = 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev)
g = torch.Generator(device="cuda").manual_seed(0)
pos = torch.randn(L, A, 3, device=dev, generator=g) * 4
data = dict(pos=pos, pos_noised=pos,
            aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
            seq=torch.full((L,), 20, dtype=torch.long, device=dev),
            seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
            residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
            mask=torch.ones(L, dtype=torch.bool, device=dev),
            cyclic_mask=torch.zeros(L, dtype=torch.bool, device=dev),
            t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
prev = {"pos": torch.zeros(L, A, 3, device=dev),
        "local": torch.zeros(L, cfg.local_size, device=dev)}


def bench(fn, it=60, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3


out = {"tag": TAG, "L": L, "preset": PRESET, "arm": ARM,
       "mm_mode": opt.MM_MODE, "env_gemmtpl": opt.GEMM_TEMPLATES,
       "detseg": opt.DET_SEGMENT,
       "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
t0 = time.perf_counter()
fast = FastDenoiser(m, L, device=dev, preset=PRESET).warmup(pair_cond=False)
out["compile_s"] = time.perf_counter() - t0
import torch._inductor.config as _ind                             # noqa: E402
out["resolved_max_autotune_gemm"] = _ind.max_autotune_gemm
out["arm_info"] = ARM_INFO
out["counters"] = x_arm.counters()
assert _ind.max_autotune_gemm == WANT_TPL, (
    f"arm {ARM} did not take: max_autotune_gemm={_ind.max_autotune_gemm}")
_pgf_dec = _ind.post_grad_fusion_options.get("decompose_mm_pass")
assert bool(_pgf_dec) == bool(DEC), (
    f"arm {ARM}: decompose_small_bmm did not take (DEC={DEC}, "
    f"post_grad_fusion_options={_pgf_dec}) - FastDenoiser clobbered it")
r = x_arm.report(ARM)
if r is not None:
    out["filter_per_shape"] = r.pop("per_shape")
    out["filter"] = r
    # anti-trap: an arm whose filter saw ZERO shapes did not run the filter,
    # and would silently report the unfiltered arm's time.
    assert r["shapes"] > 0, f"arm {ARM}: the candidate filter never ran"

out["full_ms"] = [bench(lambda: fast(data, prev)) for _ in range(REPS)]
out["lean_ms"] = [bench(lambda: fast(data, prev, need_aa=False)) for _ in range(REPS)]

graph, _ = fast._graphs[(False, False)]
for _ in range(5):
    graph.replay()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
ev = [e for e in p.key_averages()
      if e.device_type.name == "CUDA" and e.device_time_total > 0]


def klass(low):
    if low.startswith("triton_tem"):
        return "triton_tem"
    if low.startswith("triton"):
        return "triton"
    if "memcpy" in low or "memset" in low:
        return "memcpy"
    if ("gemm" in low or "cutlass" in low or "cublas" in low or "gemv" in low
            or "splitk" in low or "dot_kernel" in low or "reduce" in low
            and "cublas" in low):
        return "gemm"
    return "other"


census = {k: [0, 0.0] for k in ("triton_tem", "triton", "gemm", "memcpy", "other")}
gemms = []
for e in ev:
    k = klass(e.key.lower())
    census[k][0] += e.count / 10
    census[k][1] += e.device_time_total / 10
    if k == "gemm":
        gemms.append((e.key[:90], round(e.count / 10, 1),
                      round(e.device_time_total / 10, 1)))
out["census"] = census
out["kernels"] = sum(v[0] for v in census.values())
out["gpu_busy_us"] = sum(v[1] for v in census.values())
out["gemm_kernels"] = sorted(gemms, key=lambda r: -r[2])
out["other_kernels"] = sorted(
    [(e.key[:90], round(e.count / 10, 1), round(e.device_time_total / 10, 1))
     for e in ev if klass(e.key.lower()) == "other"], key=lambda r: -r[2])[:15]
a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
a.record()
for _ in range(200):
    graph.replay()
b.record()
b.synchronize()
out["replay_us"] = a.elapsed_time(b) * 1000 / 200
r, _ = fast(data, prev, need_aa=False)
out["pos_sum"] = float(r["pos"].double().sum())
out["pos_absmax"] = float(r["pos"].abs().max())
print("RESULT " + json.dumps(out), flush=True)

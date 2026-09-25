"""Shared setup for the N007 block-restructure measurements. Throwaway."""
import os, sys, time, torch
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser

CKPT = "/hnvme/workspace/v181bb10-salad/{}-200k.npz"


def build(config="default_vp", device="cuda"):
    cfg = get_config(config)
    m = StructureDiffusionPredict(cfg).eval().to(device)
    load_model(m, load_param_npz(CKPT.format(config)), cfg, strict=True)
    return m, cfg


def make_data(cfg, L, device="cuda"):
    A = 5 + cfg.augment_size
    z = torch.zeros(L, dtype=torch.long, device=device)
    pos = torch.randn(L, A, 3, device=device) * 4
    d = dict(pos=pos, pos_noised=pos,
             aa_gt=torch.full((L,), 20, dtype=torch.long, device=device),
             seq=torch.full((L,), 20, dtype=torch.long, device=device),
             seq_noised=torch.full((L,), 20, dtype=torch.long, device=device),
             residue_index=torch.arange(L, device=device),
             chain_index=z, batch_index=z,
             mask=torch.ones(L, dtype=torch.bool, device=device),
             cyclic_mask=torch.zeros(L, dtype=torch.bool, device=device),
             t_pos=torch.full((L,), 0.5, device=device),
             t_seq=torch.ones(L, device=device))
    p = {"pos": torch.zeros(L, A, 3, device=device),
         "local": torch.zeros(L, cfg.local_size, device=device)}
    return d, p


def bench(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_compiled(fn, iters=50, warm=10):
    """Same but marks a cudagraph step boundary each call."""
    def call():
        torch.compiler.cudagraph_mark_step_begin()
        fn()
    return bench(call, iters, warm)


def kernel_stats(fn, iters=10):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    tot_us, tot_n = 0.0, 0
    rows = []
    for e in ka:
        v = getattr(e, "self_device_time_total", None)
        if v is None:
            v = getattr(e, "self_cuda_time_total", 0.0)
        if v > 0:
            tot_us += v
            tot_n += e.count
            rows.append((v / iters, e.count / iters, e.key))
    rows.sort(reverse=True)
    return tot_us / iters, tot_n / iters, rows

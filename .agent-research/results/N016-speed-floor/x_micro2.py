"""N016 - follow-ups that decide two contested terms.

M5  Can the per-launch cost be PARALLELISED for the model's real kernels?
    M1 showed an empty 1-element kernel drops 1.166 -> 0.20 us when the graph
    is captured on 8 streams, but a 1.6 MiB elementwise kernel does not move
    (it already saturates HBM). The model's kernels are mostly in between, and
    most of its time is cuBLAS, so the question has to be asked of the real
    shapes: replay N independent copies of a model GEMM / a model-sized
    pointwise op, captured on 1 vs 2 vs 4 vs 8 streams.

M6  How cheap can a global barrier get?  M2 measured 1.91 us at 132 blocks
    with a nanosleep spin. Sweep block count and spin strategy so the number
    quoted in the bound is the best case, not an artefact.

M7  Kernel-duration histogram of the real compiled forward: how much of the
    measured GPU-busy time is ALREADY at the 1.166 us per-kernel floor.
"""
from __future__ import annotations

import argparse
import collections
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="m5,m6")
ap.add_argument("--reps", type=int, default=100)
A = ap.parse_args()
WANT = set(A.only.split(","))
dev = torch.device("cuda")


def ev_time(fn, reps, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / reps


def cap(build, streams):
    build(None)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        build(None)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        build(streams)
    return g


# ------------------------------------------------------------------ M5
M5_GEMM = [
    (6400, 400, 64, "pair.dist   (6400,400)x(400,64)"),
    (4800, 240, 64, "pair.pairvec(4800,240)x(240,64)"),
    (4800, 9, 64, "pair.rot    (4800,  9)x(  9,64)"),
    (100, 128, 512, "update.lin  ( 100,128)x(128,512)"),
    (100, 1024, 128, "proj_out    ( 100,1024)x(1024,128)"),
]


def m5():
    N = 48
    for prec in ("highest", "high"):
        torch.set_float32_matmul_precision(prec)
        for (M, K, Nn, label) in M5_GEMM:
            aa = [torch.randn(M, K, device=dev) for _ in range(N)]
            ww = [torch.randn(K, Nn, device=dev) for _ in range(N)]
            cc = [torch.empty(M, Nn, device=dev) for _ in range(N)]
            row = []
            for ns in (1, 2, 4, 8):
                strms = [torch.cuda.Stream() for _ in range(ns)] if ns > 1 else None

                def build(streams, ns=ns):
                    if streams is None:
                        for i in range(N):
                            torch.mm(aa[i], ww[i], out=cc[i])
                        return
                    cur = torch.cuda.current_stream()
                    for s in streams:
                        s.wait_stream(cur)
                    for i in range(N):
                        with torch.cuda.stream(streams[i % ns]):
                            torch.mm(aa[i], ww[i], out=cc[i])
                    for s in streams:
                        cur.wait_stream(s)

                try:
                    g = cap(build, strms)
                    row.append(ev_time(g.replay, A.reps) / N)
                    del g
                except Exception as exc:
                    row.append(float("nan"))
                    print(f"M5 capture failed ns={ns}: {exc}", flush=True)
            print(f"M5 {prec:<8} {label:<32} per-GEMM us by streams "
                  f"1={row[0]:6.3f} 2={row[1]:6.3f} 4={row[2]:6.3f} 8={row[3]:6.3f}  "
                  f"speedup_8={row[0]/row[3]:4.2f}x", flush=True)
            del aa, ww, cc
            torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("highest")
    # small pointwise at the model's per-residue tensor size (100x128 = 50 KiB)
    for numel, tag in ((100 * 128, "local (100,128) 50KiB"),
                       (100 * 64 * 64, "pair (100,64,64) 1.6MiB"),
                       (100 * 64 * 400, "dist feat (100,64,400) 9.8MiB")):
        N2 = 64 if numel < 10 ** 6 else 24
        ts = [torch.zeros(numel, device=dev) for _ in range(N2)]
        row = []
        for ns in (1, 2, 4, 8):
            strms = [torch.cuda.Stream() for _ in range(ns)] if ns > 1 else None

            def build(streams, ns=ns):
                if streams is None:
                    for t in ts:
                        t.add_(1.0)
                    return
                cur = torch.cuda.current_stream()
                for s in streams:
                    s.wait_stream(cur)
                for i, t in enumerate(ts):
                    with torch.cuda.stream(streams[i % ns]):
                        t.add_(1.0)
                for s in streams:
                    cur.wait_stream(s)

            g = cap(build, strms)
            row.append(ev_time(g.replay, A.reps) / N2)
            del g
        print(f"M5 pointwise {tag:<26} per-kernel us by streams "
              f"1={row[0]:6.3f} 2={row[1]:6.3f} 4={row[2]:6.3f} 8={row[3]:6.3f}  "
              f"speedup_8={row[0]/row[3]:4.2f}x", flush=True)
        del ts
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ M6
M6_CPP = """
void run_barrier(torch::Tensor scratch, torch::Tensor count, torch::Tensor sense,
                 int64_t iters, int64_t nblocks, int64_t nthreads, int64_t spin);
"""
M6_CU = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int SPIN>
__global__ void bk(float *scratch, unsigned int *count, unsigned int *sense,
                   int iters, unsigned int nblocks) {
    __shared__ unsigned int s_sense;
    unsigned int local_sense = 0u;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters; ++i) {
        scratch[tid] += 1.0f;
        __syncthreads();
        if (threadIdx.x == 0) {
            local_sense = 1u - local_sense;
            __threadfence();
            unsigned int old = atomicAdd(count, 1u);
            if (old == nblocks - 1u) {
                atomicExch(count, 0u);
                __threadfence();
                atomicExch(sense, local_sense);
            } else {
                if (SPIN) {
                    while (atomicAdd(sense, 0u) != local_sense) { __nanosleep(32); }
                } else {
                    while (atomicAdd(sense, 0u) != local_sense) { }
                }
            }
            s_sense = local_sense;
        }
        __syncthreads();
        local_sense = s_sense;
    }
}

void run_barrier(torch::Tensor scratch, torch::Tensor count, torch::Tensor sense,
                 int64_t iters, int64_t nblocks, int64_t nthreads, int64_t spin) {
    if (spin)
        bk<1><<<(int)nblocks,(int)nthreads>>>(scratch.data_ptr<float>(),
            reinterpret_cast<unsigned int*>(count.data_ptr<int>()),
            reinterpret_cast<unsigned int*>(sense.data_ptr<int>()),
            (int)iters,(unsigned int)nblocks);
    else
        bk<0><<<(int)nblocks,(int)nthreads>>>(scratch.data_ptr<float>(),
            reinterpret_cast<unsigned int*>(count.data_ptr<int>()),
            reinterpret_cast<unsigned int*>(sense.data_ptr<int>()),
            (int)iters,(unsigned int)nblocks);
}
"""


def m6():
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="n016_barrier3", cpp_sources=M6_CPP, cuda_sources=M6_CU,
                      functions=["run_barrier"], extra_cuda_cflags=["-O3"],
                      verbose=False)
    for spin in (0, 1):
        for nblocks in (1, 8, 32, 66, 132, 264):
            nthreads = 128
            scratch = torch.zeros(nblocks * nthreads, device=dev)
            count = torch.zeros(1, dtype=torch.int32, device=dev)
            sense = torch.zeros(1, dtype=torch.int32, device=dev)
            t = {}
            for iters in (64, 1024):
                t[iters] = ev_time(lambda it=iters: mod.run_barrier(
                    scratch, count, sense, it, nblocks, nthreads, spin), 30, warm=3)
            marg = (t[1024] - t[64]) / (1024 - 64)
            print(f"M6 spin={'nanosleep' if spin else 'busy':<9} blocks={nblocks:<4} "
                  f"per_barrier={marg:7.4f} us", flush=True)


for name, fn in (("m5", m5), ("m6", m6)):
    if name in WANT:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"{name.upper()} FAILED: {exc}", flush=True)
            traceback.print_exc()
        print(f"# {name} done in {time.time()-t0:.1f}s", flush=True)

"""N016 - hardware microbenchmarks that pin the contestable terms of the floor.

v2. Everything that times a *kernel* now does so inside a captured CUDA graph.
v1 timed eager loops and measured the ~6-8 us CPU dispatch cost instead of the
GPU: every small GEMM came out at "8.0 us" regardless of its shape, which is
the dispatch floor, not a hardware floor. That is the same class of mistake
N012 refuted, so it is fixed here rather than reported.

  DEV   hardware peaks on this node
  M1    per-kernel REPLAY cost inside a captured graph, vs kernel count,
        dependency chain (one stream) vs forked streams
  M2    cheapest possible GLOBAL BARRIER: a hand-written sense-reversing grid
        barrier inside ONE persistent kernel
  M3    achieved bandwidth at the model's tensor sizes, graph-replayed,
        L2-hot vs HBM-cold
  M4    GEMM at the model's real shapes, graph-replayed, fp32 vs TF32,
        one-at-a-time vs batched-by-6
"""
from __future__ import annotations

import argparse
import subprocess
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--reps", type=int, default=100)
A = ap.parse_args()
WANT = set(A.only.split(",")) if A.only else {"dev", "m1", "m2", "m3", "m4"}

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
    return s.elapsed_time(e) * 1e3 / reps          # microseconds per call


def capture(build, nstreams: int = 1):
    """Capture `build()` (a callable that issues the work) into a CUDA graph.

    nstreams>1 forks the capture onto side streams so INDEPENDENT work becomes
    parallel graph nodes; with nstreams=1 stream capture records a linear
    dependency chain whatever the data dependencies are.
    """
    # warm up outside capture (allocator, cuBLAS handles, autotune)
    build(None)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        build(None)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    if nstreams == 1:
        with torch.cuda.graph(g):
            build(None)
    else:
        streams = [torch.cuda.Stream() for _ in range(nstreams)]
        with torch.cuda.graph(g):
            build(streams)
    return g


# ----------------------------------------------------------------- DEV
if "dev" in WANT:
    p = torch.cuda.get_device_properties(0)
    print(f"DEV name={p.name!r} sm={p.major}.{p.minor} SMs={p.multi_processor_count} "
          f"L2={getattr(p,'L2_cache_size',0)/2**20:.1f}MiB "
          f"mem={p.total_memory/2**30:.1f}GiB torch={torch.__version__} "
          f"cuda={torch.version.cuda}", flush=True)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm,"
                            "clocks.max.mem,power.limit", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        print("DEV nvidia-smi:", q.stdout.strip(), flush=True)
        sm_mhz = float(q.stdout.split(",")[2].strip().split()[0])
        peak = p.multi_processor_count * 128 * 2 * sm_mhz * 1e6 / 1e12
        print(f"DEV fp32_FMA_peak = {p.multi_processor_count} SM x 128 cores x 2 x "
              f"{sm_mhz/1000:.3f} GHz = {peak:.2f} TFLOP/s "
              f"(TF32 tensor-core peak is ~{peak*7.4:.0f} TFLOP/s dense per NVIDIA's "
              f"ratio; measured in M4)", flush=True)
    except Exception as exc:
        print("DEV nvidia-smi failed:", exc, flush=True)


# ------------------------------------------------------------------ M1
def m1():
    for numel, tag in ((1, "tiny_1elem"), (100 * 64 * 64, "pairsize_1.6MiB")):
        for nstreams in (1, 8):
            prev = None
            for n in (1, 32, 128, 256, 512, 921, 1024, 2048):
                if numel > 1 and n > 1024:
                    continue
                if nstreams > 1:
                    ts = [torch.zeros(numel, device=dev) for _ in range(n)]
                else:
                    ts = [torch.zeros(numel, device=dev)] * n    # same tensor: chain

                def build(streams, ts=ts, n=n, nstreams=nstreams):
                    if streams is None:
                        for t in ts:
                            t.add_(1.0)
                        return
                    cur = torch.cuda.current_stream()
                    for s in streams:
                        s.wait_stream(cur)
                    for i, t in enumerate(ts):
                        with torch.cuda.stream(streams[i % nstreams]):
                            t.add_(1.0)
                    for s in streams:
                        cur.wait_stream(s)

                try:
                    g = capture(build, nstreams)
                except Exception as exc:
                    print(f"M1 {tag} ns={nstreams} n={n} CAPTURE FAILED: {exc}", flush=True)
                    continue
                us = ev_time(g.replay, A.reps)
                marg = ((us - prev[1]) / (n - prev[0])) if prev else float("nan")
                print(f"M1 {tag:<16} streams={nstreams} n={n:<5} replay={us:9.2f} us  "
                      f"per_kernel={us/n:6.3f} us  marginal={marg:6.3f} us", flush=True)
                prev = (n, us)
                del ts, g
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


# ------------------------------------------------------------------ M2
M2_CPP = """
void run_barrier(torch::Tensor scratch, torch::Tensor count, torch::Tensor sense,
                 int64_t iters, int64_t nblocks, int64_t nthreads);
"""
M2_CU = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void barrier_kernel(float *scratch, unsigned int *count,
                               unsigned int *sense, int iters,
                               unsigned int nblocks) {
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
                while (atomicAdd(sense, 0u) != local_sense) { __nanosleep(32); }
            }
            s_sense = local_sense;
        }
        __syncthreads();
        local_sense = s_sense;
    }
}

void run_barrier(torch::Tensor scratch, torch::Tensor count, torch::Tensor sense,
                 int64_t iters, int64_t nblocks, int64_t nthreads) {
    barrier_kernel<<<(int)nblocks, (int)nthreads>>>(
        scratch.data_ptr<float>(),
        reinterpret_cast<unsigned int*>(count.data_ptr<int>()),
        reinterpret_cast<unsigned int*>(sense.data_ptr<int>()),
        (int)iters, (unsigned int)nblocks);
}
"""


def m2():
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="n016_barrier2", cpp_sources=M2_CPP,
                      cuda_sources=M2_CU, functions=["run_barrier"],
                      extra_cuda_cflags=["-O3"], verbose=False)
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    for nblocks in (nsm, 2 * nsm):
        for nthreads in (128, 256):
            scratch = torch.zeros(nblocks * nthreads, device=dev)
            count = torch.zeros(1, dtype=torch.int32, device=dev)
            sense = torch.zeros(1, dtype=torch.int32, device=dev)
            base = None
            for iters in (1, 64, 256, 1024):
                fn = (lambda it=iters: mod.run_barrier(scratch, count, sense,
                                                       it, nblocks, nthreads))
                us = ev_time(fn, 30, warm=3)
                if base is None:
                    base, marg = (iters, us), float("nan")
                else:
                    marg = (us - base[1]) / (iters - base[0])
                print(f"M2 blocks={nblocks:<4} threads={nthreads:<4} iters={iters:<5} "
                      f"total={us:9.2f} us  per_barrier_marginal={marg:7.4f} us",
                      flush=True)


# ------------------------------------------------------------------ M3
def m3():
    NREP = 32
    for mib in (0.25, 1, 1.6, 4, 10, 16, 32, 46, 64, 128, 256):
        n = int(mib * 2**20) // 4
        a = torch.randn(n, device=dev)
        b = torch.randn(n, device=dev)
        c = torch.empty(n, device=dev)

        def build(_s, a=a, b=b, c=c):
            for _ in range(NREP):
                torch.add(a, b, out=c)

        g = capture(build)
        us = ev_time(g.replay, A.reps) / NREP
        # `add(out=)` reads 2 arrays and writes 1; on a write-allocate cache the
        # write also reads, so report both accountings
        gbs3 = 3 * n * 4 / (us * 1e-6) / 1e9
        gbs4 = 4 * n * 4 / (us * 1e-6) / 1e9
        print(f"M3 add     size={mib:7.2f} MiB (ws {3*mib:7.2f} MiB)  {us:9.3f} us  "
              f"{gbs3:8.1f} GB/s (3N)  {gbs4:8.1f} GB/s (4N)", flush=True)
        del a, b, c, g
        torch.cuda.empty_cache()
    # pure read: a sum reduction, 1N traffic, the cleanest bandwidth probe
    for mib in (10, 46, 256, 1024):
        n = int(mib * 2**20) // 4
        a = torch.randn(n, device=dev)
        o = torch.empty((), device=dev)

        def build(_s, a=a, o=o):
            for _ in range(8):
                torch.sum(a, out=o)

        g = capture(build)
        us = ev_time(g.replay, A.reps) / 8
        print(f"M3 sum(1N) size={mib:7.2f} MiB  {us:9.3f} us  "
              f"{n*4/(us*1e-6)/1e9:8.1f} GB/s", flush=True)
        del a, o, g
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ M4
M4_SHAPES = [
    (6400, 400, 64, 6, "pair1.dist      (6400,400)x(400,64)"),
    (4800, 400, 64, 6, "pair2.dist      (4800,400)x(400,64)"),
    (6400, 240, 64, 6, "pair1.pairvec   (6400,240)x(240,64)"),
    (4800, 240, 64, 6, "pair2.pairvec   (4800,240)x(240,64)"),
    (6400, 60, 64, 6, "pair1.dir       (6400, 60)x( 60,64)"),
    (4800, 60, 64, 6, "pair2.dir       (4800, 60)x( 60,64)"),
    (6400, 9, 64, 6, "pair1.rot       (6400,  9)x(  9,64)"),
    (4800, 9, 64, 6, "pair2.rot       (4800,  9)x(  9,64)"),
    (6400, 64, 128, 6, "pair1.mlp1      (6400, 64)x( 64,128)"),
    (6400, 128, 64, 6, "pair1.mlp2      (6400,128)x(128, 64)"),
    (4800, 64, 128, 6, "pair2.mlp1      (4800, 64)x( 64,128)"),
    (4800, 128, 64, 6, "pair2.mlp2      (4800,128)x(128, 64)"),
    (6400, 64, 8, 6, "attn1.bias      (6400, 64)x( 64,  8)"),
    (4800, 64, 8, 6, "attn2.bias      (4800, 64)x( 64,  8)"),
    (100, 128, 768, 12, "attn.qkv        ( 100,128)x(128,768)"),
    (100, 128, 576, 12, "attn.qkv_global ( 100,128)x(128,576)"),
    (100, 1024, 128, 12, "attn.project_out( 100,1024)x(1024,128)"),
    (100, 128, 512, 24, "update.linear_* ( 100,128)x(128,512)"),
    (100, 512, 128, 6, "update.linear_out(100,512)x(512,128)"),
    (100, 60, 256, 6, "update.mlp1     ( 100, 60)x( 60,256)"),
    (100, 256, 128, 6, "update.mlp2     ( 100,256)x(256,128)"),
    (100, 128, 60, 6, "update_positions( 100,128)x(128, 60)"),
]


def m4():
    NREP = 20
    for prec in ("highest", "high"):
        torch.set_float32_matmul_precision(prec)
        tot1 = tot6 = totflop = 0.0
        for (M, K, N, cnt, label) in M4_SHAPES:
            a = torch.randn(M, K, device=dev)
            w = torch.randn(K, N, device=dev)
            c = torch.empty(M, N, device=dev)

            def b1(_s, a=a, w=w, c=c):
                for _ in range(NREP):
                    torch.mm(a, w, out=c)

            us1 = ev_time(capture(b1).replay, A.reps) / NREP
            ab = torch.randn(6, M, K, device=dev)
            wb = torch.randn(6, K, N, device=dev)
            cb = torch.empty(6, M, N, device=dev)

            def b6(_s, ab=ab, wb=wb, cb=cb):
                for _ in range(NREP):
                    torch.bmm(ab, wb, out=cb)

            us6 = ev_time(capture(b6).replay, A.reps) / NREP
            fl = 2.0 * M * K * N
            tot1 += cnt * us1
            tot6 += (cnt / 6.0) * us6
            totflop += cnt * fl
            print(f"M4 {prec:<8} {label:<38} single {us1:7.3f} us "
                  f"({fl/us1/1e6:7.2f} TF/s)   bmm6 {us6:7.3f} us "
                  f"({6*fl/us6/1e6:7.2f} TF/s = {6*us1/us6:4.2f}x)  x{cnt}", flush=True)
            del a, w, c, ab, wb, cb
            torch.cuda.empty_cache()
        print(f"M4 {prec:<8} GEMM-STAGE TOTAL: as-launched {tot1/1e3:7.4f} ms   "
              f"batched-by-6 {tot6/1e3:7.4f} ms   arithmetic {totflop/1e9:7.4f} GFLOP "
              f"({totflop/tot1/1e6:5.2f} TF/s achieved as-launched)", flush=True)
    torch.set_float32_matmul_precision("highest")


for name, fn in (("m1", m1), ("m2", m2), ("m3", m3), ("m4", m4)):
    if name in WANT:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"{name.upper()} FAILED: {exc}", flush=True)
            traceback.print_exc()
        print(f"# {name} done in {time.time()-t0:.1f}s", flush=True)

# N012 · Red-team audit: is the 3.4-3.5 ms "floor" real?

type: research · date: 2026-09-26 · status: backfilled (done)

Backfilled by work-lint 2026-09-26: `results/N012-codex-floor-audit/` existed
with a full `FINDINGS.md` but no INDEX row or item file.

## Goal
Independently check two claims from the N009/N014 speed work on commit
`e2a3464` (`n006-manual-cudagraph`): (1) the compiled fast path is one graph
with zero breaks/recompiles, and (2) 3.4-3.5 ms is a hard software floor at
L=100.

## Plan
Since the live worktree's `.git` was read-only (no `git fetch`), clone the
pinned commit via `git ls-remote` + a fresh checkout into `latest/` and audit
that isolated copy so concurrent live-worktree edits can't contaminate the
verdict. Re-run cold `torch._dynamo.explain` across all flag/config
combinations, profile a captured-vs-noncaptured replay, and independently
re-measure the GEMM-batching claim's arithmetic basis.

## Evidence
H100, PyTorch 2.5.1+cu121, `triad` env, jobs 903180-903204 (isolated checkout
903198-903204; live-worktree 903180-903195 kept only as diagnostic), each
<=1 GPU-hour.
- **Claim 1, upheld with a caveat**: 40 cold `torch._dynamo.explain` traces
  (`exact`/`fast` x `need_aa` x `pair_cond` x 4 configs + a cyclic variant)
  all show graphs=1, breaks=0 - but only for the compiled core
  (`forward_core`); the Python wrapper (conditioning cache, capture/replay)
  is outside it. A cold full-AA variant recompiles ONCE (5 unique graphs, not
  the claimed 4) because `sidechain.py`'s `_DEVICE_CACHE` lazily populates on
  first full-AA use - a real one-time guard change the "no recompiles"
  framing missed, though not a per-call regression (it happens before
  capture, inside `FastDenoiser._build`'s warmup calls).
- **Claim 2, NOT established**: capture-vs-noncaptured profiling confirms
  the captured graph really does include all 921 GEMM/Triton kernels (not an
  opaque single op). But the cited 6.5 us/kernel figure is an *average GPU
  busy* number, not a launch-overhead floor; a standalone GEMM-batching test
  on the shipping condition-branch shapes measured 1.87-2.13x speedup from
  batching (K=400: 1.87x, K=9: 2.13x), directly refuting "321 launches are
  irreducible." The previously-cited "76% of fp32 peak" and "0.21 ms
  arithmetic floor" figures were computed for a K=64 pair shape that the
  current unconditioned branch doesn't use (it's K=48, ~37.8 TFLOP/s not
  51); those figures don't transfer as-is. 524 Triton kernels are a census,
  not a demonstrated minimum - `aggressive_fusion=1` already measured a
  further 1.7% gain.
- **Numerical gate widened, not loosened**: CUDA self-noise samples range
  2.21e-4 A to 1.755e-2 A across different reports (not a fixed budget); a
  candidate below the smallest observed self-difference is not automatically
  safe, and bit-identity on CUDA isn't a defensible bar while `index_mean`
  uses atomic `scatter_add`. TF32 remains categorically different (~6e-3 A,
  ruled out per N005/N007).
- Explicitly **not checked**: no batched prologue or Triton fusion was
  integrated end to end here, so no new whole-forward latency or model-wide
  parity number exists yet for the batching proposal - it remains a
  candidate, gated on a CPU/JAX parity re-check.

## Outcome
Both headline claims from the prior speed work needed correction: "zero
recompiles" is true only for the traced core, not the one cold full-AA
warmup; and "3.4-3.5 ms floor" is a measured number, not a proven lower
bound - the GEMM-batching idea (N007 P7, N014) is independently supported by
a from-scratch measurement, not refuted. No production code changed by this
audit.

## Outputs
`results/N012-codex-floor-audit/`: `FINDINGS.md` (full writeup),
`graph_audit.py`/`graph.sbatch`/`graph_903180-903184.log` (cold trace
census), `capture_compare.py`/`capture.sbatch`/`capture_903195.log`
(captured-vs-noncaptured kernel profile), `recompile_audit.py`/
`recompile.sbatch`/`recompile_903192.log`, `perf_audit.py`/`perf.sbatch`/
`perf_903185.log` (GEMM-batching measurement), `latest/` (isolated clone of
commit `e2a3464` the audit ran against, incl. its own `.agent-research/`
snapshot - not this repo's live notebook), plus `latest_*` reruns
(903198-903224) after the isolated checkout was created.

## Next
Feeds directly into N014 (batched condition-branch GEMM, P7) - that item's
own evidence should be checked against the K=48-vs-K=64 correction here
before quoting the older 76%-of-peak/0.21ms-floor figures.

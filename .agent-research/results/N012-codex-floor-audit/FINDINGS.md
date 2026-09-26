# N012 red-team audit — commit e2a3464

**Claim 1 verdict: upheld on graph breaks, but a cold full-AA variant recompiles once because a sidechain-table cache changes state.**

**Claim 2 verdict: the 3.4–3.5 ms hard floor is not established.** The current implementation's measured speed is real, but the launch count is reducible and the cited roofline numbers are not a lower-bound proof. No changed implementation was tested end to end here, so a new whole-forward time below 3.4 ms is a candidate, not a demonstrated result.

## Scope and provenance

The live repository's `.git` is mounted read only, so `git fetch origin n006-manual-cudagraph` failed with `cannot open '.git/FETCH_HEAD': Read-only file system`. The default SSH config also failed. `GIT_SSH_COMMAND='ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new' git ls-remote origin refs/heads/n006-manual-cudagraph` returned `e2a3464767640b31c10dd5743c68acda8fa4f299`; I cloned this commit into `latest/` inside this results folder. The uncommitted live worktree changed concurrently, so the verdict uses the isolated checkout. No production code was edited.

H100, PyTorch 2.5.1+cu121, `triad` environment. Jobs 903198–903204 on the isolated checkout; each had one H100 and a walltime below one GPU-hour. The earlier jobs 903180–903195 ran against the live worktree and are retained only as diagnostic evidence.

## Claim 1: cold Dynamo trace and actual capture

`graph_audit.py` calls `torch._dynamo.reset()` and removes/recreates a unique `TORCHINDUCTOR_CACHE_DIR` before **each** `torch._dynamo.explain` call. It covers `exact` and `fast`, `need_aa=False/True`, `pair_cond=False/True`, all four checkpoint configs, and a cyclic `default_vp` variant (40 cold traces). Jobs 903198–903202 each print eight rows like:

```
EXPLAIN exact False False graphs 1 breaks 0 reasons []
EXPLAIN exact False True  graphs 1 breaks 0 reasons []
EXPLAIN exact True  False graphs 1 breaks 0 reasons []
EXPLAIN exact True  True  graphs 1 breaks 0 reasons []
EXPLAIN fast  False False graphs 1 breaks 0 reasons []
EXPLAIN fast  False True  graphs 1 breaks 0 reasons []
EXPLAIN fast  True  False graphs 1 breaks 0 reasons []
EXPLAIN fast  True  True  graphs 1 breaks 0 reasons []
```

The exact output for every variant is in `latest_graph_903198.log` through `latest_graph_903202.log`, lines 25–33. `FastDenoiser._compiled` targets `model.diffusion.forward_core` (`latest/salad_torch/modules/fast.py:175-190`); plain Python flags deliberately specialize four graphs (`latest/salad_torch/modules/fast.py:243-252`). The full Python wrapper, including conditioning cache checks and input/output copies, is outside this compiled region (`latest/salad_torch/modules/fast.py:300-383`). Thus the true statement is **zero breaks within each compiled core variant**, not one Dynamo graph for the entire `FastDenoiser.forward` wrapper.

`capture_compare.py` compiled and manually captured `exact`, lean, `pair_cond=False`, then profiled both the replay and the same compiled callable under `torch.no_grad()` on the current commit (job 903203, `latest_capture_903203.log:27-31`):

```
PROFILE captured kernels 921 gpu_busy_ms 3.4307256
COUNTS captured {'triton': 524, 'cublas': 321, 'memcpy': 36}
PROFILE noncaptured kernels 919 gpu_busy_ms 3.4840248
COUNTS noncaptured {'triton': 524, 'cublas': 321, 'memcpy': 36}
REPLAY_EVENT_US 3378.9481
```

This supports capture of the **entire compiled core**, including its GEMMs and Triton work. The two-kernel total discrepancy is in other kernel categories; it is not evidence of a missing projection or feature stage. The historical 946-kernel count (`notes/04_PARITY_AND_PERF.md:411-425`) profiles the `FastDenoiser` call and its wrapper operations; the 921/919 comparison profiles only the core. The source comment claiming a captured replay profiles as one opaque operation is incorrect in this environment (`.agent-research/results/N009-speed-proposals/x_kernels.py:1-5`): the CUDA profiler exposes 921 individual replay kernels here.

A first comparison accidentally omitted `no_grad()` on the noncaptured side and triggered a new compile inside the profiler (`perf_903185.log:36-51`); those noncaptured numbers are invalid and excluded. `capture_compare.py` fixes the context on both sides.

`recompile_audit.py` compiled all four specialization combinations in one cold process on the current commit (job 903204, `latest_recompile_903204.log:27-49`). Ten repeated calls per variant produced these Dynamo `unique_graphs` counts before → after: `need_aa=False,pair_cond=False` **1→1**; `False,True` **2→2**; `True,False` **3→4**; `True,True` **5→5**. The one unexpected recompile is explicit in `latest_recompile_903204.log:36-40`: guard failure `not G['__import_salad_torch_dot_geometry_dot_sidechain']._DEVICE_CACHE`. `_tables()` lazily populates that global device-table dictionary on first full-AA use (`latest/salad_torch/geometry/sidechain.py:344-356`). The final process had **5 successful frames / 5 unique graphs**, not the 4 expected from flags alone. There was no `cache_size_limit` warning and no continuing per-call recompilation. `FastDenoiser._build` runs the compiled callable three times before capture (`latest/salad_torch/modules/fast.py:254-269`), so this one-time guard change occurs before the graph is captured in the production path. It is **not a graph break**, but it makes an unqualified “one graph total / no recompiles” statement false.

## Claim 2: measured work, and what is still reducible

The 6.5 µs figure is **average GPU kernel busy time**, not a generic CPU launch overhead. N009 divides 2.096 ms by 321 profiled cuBLAS kernels (`notes/04_PARITY_AND_PERF.md:421-425`). A standalone H100 test on the *shipping condition-branch* GEMM shapes, `M=100×48=4800`, `N=64`, fp32 highest, found the following (`perf_audit.py`, `perf_903185.log:26-33`). Times are CUDA-event spans per operation over repeated submissions, including submission gaps, so they answer a different question from the profiler's kernel-only average:

| K | one `mm`, ordinary submission | one `mm` replayed | six `mm` replayed | one batch-6 `bmm` replayed | `bmm` vs six max abs difference |
|---:|---:|---:|---:|---:|---:|
| 400 | 14.84 µs | 17.92 µs | 91.80 µs | 49.13 µs | 8.39e-05 (random tensors) |
| 240 | 11.17 µs | 15.49 µs | 73.67 µs | 30.54 µs | 0 |
| 60 | 7.04 µs | 9.88 µs | 32.80 µs | 15.46 µs | 0 |
| 9 | 7.10 µs | 8.25 µs | 23.24 µs | 10.89 µs | 0 |

The narrow `K=9` case is **2.13× faster** batched, and the wide `K=400` case **1.87× faster** batched in this isolated test. This refutes “321 launches are irreducible,” though not all 321 are independent GEMMs: N009's top-kernel list includes split-K reduction and GEMV kernels (`notes/04_PARITY_AND_PERF.md:428-437`). A previous six-block pair-builder experiment on real weights found **1.017 → 0.792 ms** (0.225 ms saved) and max pair-output difference **7.629e-06** at L=100, K=64 (`.agent-research/results/N007-block-restructure/logs/n007r3_902286.log:25-29`). That experiment is isolated and uses K=64, whereas today's unconditioned branch uses K=48; it does **not** prove a 0.225 ms whole-forward gain.

The batching opportunity is structural: six standard blocks' second pair branch takes the loop-invariant `prev_pos` (`latest/salad_torch/modules/blocks.py:352-356`, `:389-400`). The motif adapter's second branch takes changing `pos`, so it needs a separate sequential path (`latest/salad_torch/modules/blocks.py:478-482`). The prior plan describes a six-way prologue and its checkpoint-layout consequences (`.agent-research/results/N007-block-restructure/PLAN.md:369-425`). It also estimates only ~30 GEMM operations would collapse; batching feature construction, layer norm and MLP at the same time is what made the measured isolated gain larger.

The 524 Triton kernels are a **census, not a demonstrated minimum**. `aggressive_fusion=1` already measured 3.564 → 3.503 ms on the lean path (1.7%; `notes/04_PARITY_AND_PERF.md:375-406`, `.agent-research/results/N009-speed-proposals/logs/abl3_903126.log:36`). The prior plan estimates a custom on-the-fly pair-feature/projection kernel could save 1.2–1.6 ms across twelve pair blocks, but this is only a proposal (`.agent-research/results/N007-block-restructure/PLAN.md:429-454`); feature formulas and reduction order need a parity gate. Current source launches separate `linear_dist`, `linear_dir`, `linear_rot`, and `linear_pairvec` projections after constructing each feature tensor (`latest/salad_torch/modules/blocks.py:208-217`), leaving a concrete fusion target. Inductor's ordinary elementwise fusion does not eliminate those GEMM boundaries.

The often cited **76% of fp32 peak** was computed for the former K=64 pair shape `(6400,400)×(400,64)` (`.agent-research/results/N007-block-restructure/PLAN.md:61-68`). The shipping unconditioned branch has K=48 (`latest/salad_torch/modules/diffusion.py:298-305`); its analogous projection is `(4800,400)×(400,64)`, 0.246 GFLOP versus 0.328 GFLOP at K=64. Applying 6.5 µs to this smaller GEMM gives ~37.8 TFLOP/s, not 51 TFLOP/s. The current-neighbour branch still has K=64, so the old wide-GEMM argument applies there, not uniformly across the lean forward. The 0.21 ms arithmetic and ~0.37 ms bandwidth figures also derive from a pre-N009 whole-model census (`.agent-research/results/N007-block-restructure/PLAN.md:22-27`), and do not prove a 3.4 ms software floor.

## Numerical gate

CUDA self-noise is real: the notes report **2.21e-04 A** maximum coordinate difference for one identical-input forward pair and zero on CPU (`notes/04_PARITY_AND_PERF.md:466-476`); a separate run reported **1.755e-02 A** (`.agent-research/results/N009-speed-proposals/logs/nd_903136.log:36`). These are samples, not a fixed error budget. An optimization below one observed CUDA self-difference is **not necessarily undetectable**: deterministic CPU comparison and repeated paired CUDA runs can reveal a systematic change. Conversely, bit identity on CUDA is not a defensible gate while `index_mean` uses atomic `scatter_add` (`latest/salad_torch/modules/blocks.py:260-261`; `notes/04_PARITY_AND_PERF.md:328-350`).

This widens the *candidate set*, not the acceptance threshold automatically. The batch-six real-weight pair-output difference 7.629e-06 is below the cited 2.21e-04 sample and the 2e-05 CPU JAX coordinate parity floor, but a full-forward CPU/JAX and distributional CUDA gate is still needed. `aggressive_fusion=1` was rejected because its numerical effect could not be resolved against CUDA self-noise (`notes/04_PARITY_AND_PERF.md:392-406`), despite a measured 1.7% speed gain. That is a reason to test it on the deterministic CPU path, not evidence that it violates parity. A custom Triton projection is also admissible as a **test candidate** if its final-coordinate deviation clears the same gate; its 842-term reassociation risk remains unmeasured. TF32 is different: the existing checkpoint comparison moved coordinates by ~6e-03 A (`notes/04_PARITY_AND_PERF.md:175-184`), far above both the CPU parity floor and the cited CUDA self-noise sample.

## Not checked

No batched prologue or Triton fusion was integrated, so neither a new end-to-end latency nor model-wide parity for those proposals is established. The standalone GEMM tensors were random; their arithmetic difference is not a final-coordinate difference. The cold graph audit used synthetic input buffers and initialized model weights, rather than checkpoint values; graph breaks depend on control flow and shapes, while the capture comparison loaded `default_vp-200k` real weights. CPU/JAX parity after changing a numerical optimization was not measured because no production code was changed.

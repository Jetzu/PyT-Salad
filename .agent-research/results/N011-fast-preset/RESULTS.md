# N011 — the `fast` inference preset

Measured 2026-09-26 on an H100 (`--partition h100`, one GPU), `default_vp-200k`,
L=100 unless stated. Scripts and raw logs live beside this file (`x_fast.py`,
`x_fast2.py`, `x_preset.py`, `x_combo.py`, `x_gen_k32.py`, `ss.py`,
`run_*.sbatch`, `logs/`).

Two kinds of measurement appear below and they must not be confused:

* **`x_fast.py` / `x_fast2.py`** — a private copy of the `FastDenoiser`
  mechanism (compile `mode="default"` + one explicit whole-forward
  `torch.cuda.CUDAGraph`) used to *explore* candidates by monkeypatch. Fast to
  iterate, but it measures a patched copy, not the product.
* **`x_preset.py`** — the production `FastDenoiser(preset=...)`, i.e. exactly
  what `scripts/generate.py --fast --preset X` runs. **Every number quoted as a
  result comes from this one.**

## 1. Verdict

The fast preset is **TF32 plus half the sampling schedule, and nothing else** —
because nothing else that was tried paid for itself. Both halves are large and
independent, and neither touches what the model is: same checkpoint, same
architecture, same K=64 neighbourhoods, same sampler.

| | reporting step | lean step | per design | vs exact |
|---|---|---|---|---|
| exact (the default, unchanged) | 4.381 / 4.368 ms | 3.568 / 3.554 ms | **1.43 s** | — |
| exact + TF32 only | 3.719 ms | 2.996 ms | 1.20 s | 1.19× |
| fast without TF32 (schedule only) | 4.385 ms | 3.563 ms | 0.72 s | 1.99× |
| **fast** | **3.614 ms** | **2.894 ms** | **0.58 s** | **2.47×** |

JAX salad is 4.73 ms / 1.56 s per design, so the fast preset is **2.7× JAX end
to end** and the exact path is now 1.09× JAX. Against the 2.1 s the brief
started from, fast is **3.6×**.

Length sweep, fast preset, one process per length:

| L | reporting | lean | per design |
|---|---|---|---|
| 50 | 3.229 ms | 2.609 ms | 0.52 s |
| 100 | 3.614 ms | 2.894 ms | 0.58 s |
| 200 | 4.968 ms | 4.086 ms | 0.82 s |
| 400 | 7.938 ms | 6.692 ms | 1.35 s |

**What was rejected is the more useful half of this document**: of eight
candidate deviations, six bought nothing measurable and one bought 10 % at the
cost of visibly worse structures. See §5.

## 2. Ranking by speed per unit of deviation

The brief asks for this ranking explicitly. "Deviation" is the single-forward
`|Δpos|` against the unpatched fp32 model on identical input — read against the
**2.2e-04 Å floor of CUDA's own run-to-run nondeterminism** on this model
(N010: `index_mean` → `scatter_add` atomic ordering, amplified by each block's
`pos` feeding the next block's neighbour argsort). Anything at or below that
floor is free; anything above it is a real perturbation.

| rank | change | speed | deviation | in the preset? |
|---|---|---|---|---|
| 1 | halve the schedule (250/200) | **−50 % of the run** | trajectory resolution, not arithmetic | **yes** |
| 2 | TF32 | −16 … −19 % of the step | 3.4e-01 Å (~1500× the noise floor) | **yes** |
| 3 | K 64→32 both branches | −10 % of the step | 6.8 Å, and measurably worse structures | no — §5.5 |
| — | shared self-conditioning draw | 0 % | 6.4 Å | no — §5.2 |
| — | bypassing the conditioning cache key | 0 % | correctness risk | no — §5.6 |
| — | `need_aux=False` | 0 % | behavioural | no |
| — | bf16 in the GEMM (`"medium"`) | **+3 %** | 2.9e-01 Å | no — §5.3 |
| — | `topk` for the stable argsort | **+5.6 %** | tie order | no — §5.1 |

TF32 buys twice K-reduction's speed for one twentieth of its perturbation, and
K-reduction is the only candidate that alters what the network *sees* rather
than how accurately it computes. That is the whole argument for the preset's
shape.

## 3. Per-forward exploration (`x_fast.py`, L=100, lean step)

| variant | ms | vs exact | dev_max (Å) |
|---|---|---|---|
| BASE — everything on, reporting step | 4.535 | +34 % | 3.0e-4 |
| EXACT_AA — exact preset, reporting step | 4.200 | +24 % | 3.0e-4 |
| **EXACT — exact preset, lean step** | **3.371–3.393** | — | 3.0e-4 |
| need_aux=False | 3.416 | +0.7 % | 2.9e-4 |
| topk instead of stable argsort | 3.582 | **+5.6 %** | 2.7e-4 |
| current branch K 64→48 | 3.261 | −3.9 % | 9.2 |
| condition branch K 64→32 | 3.294 | −2.9 % | 8.6 |
| both branches K→48 | 3.216 | −5.2 % | 9.8 |
| both branches K→32 | 3.048 | −10.2 % | 6.7 |
| current K→32 | 3.166 | −6.6 % | 11.7 |
| `matmul_precision("medium")` (bf16 GEMM) | 2.856 | −15.3 % | 2.9e-1 |
| **`matmul_precision("high")` (TF32)** | **2.760 / 2.848** | **−16 … −18 %** | 3.4e-1 |
| TF32 + current K→48 | 2.729 | −19.2 % | 9.2 |
| TF32 + both branches K→32 | 2.572 | −24 % | 6.8 |

Note the `dev_max` column against the 2.2e-04 Å nondeterminism floor: every
"exact" row is *at* the floor, i.e. indistinguishable from a rerun.

## 4. Quality gate — 8 de-novo designs at L=100, seed 42 (`ss.py`)

Secondary structure is the CA-only P-SEA assignment (Labesse et al. 1997);
"ordered" is helix + strand. Clash = CA–CA pairs closer than 4 Å at
|i−j| ≥ 3.

| arm | CA-CA mean (Å) | Rg (Å) | helix | strand | coil | ordered | clash | NaN | s/design |
|---|---|---|---|---|---|---|---|---|---|
| exact, 500/400 | 3.7601 ± 0.0062 | 12.50 ± 0.78 | 0.605 | 0.319 | 0.076 | 0.924 | 0.1 | 0 | 1.5 |
| fast numerics, 500/400 | 3.7666 ± 0.0110 | 12.35 ± 0.46 | 0.607 | 0.316 | 0.076 | 0.924 | 0.0 | 0 | 1.2 |
| **fast, 250/200** | **3.7572 ± 0.0160** | **12.92 ± 0.86** | 0.591 | 0.328 | 0.081 | **0.919** | 0.5 | 0 | **0.6** |
| fast, 125/100 | 3.7572 ± 0.0107 | 13.26 ± 1.34 | 0.570 | 0.348 | 0.082 | 0.918 | 0.0 | 0 | 0.3 |
| fast + K 64→32, 250/200 | 3.7597 ± 0.0142 | 12.90 ± 1.00 | 0.601 | 0.290 | **0.109** | 0.891 | **1.1** | 0 | 0.5 |

Gate: CA-CA in 3.70–3.85 Å ✓ (every arm, and every individual design), no NaN ✓,
Rg sane ✓, secondary structure not collapsed ✓ for every arm except the K→32
one, which is where the degradation shows: coil 0.076 → 0.109 and clashes
0.1 → 1.1 per design.

Two honest caveats on the fast arm:

* the **spread** widens slightly — CA-CA σ 0.0062 → 0.0160 Å, Rg σ 0.78 → 0.86 —
  and individual designs show the occasional short bond (fast min 3.462 Å,
  fast-numerics-500 min 2.939 Å, against exact's worst 3.582 Å). The means stay
  in band and no design fails the gate, but the fast preset is visibly a little
  looser at the tails. On 8 designs this is suggestive, not established.
* **125/100 also passes** every metric here. It is not the default anyway: the
  gate used is structural, and at a 4× cut I would want a designability check
  (ProteinMPNN + a folding model) before recommending it. 250/200 is the
  cautious choice that is already 2.5× the exact path.

## 5. Refuted / dropped — measured, not argued

### 5.1 `topk` instead of the stable argsort — SLOWER

3.582 ms against the exact path's 3.371–3.393 ms, i.e. **+5.6 %**. The idea was
that a full `argsort` of an `(N, N)` matrix followed by a head slice does
`O(N log N)` work to answer an `O(N)` question. It does, and it is still
faster: `topk` returns values as well as indices (so the separate
`distance[knn]` gather it saves is paid back inside the kernel), and inductor
fuses the argsort path better. Neighbour selection is 3.5 % of the forward
(N007 §2), so there was never more than ~0.12 ms on the table.

It was re-tested for a second, better reason in §6.

### 5.2 One shared self-conditioning Gumbel draw — no gain

The reasoning was sound and testable: after the P6 hoist only the *random*
stage of the condition branch differs between blocks, so pinning that draw
makes `cond_neighbours` one tensor for the whole trunk, and then `pair_2`'s
seven geometric feature tensors are the same expression on the same inputs in
all six blocks — P7's "the condition branch is a prologue" for free, *if*
inductor CSEs them.

**It does not.** Production path, switch verified live (6.4464 Å change in
`pos`, against a 0.0 Å same-flag rerun as the noise floor):

```
preset=fast                      lean 2.894 ms
preset=fast --override no_shared_draw   lean 2.882 ms
```

Nothing — if anything marginally worse. The six feature constructions are still
emitted six times. **Removed from the preset and the code**: a deviation with
no payoff is strictly worse than no deviation.

> **Process note, and the more important finding.** The *first* GPU
> measurement of this said the same thing for the wrong reason and nearly
> shipped a false conclusion. The monkeypatch rewrote `Diffusion._slice`, which
> the tree had meanwhile replaced with `_unbind_draws` + a `draws[...]` list,
> so the patched function was never called. The tell was in the data: the
> "patched" run's deviation was 3.057e-04 Å, identical to the unpatched
> baseline, where the real switch moves `pos` by 6.45 Å. **A monkeypatch
> benchmark must assert that it changed the output before its timing means
> anything.**

### 5.3 Reduced precision — declined, with the mechanism

`matmul_precision("medium")` puts bf16 into cuBLAS with fp32 accumulation and
leaves everything else fp32. It is the clean isolation of "does narrowing the
GEMM inputs below TF32 help?", and the answer is **no: 2.856 ms against TF32's
2.760 ms — slower.** On an H100 bf16 has roughly twice TF32's throughput, so if
these GEMMs were math-bound this had to win. It did not.

That measurement, plus the kernel census (321 cuBLAS launches at 6.5 µs, with
the narrow projections paying the same 6.4 µs as *pure launch latency*), says
the cuBLAS time here is launch-bound, not math-bound. TF32 already collected
everything that was available on the wide projections.

A genuine whole-graph bf16 cast (weights cast once, no autocast) therefore has
only one remaining mechanism: halving traffic in the 524 Triton kernels, which
total 1.27 ms at ~2.4 µs each — already within ~20 % of the ~2 µs launch floor,
so there is at most a small fraction of 0.22 ms in it. Against that: bf16's 8
mantissa bits quantise a 20 Å coordinate at ~0.06 Å, which is most of the
±0.08 Å band the CA-CA gate allows, in a state that is fed back 250 times.

**Declined on my own measurement**, not on the inherited warning. (Autocast is
separately dead — it adds a cast kernel per op boundary and measured ~3×
slower; and a naive `model.to(float16/bfloat16)` currently aborts before
running, on `at::Half` overflow of the `+inf` sentinels and a Float/BFloat16
mismatch.)

### 5.4 Neighbour reuse across blocks ("stride") — crashes Dynamo

Written as the natural stateful closure (recompute the current branch's list
every `k`-th block, reuse otherwise) this raises
`InternalTorchDynamoError: 'InstructionTranslator' object has no attribute
'parent'` at trace time, for stride 2, 3 and 6 alike. Predicted gain ~0.1 ms —
neighbour selection is 3.5 % of the forward and only the current branch is left
after P6 — against a deviation in the same class as halving K. Dropped rather
than worked around.

### 5.5 Shrinking the neighbourhood (K 64 → 48 / 32) — fast, and worse

The only candidate that bought real time on top of TF32: 2.572 ms against
2.760, i.e. **−10 %**. It is also the only one that changes what the network
sees — 32 neighbours instead of 64 — rather than how accurately it computes it,
and it is the only one that **failed the quality gate's spirit** even while
passing its letter:

| | coil | ordered | clash/design | strand |
|---|---|---|---|---|
| exact | 0.076 | 0.924 | 0.1 | 0.319 |
| fast | 0.081 | 0.919 | 0.5 | 0.328 |
| fast + K→32 | **0.109** | 0.891 | **1.1** | 0.290 |

43 % more coil and 11× the clashes, on the same 8 seeds. Dropped. (Getting this
number needed a fixed harness: `runpy.run_path(run_name="__main__")` replaces
the patching module's globals, so a lambda that reads the original factory from
a global resolves against the wrong module and Dynamo aborts with
`module '__main__' has no attribute '_orig_cur'`. Bind originals as default
arguments — `x_gen_k32.py`.)

### 5.6 Bypassing the conditioning-cache key — no gain, and not ours to trade

A first version of the preset skipped the per-step cache key and relied on the
caller's `invalidate_condition_cache()`. Measured worth: `lean 3.563` (bypass
on) against `3.568` (off) — **nothing**, because for de-novo generation `dmap`
and `omap` are absent and the key covers only four small `(L,)` tensors.

It is also precisely the shape of red-team finding N010/S1 (a sum-based key
collided under permutation and silently reused the previous design's
conditioning, 2.56 Å wrong while looking healthy). A stale cache is a bug in
both presets, not a speed/accuracy trade. **Removed.** What survives from it is
the unconditional `fast_denoiser.invalidate_condition_cache()` at the top of
`generate.py`'s per-design loop, which is correct for both presets and costs
nothing.

Residual note for the exact path, not acted on here: the replacement key does
`v.detach().cpu()` per conditioning tensor per step. For de-novo that is ~2 KB
and unmeasurable (exact lean 3.568 ms before the change, 3.554 ms after). With
a template it covers `dmap` `(L,L)` and `omap` `(L,L,9)` — 5.8 MB copied off
the GPU per step at L=400. If that ever matters, the sound fix is to memoise
the digest on each tensor's `(id, _version)` while holding a reference to it
(so ids cannot be recycled), which preserves content-hash semantics exactly.

## 6. The one lever left, and the cheap test of it

The exact path's floor is ~3.4–3.5 ms in ~946 kernels; 321 cuBLAS launches at
6.5 µs are 2.1 ms of it and 524 Triton kernels are 1.27 ms. Against an
arithmetic floor of 0.21 ms, the only thing that removes launches *by the
hundred* is merging kernels — N007's P8 (a hand-written Triton pair-feature
kernel, 1–2 weeks) or inductor's own horizontal fusion,
`torch._inductor.config.combo_kernels`.

`combo_kernels` is reported to crash codegen on the **stable-argsort** node.
The exact path is stuck there — that argsort is the neighbour selector's
contract with JAX. The fast preset is not. So the one hypothesis worth a job:
`topk` costs +0.19 ms on its own (§5.1), but may pay for itself many times over
if removing the argsort node is what lets `combo_kernels` compile.
`x_combo.py` tests `base` / `topk` / `combo` / `combo_topk`.

*(result below)*

On P8 itself: the mechanism is real and it is the largest single number on the
board, but it is not a preset switch — it is a rewrite of
`DiffusionPairFeatures` that both presets want, and with CUDA's own
nondeterminism at 2.2e-04 Å a reassociated Triton kernel may well qualify for
the *exact* path. It belongs in its own task with its own bit-level harness,
not half-landed inside a preset.

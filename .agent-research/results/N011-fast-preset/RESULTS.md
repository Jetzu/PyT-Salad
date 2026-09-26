# N011 — the `fast` inference preset

Measured 2026-09-26 on an H100 (`--partition h100`, one GPU), `default_vp-200k`,
L=100 unless stated, whole-forward `torch.cuda.CUDAGraph` under a
`mode="default"` compile — the same mechanism `FastDenoiser` uses.

Scripts and raw logs live beside this file (`x_fast.py`, `x_fast2.py`,
`x_preset.py`, `ss.py`, `run_*.sbatch`, `logs/`).

## 1. Verdict

*(filled in below from the measurements)*

## 2. Per-forward candidates (L=100, lean step, `x_fast.py`)

`dev_max` is the single-forward `|Δpos|` against the unpatched fp32 eager
model on the SAME input. The input is a random atom cloud at t=0.5, i.e. the
most sensitive point of a chaotic map, so these are a *sensitivity ranking* and
not a prediction of the end-to-end deviation — that is section 4's job.

| variant | ms | vs exact | dev_max (Å) | verdict |
|---|---|---|---|---|
| BASE — everything on, output step | 4.535 | +34% | 3.0e-4 | the pre-N009 fast path |
| EXACT_AA — exact preset, output step | 4.200 | +24% | 3.0e-4 | 1 step in 401 |
| **EXACT — exact preset, lean step** | **3.371–3.393** | — | 3.0e-4 | the baseline |
| need_aux=False | 3.416 | +0.7% | 2.9e-4 | **no gain** — confirms notes/04 |
| topk instead of stable argsort | 3.582 | **+5.6%** | 2.7e-4 | **SLOWER — dropped** |
| current branch K 64→48 | 3.261 | −3.9% | 9.2 | poor value |
| condition branch K 64→32 | 3.294 | −2.9% | 8.6 | poor value |
| both branches K→48 | 3.216 | −5.2% | 9.8 | poor value |
| both branches K→32 | 3.048 | −10.2% | 6.7 | poor value |
| current K→32 | 3.166 | −6.6% | 11.7 | poor value |
| neighbour reuse across blocks (stride 2/3/6) | — | — | — | **crashes Dynamo** |
| fp16 / bf16 whole graph | — | — | — | **aborts** (see §5) |
| `matmul_precision("medium")` (bf16 GEMM) | 2.856 | −15.3% | 2.9e-1 | worse than TF32 |
| **`matmul_precision("high")` (TF32)** | **2.760** | **−18.3%** | 4.1e-1 | **the winner** |
| TF32 + current K→48 | 2.729 | −19.2% | 9.2 | +1% for 9 Å — no |
| TF32 + shared condition draw | 2.845 | −16.1% | 2.9e-1 | no gain (see §5.2) |

## 3. Production preset (`x_preset.py`, the real `FastDenoiser`)

*(filled in)*

## 4. Quality gate (`ss.py`, 8 de-novo designs at L=100, seed 42)

*(filled in)*

## 5. Refuted / dropped — measured, not argued

### 5.1 `topk` instead of the stable argsort — SLOWER

3.582 ms against the exact path's 3.371–3.393 ms, i.e. **+5.6 %**. The idea was
that a full `argsort` of an `(N, N)` matrix followed by a head slice does
`O(N log N)` work to answer an `O(N)` question. It does, and it is still
faster: `topk` returns values as well as indices (so the separate
`distance[knn]` gather it saves is paid back in the kernel), and inductor fuses
the argsort path better. Neighbour selection is 3.5 % of the forward
(N007 §2), so there was never more than ~0.12 ms on the table anyway.

### 5.2 One shared self-conditioning Gumbel draw — no gain

The reasoning was sound and testable: after the P6 hoist only the *random*
stage of the condition branch differs between blocks, so pinning that draw
makes `cond_neighbours` one tensor for the whole trunk, and then `pair_2`'s
seven geometric feature tensors are the same expression on the same inputs in
all six blocks — which is P7's "the condition branch is a prologue" for free,
if inductor CSEs them.

**RETRACTED AND RE-MEASURED.** The first GPU numbers (`x_fast2.py condsame`
3.387 ms vs 3.371, and `tf32_condsame` 2.845 vs 2.848) measured *nothing*: the
monkeypatch rewrote `Diffusion._slice`, which the tree had meanwhile replaced
with `_unbind_draws` + a `draws[...]` list, so the patched function was never
called. The tell was in the data and was nearly missed — the "patched" run's
deviation was 3.057e-04 Å, identical to the unpatched baseline, where the CPU
check of the real switch moves `pos` by 6.45 Å. **A monkeypatch benchmark must
assert that it changed the output before its timing means anything.**

The real switch (`salad_torch/fastopt.SHARED_COND_DRAW`, verified live at
6.4464 Å with a 0.0 Å same-flag rerun as the noise floor) is measured on the
production path in §3.

### 5.3 fp16 / bf16 for the whole graph — aborts, and not worth fixing

`model.to(torch.float16)` dies with `value cannot be converted to type
at::Half without overflow` (the `+inf`/`1e-12` sentinels in the neighbour
selectors and `vec_norm`'s clamp); bf16 dies with a plain Float/BFloat16
mismatch. Making either run means auditing every literal in the geometry code.

Not chased, and the reason is the measurement rather than the effort: with
TF32 doing the GEMMs the remaining 2.8 ms is launch-bound, not
bandwidth-bound — N007 puts the memory floor at ~0.37 ms — so halving the
traffic cannot buy more than a fraction of that. And bf16 carries 8 mantissa
bits, which at a 20 Å coordinate is a 0.06 Å quantisation of the structure
itself.

`matmul_precision("medium")` (bf16 inside the GEMM, fp32 elsewhere) *does*
run, and is **slower** than TF32: 2.856 vs 2.760 ms.

### 5.4 Neighbour reuse across blocks ("stride") — crashes Dynamo

Written as the natural stateful closure (recompute the current branch's list
every `k`-th block, reuse otherwise) this raises
`InternalTorchDynamoError: 'InstructionTranslator' object has no attribute
'parent'` at trace time, for stride 2, 3 and 6 alike. Since the predicted gain
was ~0.1 ms — neighbour selection is 3.5 % of the forward and only the current
branch is left after P6 — against a deviation in the same class as halving K,
it was dropped rather than worked around.

### 5.5 Shrinking the neighbourhood (K 64 → 48 or 32) — the worst ratio measured

| | ms | vs its own baseline | single-forward dev |
|---|---|---|---|
| exact | 3.371–3.393 | — | 3.0e-4 Å |
| both branches K→32 | 3.048 | −10 % | 6.7 Å |
| TF32 | 2.760–2.848 | −16…−18 % | 3.4e-1 Å |
| TF32 + both branches K→32 | 2.572 | −10 % on top of TF32 | 6.8 Å |

Ranked the way the brief asks — speed per unit of deviation — K reduction is
the worst option on the board by three orders of magnitude, and it is also the
one change that alters what the network *sees* (32 neighbours instead of 64)
rather than how accurately it computes. TF32 buys twice the speed for 1/20th
the perturbation. Its end-to-end quality is reported in §4 for completeness.

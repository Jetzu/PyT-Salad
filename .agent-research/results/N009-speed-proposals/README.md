# N009 — the exact path: implementing the N007 ranked proposals

Everything here measures **one denoiser forward at L=100**, `default_vp-200k`,
H100, fp32 `matmul_precision="highest"`, manual whole-forward
`torch.cuda.CUDAGraph` under a `mode="default"` compile. One length per
process (N007 appendix: compiling several shapes in one process degrades
cudagraphs and flattens everything).

| file | what |
|---|---|
| `ablate.sbatch`, `ablate2.sbatch`, `ablate3.sbatch` | the ablation ladder, one process per rung |
| `x_equiv.py` | equivalence harness for P2/P4/P5 — the flags the parity fixtures cannot reach — **with an identical-input noise-floor control** |
| `x_kernels.py` | kernel census (`capture=False`, since a captured graph profiles as one opaque op) |
| `e2e.sbatch` | end-to-end: three designs, same seed, PDBs compared byte for byte |
| `logs/` | raw job output |

## The rule this work is judged by

The parity gate is `tests/test_model_parity.py` on all four configs
(0.00002 A / 0.00020 A for multimotif), plus geometry 38/38, stack and
encoder. Those tests drive the **eager** `Diffusion.forward`, so they cover
every change that lives in the shared modules (P3, P6, the config hoist) but
**not** the `forward_core` flags (P2, P4, P5). Those are covered by
`x_equiv.py`, and there the control comes first: a deviation only counts as a
deviation if it exceeds what the identical call run twice already produces.

## Results

One forward, L=100. `full` = a reporting sampling step (AA decoder + angle head
run); `lean` = the other 499 steps of a 500-step design.

| rung | full ms | lean ms | job |
|---|---|---|---|
| N007 baseline | 4.966 / 4.941 | 4.489 / 4.485 | 903079 |
| + P0 Dynamo graph-break fix (30 graphs -> 1) | 4.886 / 4.870 | 4.006 / 4.015 | 903105 |
| + P3 one-hot x matrix -> row gather | 4.713 | 3.906 | 903105 |
| + P6 hoist the loop-invariant neighbour stages | 4.650 | 3.828 | 903105 |
| + P4 + P5 skip dead pair conditioning, K 64->48 | **4.390** | **3.574** | 903105 |
| (repeats) | 4.384 / 4.365 / 4.385 | 3.569 / 3.556 / 3.564 | 903105, 903126 |

Per proposal, on a lean step: **P0 -0.48, P3 -0.11, P6 -0.08, P4+P5 -0.25 ms**.
P2 is the full->lean column itself: **-0.82 ms**.
End to end, `generate.py --num_aa 100 --fast`: **1.5 s/design**, was ~2.1 s.

Not taken, with reasons: P1 (subsumed by the manual capture, and after P2 the
angle head does not run on 400 of 401 steps); `need_aux=False` (re-measured
post-P0: 4.378 vs 4.385 - nothing); `combo_kernels` (crashes inductor on the
stable-argsort node); `aggressive_fusion` (-1.7 %, but a process-global flag
whose numerics cannot be resolved against a 1.8e-2 CUDA noise floor); P7 (~30
of 321 GEMM launches for 2-3 days, and it spends parity budget); P8 (the only
route to 2.0 ms, and "materially different" numerics by construction - that is
the `fast` preset's licence).

Kernel census (the unit of account the brief asked for): **1349 -> 946
kernels, 403 removed**, 4.872 -> 3.521 ms GPU-busy.

**2.0 ms is not reachable on the exact path. The measured floor is ~3.4-3.5 ms
per lean step**, and notes/04 section 7.8 names where it goes: 60 % of a lean
step is 321 cuBLAS launches at 6.5 us each against a 0.21 ms arithmetic floor.

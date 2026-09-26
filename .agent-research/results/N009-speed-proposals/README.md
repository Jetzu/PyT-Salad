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

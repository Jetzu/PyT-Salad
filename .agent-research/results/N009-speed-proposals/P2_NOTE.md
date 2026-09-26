# P2 and P5: why the PDB test could not decide them, and what did

## The PDB test is void on CUDA

`e2e.sbatch` (job 903106) ran `scripts/generate.py --num_aa 100 --num_designs 3
--fast --seed 1234` four times. Every pair of runs differs, INCLUDING the
control pair `allon` vs `allon2`, which is the same flags and the same seed:

    design_0  allon vs nolean : DIFFERS      (P2 on vs off)
    design_0  allon vs alloff : DIFFERS      (every proposal on vs off)
    design_0  allon vs allon2 : DIFFERS      <-- THE CONTROL
    (same for designs 1 and 2)

So a CUDA `--fast` run is not reproducible at all, and "the written PDB is
unchanged" cannot decide any optimisation. This is not an N009 regression:
nothing in the proposals touches the source.

## Where the nondeterminism comes from (job 903136, `x_nondet.py`)

    index_put_ stage chaining reproducible over 20 reps : True
    selected neighbour indices reproducible             : True   0 differing slots
    index_mean reproducible                             : False  max|d| 1.639e-07

The `index_put_` duplicate-write path (notes/04 section 4, bug B1) was the
prime suspect and is **REFUTED**: it reproduces exactly. The source is
`index_mean` -> `scatter_add`, whose CUDA atomicAdd ordering varies run to run
at 1.6e-7. `Update.forward` calls it twice per block, six blocks deep.

That 1.6e-7 does not stay at 1.6e-7. Each block's `pos` feeds the NEXT block's
spatial and Gumbel neighbour selection, and a stable argsort over near-ties
flips discretely under a 1e-7 nudge - one different neighbour is a different
forward. Measured on one forward:

    NOISE FLOOR: the identical call run twice differs by 1.755e-02 on cuda

i.e. the CUDA denoiser's own run-to-run noise is ~900x the 2e-5 CPU parity
floor, and it is a DISCRETE effect, not accumulated rounding.

## What that leaves as the gate, and what the gate says

CPU is deterministic (`NOISE FLOOR ... 0.000e+00 on cpu`), so the honest test
is the single forward on CPU, and there every proposal is **bit-identical**:

| | CPU (deterministic) | CUDA, against a 1.755e-02 noise floor |
|---|---|---|
| P2  `need_aa=False`, 1 chain  | `torch.equal` **True** | 1.514e-02 = **0.86x** the floor |
| P2  `need_aa=False`, 2 chains | `torch.equal` **True** | 1.307e-02 = **0.74x** the floor |
| P4  dmap/flags terms dropped  | `torch.equal` **True** | 0.000e+00 = 0.00x |
| P4+P5 `pair_cond=False`, K 64->48 | `torch.equal` **True** | 1.611e-02 = **0.92x** the floor |

P3 and P6 are not in this table because they live in the shared modules and are
therefore covered directly by `tests/test_model_parity.py`, which did not move.

**Conclusion.** Every proposal shipped in the `exact` preset is bit-identical
on a deterministic backend, and on CUDA none of them is distinguishable from
the backend's own noise. P2 and P5 stay in `exact`. What must NOT be claimed is
that a given seed reproduces a given protein on CUDA - it does not, with or
without them.

## Follow-up worth its own item

`--seed` does not pin a CUDA `--fast` run. If reproducibility is wanted, the
one op to fix is `index_sum`/`index_mean`. A deterministic segment reduction
(sort-then-segment, or - since the segment count is tiny - a one-hot matmul)
would remove it, at some cost in speed and, for the matmul form, at the cost of
changing the summation order relative to the JAX reference, so it needs its own
parity re-baseline. N009 did not take that on.

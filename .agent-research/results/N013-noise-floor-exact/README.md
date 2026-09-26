# N013 — the `exact` preset, re-examined against the MEASURED noise floor

The brief: N009 refused `aggressive_fusion`, `need_aux=False` and P7 on a
standard stricter than the hardware honours, because CUDA's own run-to-run
spread was reported as 2.2e-04 A. Measure that floor properly, then re-judge.

All numbers: H100, `default_vp-200k`, fp32 `matmul_precision="highest"`,
`preset="exact"`, manual whole-forward CUDA graph, **one length per process**.

| file | what |
|---|---|
| `x_floor.py` | the floor harness: N identical forwards on ONE frozen input, full distribution of pairwise max\|d\| |
| `x_detprobe.py` | isolated: does a deterministic segment reduction exist, and what does it cost |
| `x_cpu_bits.py` / `cmp_bits.py` | the CPU bit-equality gate for a process-global inductor flag |
| `x_p7_equiv.py` | P7 batched-vs-sequential on real weights, all four configs |
| `x_p7_cpu_sweep.py` | P7's per-step perturbation along a real trajectory, deterministic backend |
| `cmp_floor.py` | two rep-sets: within-A / within-B / cross distributions + bit-pattern clusters |
| `run_parity.sh` | the four checkpoint parity tests + geometry/stack/encoder/sampling |
| `logs/` | raw job output (903240, 903241, 903260, 903261, 903262, 903291, 903293) |

## Results

Picked up (with N014) from a prior agent that died on a rate limit without a
headline number. This item's own two deliverables:

### The noise floor, measured properly

CUDA run-to-run self-divergence on an identical, frozen input (`x_floor.py`,
40 reps, H100, `default_vp-200k`, `preset=exact`) is **~1e-5 A on `pos`**
(mean 8.6e-06, p95 1.4e-05, max 2.6e-05 at L=100 on the real-trajectory state,
`floor2_903625.log`), not the 2.2e-4 A figure N009/N010 had been citing. CPU is
exactly 0.0 (segment reductions there are deterministic by construction). This
re-judges N009's refusal of `aggressive_fusion`/P7 on "a standard stricter than
the hardware honours" - see N014 for what actually gates those.

### Two shipped optimisations, at their own floors

- **`aggressive_fusion` (inductor scheduler knob)**: BIT-IDENTICAL, not a
  numerics trade at all. Verified twice: on CPU (`x_cpu_bits.py`/
  `cmp_bits.py`, `cpubits_903241.log`, L=64 and L=100, all eleven outputs,
  `WORST 0.000000e+00`), and on the H100 itself under the deterministic
  segment reduction (`cmp_floor.py dg_synth_base dg_synth_aggr`, one shared
  bit pattern - `detgate_903291.log`). Worth -1.4 to -2.4% of a lean step at
  L=50/L=100 (see N014's README for the isolated-arm timing table, since D1's
  headline number covers both this and P7 together).
- **`det_segment_sum` (`SALAD_OPT_DETSEG`)**: a one-hot-matmul replacement for
  `index_sum`'s `scatter_add`, which lowers to a nondeterministic `atomicAdd`
  (~1.6e-7/call on H100) that the neighbour argsort amplifies into a discrete
  tie flip. The matmul form is reproducible by construction (`x_floor.py`
  --seed: 40 identical forwards -> 1 distinct bit pattern instead of 40,
  `det_903261.log`/`detgate2_903317.log`), at +2.1% of the step
  (3.634 ms lean / 4.457 ms full vs `exact`'s 3.558/4.384, same job).

  **It is its OWN parity baseline, and does NOT hold `exact`'s sampling
  floor.** Reproduced independently this session (job 903683, isolated
  snapshot, 2x2 reps, bit-identical within each setting; see N014's README
  D3 section for the full numbers): DETSEG=1 fails `test_sampling_parity`
  (final structure 0.19861 A / ~3.2x the DETSEG=0 drift from JAX; **7 of 64
  residues take a different amino acid identity** vs 1/64 at DETSEG=0 - a
  discretely different design). The four checkpoint parity configs,
  geometry 38/38, stack and encoder all still hold at the usual floors under
  DETSEG=1 - only sampling moves. Shipped as its own preset,
  `PRESETS["deterministic"]` in `salad_torch/modules/fast.py`, NOT as the
  default, with this finding recorded in its docstring (verified already
  correct this session, not rewritten - see N014 README D3).

### What this re-judges from N009/N010

N009 refused P7/`aggressive_fusion`/`need_aux=False` against a 2.2e-4 A floor
that was never actually measured on this configuration; the real floor is
~1e-5 A and both of the above clear it once the "is it bit-exact or
reassociation-only" question is answered directly (bit-exact for
`aggressive_fusion`; reassociation for P7 - see N014 for its own dedicated
gate, since P7 is N014's deliverable, not this item's).

## Outputs

- `x_floor.py`/`x_detprobe.py`/`x_cpu_bits.py`/`cmp_bits.py`/`cmp_floor.py` -
  harnesses (pre-existing, this session only read their logs)
- `run_arms.sbatch`/`run_arms2.sbatch` - per-flag isolated timing on synthetic
  input (superseded for the D1 headline by N014's `bench_one.py`-based
  2x2, which uses real weights/preset plumbing and one-length-per-process
  discipline; kept here as corroborating evidence, same direction and
  magnitude)
- `run_det.sbatch`/`run_detgate.sbatch`/`run_detgate2.sbatch`/`run_final.sbatch` -
  the DETSEG determinism + parity-under-DETSEG jobs
- `salad_torch/geometry/segment.py: det_segment_sum` + `opt.DET_SEGMENT` /
  `opt.AGGRESSIVE_FUSION` flags, `salad_torch/modules/fast.py:
  PRESETS["deterministic"]` - all landed this session (see commit)

## Next

Nothing further planned. `PRESETS["deterministic"]` has no sampling-quality
baseline of its own recorded (by design - see its docstring); anyone reporting
results FROM that preset should record one first rather than reading it
against `exact`'s.

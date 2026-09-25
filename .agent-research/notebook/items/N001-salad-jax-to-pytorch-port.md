# N001 · SALAD JAX to PyTorch port

type: task · date: 2026-09-25 · status: open

## Goal

Full PyTorch port of **salad** (sparse all-atom denoising, github.com/mjendrusch/salad,
Apache-2.0 code / CC-BY-4.0 weights), inference AND training, loading the released
weights unchanged and matching JAX numerically and in speed.

Note: FlexCraft is a *separate* repo by the same author that wraps salad in pipelines,
not another name for it.

## Plan

Four-role agent workflow throughout (reader -> writer -> control -> red team), with the
reader's spec persisted so the writer never reads the JAX source; that makes the control
agent's equivalence check non-circular.

Verification ladder, each level gating the next: pure geometry functions -> one trunk
block -> 6-layer trunk -> whole denoiser -> full sampling trajectory -> generated
structure statistics.

## Changes

`salad_torch/`: geometry (frames, neighbours, pair features, segment ops, sidechains),
modules (base layers, sparse IPA, diffusion blocks, decoders, encoder, model),
diffusion (schedules, noising, preconditioning, sampling), losses (12 terms),
data + training, io (PDB), convert (Haiku checkpoint -> state dict).

Specs the code was written from: `notes/specs/01..06`.
Measured results write-up: `notes/04_PARITY_AND_PERF.md`.

## Evidence

Parity, all against the real released checkpoints
(`/hnvme/workspace/v181bb10-salad/*.npz`):

| level | test | result |
|---|---|---|
| geometry, 38 functions | `tests/test_geometry_parity.py` | exact, worst 4e-6 |
| 6-layer trunk | `tests/test_stack_parity.py` | 0.0003 A max coord |
| whole denoiser, `default_vp` | `tests/test_model_parity.py` | 0.00002 A max, aatype 0/48 |
| whole denoiser, `default_ve_scaled` | same, `SALAD_CONFIG=default_ve_scaled` | 0.00002 A max |
| 12 training loss terms | `tests/harness/check_loss.py` | worst 2.1e-7, total 7.2e-8 |
| 81-step trajectory | `tests/test_sampling_parity.py` | within the chaos noise floor |

All four released checkpoints load with `load_state_dict(strict=True)`:
default_vp / default_vp_scaled 779 tensors, default_ve_scaled 827, multimotif_vp 851.

Full generation, 5 x 100 aa, 500 steps, H100 (job 899773, `outputs/gen_torch`,
`outputs/gen_jax`): torch CA-CA 3.769 +- 0.005 A, Rg 12.5 A, helix 63%;
JAX 3.764 +- 0.014 A, Rg 14.0 A, helix 63%. Statistically indistinguishable.

Bugs found that only numerical parity could catch:
- `jax.nn.gelu` defaults to the TANH approximation, `torch.nn.functional.gelu` to exact
  erf. Wrong variant = 0.31 A max deviation per denoiser call over the trunk.
- `violation/angles_ca_c_n` was 2.12x too small: AlphaFold-multimer uses the C-N
  BOND-LENGTH stddev (0.014) as the tolerance for a cosine-ANGLE error, not the angle
  stddev (0.0311). The reader spec assumed the natural pairing.
- `apply_diffusion` passed `noise=` to a function whose kwarg is `uniform=`, so the
  entire noise->denoise entry point raised `TypeError` on every call. The parity tests
  missed it because they feed `pos_noised` from the fixture.

## Outcome

Numerically on par. NOT yet on par for speed — see Next.

## Outputs

- `notes/00_SETUP.md`, `01_INFERENCE_PATH.md`, `02_ARCHITECTURE.md`,
  `03_GEOMETRY_AND_DIFFUSION.md`, `04_PARITY_AND_PERF.md`
- `notes/specs/01..06` (reader specs)
- fixtures + harnesses: `fixtures/*.npz`, `tests/harness/`
- generated structures: `outputs/gen_torch/`, `outputs/gen_jax/` (job 899773)
- benchmarks: `outputs/bench_*.json`, `outputs/prof_899930.log`

## Next

**Open problem: 6x too slow.** Per 100 aa design, 500 steps, H100: JAX 1.6 s, torch
compiled 9.7 s (eager 30.9 s). One-time compile warmup is acceptable; per-design cost
is not.

Diagnosis: the model is launch-bound, not compute-bound. Eager time is a flat ~75 ms
from L=50 to L=800, and one DiffusionBlock (8 ms) costs more than JAX's whole forward
(3.1 ms). `dynamo.explain` found 31 graphs / 30 graph breaks / 2244 ops, which disables
CUDA graphs for the region. Profile at L=100: blocks 64%, aa_decoder 15%, neighbours 15%.

Done so far: `Diffusion.forward_core` (tensor-in/tensor-out so Dynamo can trace it),
RNG hoisted out of the graph as a pre-drawn `(19, N, N)` stack, `.item()` removed from
the segment ops, the uniform-pool counter replaced by trace-constant indices, and
`Rigid` converted to a NamedTuple. 75 -> 19 ms so far.

Handed to Codex as an independent voice (see TODO).

Also open: `default_vp_scaled` and `multimotif_vp` verified structurally but not
numerically; the training loop is verified to train but not compared to JAX's; the
training-time conditioning reveal is a reconstruction, not a transcription.

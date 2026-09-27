# salad-torch

PyTorch implementation of **salad** (sparse all-atom denoising), a diffusion
model for protein backbone generation. Loads the released salad checkpoints
directly; no JAX installation is required.

Generates protein backbones with optional conditioning on secondary structure,
a template motif, or cyclic/screw symmetry, for chains up to ~1000 residues.

Paper: *Efficient protein structure generation with sparse denoising models*,
bioRxiv 2025.01.31.635780. Original implementation:
[mjendrusch/salad](https://github.com/mjendrusch/salad).

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Presets](#presets)
- [Throughput](#throughput)
- [Flags](#flags)
  - [Model and output](#model-and-output)
  - [What to generate](#what-to-generate)
  - [Conditioning](#conditioning)
  - [Symmetry](#symmetry)
  - [Sampling schedule](#sampling-schedule)
  - [Performance](#performance)
  - [Reproducibility](#reproducibility)
- [Operational notes](#operational-notes)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [License](#license)

## Install

```bash
git clone https://github.com/Jetzu/PyTorch-Salad.git
cd PyTorch-Salad
conda env create -f environment.yml
conda activate salad-torch
pip install -e .
```

Download the checkpoints:

```bash
wget https://zenodo.org/records/14711580/files/salad_params.tar.gz
tar -xzf salad_params.tar.gz
```

`environment.yml` pins Python 3.11, PyTorch 2.5.1+cu121 and numpy 1.26. numpy
must stay below 2.0: the checkpoints are array pickles that need the 1.x ABI.
Drop the `pytorch-cuda` line for a CPU-only install — the eager path works
there, but `--fast`, `--group_size` and `--batch_size` require CUDA.

Twelve checkpoints ship. Four are verified in the test suite: `default_vp`,
`default_vp_scaled`, `default_ve_scaled`, `multimotif_vp`. The rest load but
are untested.

## Quick start

```bash
python scripts/generate.py \
    --config default_vp \
    --params params/default_vp-200k.jax \
    --num_aa 100 \
    --num_designs 10 \
    --out_path outputs/run1/ \
    --fast --preset fast --group_size auto
```

Writes `outputs/run1/design_0.pdb` … `design_9.pdb`.

Drop `--fast` for the eager path: same output, roughly 20x slower per step, and
the only option on CPU.

A 1000-residue chain, using the checkpoint trained for long proteins:

```bash
python scripts/generate.py \
    --config default_ve_scaled \
    --params params/default_ve_scaled-200k.jax \
    --num_aa 1000 \
    --num_designs 10 \
    --out_path outputs/large/ \
    --fast --preset fast
```

`--group_size` is omitted here on purpose — at that length a single forward pass
already saturates the GPU, so grouping gains nothing.

## Presets

`--preset` selects a point on the speed/precision curve. Only applies with
`--fast`.

| preset | what it changes | step at L=100 |
|---|---|---|
| `exact` (default) | nothing; full-precision reference path | 2.53 ms |
| `fast` | TF32 matmuls, half the sampling steps, two reassociated contractions | 2.00 ms |
| `deterministic` | `exact` plus non-atomic segment reductions | ~2.7 ms |

`fast` was validated against `exact` at 160 designs per arm on CA–CA bond
length, radius of gyration, secondary-structure composition, clash count and
divergence rate; none of those distributions shift measurably. Most of its
per-design advantage comes from the halved schedule rather than the faster step.

`deterministic` exists because `--seed` alone does not pin a CUDA run (see
[Reproducibility](#reproducibility)). It is slower than `exact` and its output
differs slightly from both other presets, so use it to reproduce a run, not as
a default.

## Throughput

H100, 500-step schedule, per finished design.

| configuration | L=100 |
|---|---|
| eager (no `--fast`) | ~30 s |
| `--fast --preset exact` | 1.27 s |
| `--fast --preset fast` | 1.00 s |
| `--fast --preset fast --group_size auto` | ~0.22 s |

`--group_size` reclaims GPU idle time between one design's kernel launches, so
it helps most on short chains and large GPUs. `--batch_size` packs several
designs along the residue axis so each kernel does more work. Throughput
saturates at about 16 designs in flight.

| length | best configuration | s/design |
|---|---|---|
| 50 | `--batch_size 8 --group_size 4` | 0.145 |
| 100 | `--batch_size 8 --group_size 2` | 0.281 |
| 200 | `--batch_size 4 --group_size 2` | 0.563 |
| 400 | `--batch_size 2 --group_size 2` | 1.130 |
| 1000 | `--group_size 2` | 2.855 |

Rule of thumb: keep `batch_size x length` between 400 and 800 residues, so
`B ~ 600/L` rounded to a power of two, then `--group_size auto`. Above ~200
residues the model becomes arithmetic-bound rather than launch-bound and these
knobs stop paying; at L=1000 `--batch_size` is a net loss because the pairwise
tensors grow quadratically in B.

Peak VRAM is approximately `(5.73*L + 235) + (N-1)*4.00*L` MiB for group size N,
linear in length to at least L=1000. A single design at L=1000 fits in 8 GB.

## Flags

### Model and output

| flag | default | |
|---|---|---|
| `--config` | `default_vp` | model configuration |
| `--params` | — | checkpoint path; must match `--config` |
| `--out_path` | `outputs/designs` | output directory, created if absent |
| `--num_designs` | 1 | how many to generate |
| `--sequence_only` | off | write FASTA to `{out_path}/sequences.fa` instead of PDB |
| `--device` | `cuda` if available | |

### What to generate

`--num_aa` (default `100`) takes an integer, or colon-separated chain lengths.
A `c` prefix makes that chain cyclic.

```
--num_aa 100          one chain of 100
--num_aa 100:200      two chains
--num_aa 100:200:c8   plus an 8-residue cyclic chain
```

Residue numbering starts at 0 and each chain starts 50 past the previous
chain's last residue, so `100:100` numbers residues `0..99` and `149..248`.
This numbering reaches the output PDB.

`--num_aa` is overridden by a literal `--dssp` string and by `--template`.

`--merge_chains` zeroes the chain index before the model sees it. It has no
effect when `--dssp` or `--template` is given, since both reassign chains.

`--order_agnostic` hides residue order from the network (`resi=-1`, `chain=0`).
The output PDB is re-threaded by following C–N backbone contacts.

### Conditioning

**`--dssp`** — `none`, `random`, or a per-residue string over `L`/`H`/`E`/`_`
(loop, helix, strand, unconstrained):

```
--dssp HHHHHHHHHH____EEEEEEEE____HHHHHHHHHH
```

A literal string sets the length and overrides `--num_aa`.

**`--dssp_mean`** — `none`, `random`, or `loop,helix,sheet` as target
fractions. Not renormalised; the three values must already sum to 1.

**`--template`** — `<pdb_path>,<constring>`. The constraint string has one
character per residue plus `:` chain separators; `F` marks residues taken from
the template, anything else is free. Overrides `--num_aa` and `--dssp`, and
disables cyclic conditioning.

```
--template motif.pdb,FFFFFF____FFFFFF
```

**`--template_aa`** additionally conditions on the template's sequence, not
just its geometry.

**`--cloud_std`** overrides the initial point-cloud standard deviation.

### Symmetry

Cyclic and screw symmetry. Requires the eager path — incompatible with
`--fast`, and therefore slow.

```bash
--sym c4 --num_aa 60:60:60:60
```

| flag | |
|---|---|
| `--sym` | `cN` sets `--replicate_count=N`, `--screw_angle=360/N`, `--screw_translation=0` as overridable defaults. Needs `--num_aa` as N colon-separated chains. |
| `--replicate_count` | number of subunits |
| `--screw_angle` | degrees about Y |
| `--screw_radius` | offset of the representative subunit from the axis |
| `--screw_translation` | rise per subunit along Y; 0 gives point-group Cn, nonzero a helical screw |
| `--sym_mean` | literal `True`/`False`; controls the initial and per-step symmetrisers |
| `--sym_threshold` | post-hoc symmetrisation runs only while `raw_t >` this; 0.0 symmetrises every step |

### Sampling schedule

| flag | default | |
|---|---|---|
| `--num_steps` | 500 | denoising steps |
| `--out_steps` | 400 | comma-separated steps at which to write output; must ascend |
| `--prev_threshold` | 0.8 | self-conditioning threshold |
| `--timescale_pos` | `cosine(t)` | position noise schedule; also `linear`, `none`, `framediff`, `ve(...)`, `edm(...)` |

The defaults write output at step 400 of 500. `--preset fast` halves both.

**`--renoise`** runs 9 additional partial re-denoise passes at each out-step,
re-entering at `raw_t=0.4` with self-conditioning off. Roughly a 3x wall-clock
multiplier, and switches PDB naming to `result_{design}_{idr}_{step}.pdb`.

**`--sample_final`** replaces the single-shot sequence decode at each out-step
with an L-step autoregressive re-decode, then rebuilds sidechains.
**`--sample_final_temperature`** (default 0.1) is its softmax temperature.

**`--depth_adjust`** — `none` or an integer up to the checkpoint's trunk depth
(6 for `default_vp`): an early-exit ablation that loads only the first N blocks.

### Performance

**`--fast`** compiles the denoiser and captures it in a CUDA graph, so each
sampling step replays one graph instead of dispatching ~700 kernels from
Python. Costs a one-time compile of about five minutes. GPU only; one fixed
length per process; incompatible with `--sym`.

**`--preset`** — see [Presets](#presets).

**`--group_size`** — `auto` or an integer. Advances N independent designs
through one CUDA graph. This is not a batch axis: each design keeps its own
buffers, noise and RNG stream and produces bit-identical output to being
generated alone. `auto` measures the best group size on the current device by
capturing at increasing N until the gain falls below 3% or VRAM runs out;
prefer it to a fixed integer, which will be wrong on a different GPU.

**`--batch_size`** — packs B designs along the residue axis. Unlike
`--group_size` this changes each design's arithmetic, because a larger matmul
picks a different tiling and accumulation order; the resulting structures are
distributionally equivalent but not identical. Worth about +42% at L=50 and
+16% at L=100 on top of `--group_size`, nothing at L=400, and a loss at L=1000.

**`--compile`** passes a `torch.compile` mode for the eager path, independently
of `--fast`. **`--timing_json`** writes a timing breakdown to the given path.

### Reproducibility

`--seed` (default 42) seeds the noise draw. **On CUDA this does not pin the
output.** Segment reductions use `scatter_add`, whose atomic accumulation order
varies between runs; the neighbour selection turns that into a discrete tie
flip, and the sampling steps amplify it. Two identical commands differ in about
48% of residues.

Add `--preset deterministic` to pin a CUDA run. CPU is deterministic already.

`--jax_seed` is an alias for `--seed`, accepted so command lines from the
original implementation run unchanged. It does not make output match across
implementations.

## Operational notes

**Do not pad batches.** A `-1` sentinel marks "no neighbour" and indexes the
last residue, which is then counted twice in attention. The same chain at a
different offset in the flat residue axis can differ by several Ångström.
Concatenate along the residue axis instead. This matches the original
implementation's behaviour.

**`--fast` compiles per shape.** Generating several lengths in one process
degrades the captured graphs; run one length per process. Set
`TORCHINDUCTOR_CACHE_DIR` to persistent shared storage, otherwise the ~5 minute
compile is repeated on every machine and job — `/tmp` is usually node-local.
Budget roughly 3800 files and 280 MB per compiled length.

**Distributed training needs `find_unused_parameters=True`.** 202 parameters
never receive gradient: the checkpoints contain bias tensors for layers
configured without bias, and those are never added.

**Comparing sampling trajectories** requires a relative tolerance. A single
forward pass is reproducible to ~2e-7, but 500 steps of feedback amplify
floating-point reduction-order differences. Compare against a rerun of the
same code, not against a fixed threshold.

## Tests

```bash
python tests/test_model_parity.py        # per-config, via SALAD_CONFIG
python tests/test_geometry_parity.py
python tests/test_design_quality.py --n 160 --preset fast
```

`test_model_parity.py` checks the four verified checkpoints against stored
reference outputs. `test_design_quality.py` generates designs and checks their
geometry — bond lengths, radius of gyration, divergence rate — which the
per-tensor tests cannot see, since a defect that only appears after hundreds of
sampling steps is invisible to a single forward pass.

## Repository layout

```
salad_torch/
  config.py        model configurations
  geometry/        frames, neighbour selection, pair features, segment ops, sidechains
  modules/         layers, sparse IPA, diffusion blocks, decoders, the model
                   fast.py — compiled + CUDA-graph inference path and presets
  diffusion/       noise schedules, forward noising, preconditioning, sampling
  losses/          training loss terms
  io/              PDB output, residue constants
  convert/         checkpoint conversion
  training/        training loop and data pipeline
scripts/generate.py
tests/
docs/USAGE.md      extended flag reference and environment variables
```

Training is implemented, including all loss terms and bin-packed batching. The
real-PDB data source is implemented but untested; the synthetic source works.

## License

Copyright (c) 2025 European Molecular Biology Laboratory (EMBL).

Code is licensed under the Apache License 2.0. Model parameters are licensed
under Creative Commons Attribution 4.0 International. This port carries the
same terms.

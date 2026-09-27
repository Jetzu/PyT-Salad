# salad-torch

PyTorch port of [salad](https://github.com/mjendrusch/salad) (sparse all-atom
denoising). Loads the released JAX checkpoints unchanged. No JAX dependency.

`--preset exact` matches the JAX original to 0.00002 A on the four verified
checkpoints. Inference and training are both ported.

Four flags have no upstream counterpart: `--fast` (compile + CUDA graph),
`--preset`, `--group_size` and `--batch_size`. Everything else behaves as
upstream.

## What's salad?

salad (**s**parse **al**l-**a**tom **d**enoising) is the name of a family of machine learning models for controlled protein structure generation.
Like many previous approaches to protein structure generation (e.g. [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), [Genie2](https://github.com/aqlaboratory/genie2) and [Chroma](https://github.com/generatebio/chroma)), salad is based on denoising diffusion models and is trained to gradually transform random noise into realistic protein structures.

Unlike previous approaches, salad was developed from the ground up with efficiency in mind. salad outperforms its predecessors by a large margin and thus allows for higher-throughput structure generation. To give an example of how fast salad really is in comparison to previous methods, imagine you want to generate a set of 1,000 amino acid proteins on your previous generation RTX 3090. You start generation and go on a 10 minute coffee break. If you used salad with default settings, you come back to ~45 generated structures. If you used RFdiffusion, you might come back to find one or two.

<p align="center">
  <img src="./assets/SALADRuntimeDark.png" alt="alt text" width="400px" align="middle"/>
</p>

*Comparison of per-structure runtime for salad models compared to Chroma, Genie2 and RFdiffusion.*

Like previous models, salad models can generate protein structures for a variety of protein design tasks:
* [unconditional protein generation](#de-novo-protein-design-script)
   up to ~1,000 amino acids
* [symmetric and repeat protein generation](#symmetric--repeat-protein-generation)
  (currently cyclic and screw-axis symmetry)
* [protein motif scaffolding](#multi-motif-scaffolding)
  (with one or more independent motifs)

Upstream additionally supports shape-conditioned generation and multi-state
design. Those are **not ported here** and their flags do not exist in
`scripts/generate.py`; use the [JAX implementation](https://github.com/mjendrusch/salad)
for them. The full extent of what salad can do is described in the
[manuscript](https://www.biorxiv.org/content/10.1101/2025.01.31.635780v1.abstract).

## What's in this repository?

The PyTorch port of the models described in "Efficient protein structure
generation with sparse denoising models". Module names differ from upstream:

```
salad_torch/
  config.py            the configurations with released weights
  geometry/            frames, neighbour selection, pair features, segment ops, sidechains
  modules/             base layers, sparse IPA, diffusion blocks, decoders, the model
                       fast.py - the compiled + CUDA-graph inference path and its presets
  diffusion/           noise schedules, forward noising, preconditioning, sampling
  losses/              the training loss terms
  io/                  PDB output, residue constants
  convert/             JAX/Haiku checkpoint -> PyTorch state dict
  training/            the training loop and data pipeline
scripts/generate.py    the one generation entry point
tests/                 parity suites - run them, they are the point
docs/USAGE.md          every CLI flag
```

Upstream's `salad.modules.noise_schedule_benchmark` becomes
`salad_torch.modules.diffusion` + `modules/blocks.py`; `salad.modules.geometric`
becomes `salad_torch/geometry/`; the autoencoders are in `modules/encoder.py`.

`tests/test_model_parity.py` checks all four verified checkpoints against stored
JAX outputs. `tests/test_design_quality.py` generates designs and checks their
geometry.

**NOTE: developed and tested on Linux with CUDA. The eager path runs on CPU;**
**`--fast`, `--group_size` and `--batch_size` are CUDA-only.**


## License
Copyright (c) 2025 European Molecular Biology Laboratory (EMBL)

### Code license
Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

  > > [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

### Model parameter license
 <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">The model parameters</span> are licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a></p> 

# Table of contents

- [salad-torch](#salad-torch)
  - [What's salad?](#whats-salad)
  - [What's in this repository?](#whats-in-this-repository)
  - [License](#license)
    - [Code license](#code-license)
    - [Model parameter license](#model-parameter-license)
- [Getting started](#getting-started)
  - [Installing salad-torch](#installing-salad-torch)
  - [Generating your first proteins](#generating-your-first-proteins)
  - [Speed](#speed)
    - [Choosing a configuration](#choosing-a-configuration)
- [salad scripts documentation](#salad-scripts-documentation)
  - [_De novo_ protein design script](#de-novo-protein-design-script)
    - [--config](#--config)
    - [--params](#--params)
    - [--out_path](#--outpath)
    - [--num_aa](#--numaa)
    - [--merge_chains (default: False)](#--mergechains-default-false)
    - [--num_steps (default: 500)](#--numsteps-default-500)
    - [--out_steps (default: 400)](#--outsteps-default-400)
    - [--num_designs (default: 10)](#--numdesigns-default-10)
    - [--prev_threshold (default: 0.8)](#--prevthreshold-default-08)
    - [--timescale_pos (default: "cosine(t)")](#--timescalepos-default-cosinet)
    - [--cloud_std (default: "none")](#--cloudstd-default-none)
    - [--dssp (default: "none")](#--dssp-default-none)
    - [--template (default: "none")](#--template-default-none)
    - [--fast](#--fast)
    - [--preset (default: "exact")](#--preset-default-exact)
    - [--group_size (default: off)](#--groupsize-default-off)
    - [--batch_size (default: 1)](#--batchsize-default-1)
    - [--sequence_only](#--sequenceonly)
    - [--order_agnostic](#--orderagnostic)
    - [--depth_adjust (default: "none")](#--depthadjust-default-none)
    - [--seed / --jax_seed](#--seed----jaxseed)
    - [--device, --compile, --timing_json](#--device---compile---timingjson)
  - [Multi-motif scaffolding](#multi-motif-scaffolding)
  - [Symmetric / repeat protein generation](#symmetric--repeat-protein-generation)
    - [--num_aa](#--numaa)
    - [--replicate_count](#--replicatecount)
    - [--screw_translation](#--screwtranslation)
    - [--screw_radius](#--screwradius)
    - [--screw_angle](#--screwangle)
- [Things that will bite you](#things-that-will-bite-you)

# Getting started
## Installing salad-torch

1. Clone this repository and enter it.
2. Create the environment: `conda env create -f environment.yml`
3. `conda activate salad-torch && pip install -e .`

`environment.yml` pins Python 3.11, torch 2.5.1+cu121 and numpy 1.26 - the
versions this port is measured on. Drop the `pytorch-cuda` line for a CPU-only
install; the eager path works there, `--fast`/`--group_size`/`--batch_size` are
CUDA-only. numpy is pinned below 2.0 because the released checkpoints are
pickles of JAX arrays and need the 1.x ABI.

No JAX is required. The released JAX checkpoints are pickles of JAX arrays and
are read directly:

```
wget https://zenodo.org/records/14711580/files/salad_params.tar.gz
tar -xzf salad_params.tar.gz
```

Four configurations have verified parity against the JAX original:
`default_vp`, `default_vp_scaled`, `default_ve_scaled` and `multimotif_vp`.
The other eight released checkpoints load, but have no parity fixture.


## Generating your first proteins

Everything goes through one script, `scripts/generate.py` - there is no separate
entry point for large proteins, and no separate script per feature.

```bash
python scripts/generate.py \
    --config default_vp \
    --params params/default_vp-200k.jax \
    --out_path output/my_first_protein/ \
    --num_aa 100 \
    --num_designs 10 \
    --fast --preset fast --group_size auto
```

The first five options mean exactly what they do upstream: `--config` and
`--params` pick a model and its checkpoint, `--out_path` is the output
directory, `--num_aa` is the length (colon-separated for several chains, e.g.
`100:50`, and a `c` prefix makes a chain cyclic), and `--num_designs` is how
many to generate.

The last three are new and are the whole reason this port exists:

`--fast` compiles the denoiser and captures it in a CUDA graph. It costs about
five minutes once and makes every step ~20x cheaper, so use it for anything
beyond two or three designs. GPU only, one fixed length per process.

`--preset` chooses how much numerical deviation you will accept:
`exact` (the default) reproduces JAX salad to its parity floor, `fast` trades
some numerics for roughly a 25% faster step and half the sampling schedule, and
`deterministic` makes `--seed` actually pin a CUDA run.

`--group_size auto` advances several independent designs through one CUDA graph.
It is **not** a batch axis: each design keeps its own buffers, noise and RNG
stream and comes out bit-identical to generating it alone. `auto` measures the
right group size on your actual GPU.

Large proteins use the same script - no `scripts/generate.py` equivalent is needed,
though `default_ve_scaled` is still the better checkpoint for them:

```bash
python scripts/generate.py \
    --config default_ve_scaled \
    --params params/default_ve_scaled-200k.jax \
    --out_path output/my_large_protein/ \
    --num_aa 1000 \
    --num_designs 10 \
    --fast --preset fast
```

At L=1000, drop `--group_size`: a single forward already saturates the GPU, so
grouping buys nothing (see below).

**[Every flag, in detail: `docs/USAGE.md`](docs/USAGE.md)**

## Speed

Measured on an H100 at L=100, against the JAX original on the same node.

| | per step | per design | vs JAX |
|---|---|---|---|
| JAX salad | 4.73 ms | 1.56 s | - |
| `--fast --preset exact` | **2.53 ms** | 1.27 s | **1.87x** |
| `--fast --preset fast` | **2.00 ms** | 1.00 s | **2.37x** |
| `--fast --preset fast --group_size auto` | 2.00 ms | **~0.22 s** | **~7x** |

Without `--fast` the eager path runs at about 30 s per design; it exists for
debugging and for CPU, not for production.

### Choosing a configuration

`--group_size` pays when the GPU is idle between one design's kernel launches,
which is a short-protein situation. Throughput saturates at about 16 designs in
flight; past that it flattens and then regresses.

| length | best configuration | s/design | speedup |
|---|---|---|---|
| 50 | `--batch_size 8 --group_size 4` | 0.145 | 3.4x |
| 100 | `--batch_size 8 --group_size 2` | 0.281 | 2.1x |
| 200 | `--batch_size 4 --group_size 2` | 0.563 | 1.44x |
| 400 | `--batch_size 2 --group_size 2` | 1.130 | 1.20x |
| 1000 | `--group_size 2` only | 2.855 | 1.03x |

The rule is `B x L` in 400-800 residues: pick `B ~ 600/L` rounded to a power of
two, then `--group_size auto`. At L=1000 packing *loses* - above about 208
residues arithmetic rather than launch latency becomes the binding constraint,
and a single long forward already fills the machine.

Peak VRAM is `(5.73*L + 235) + (N-1)*4.00*L` MiB and stays linear to L=1000. A
single design fits in 8 GB even at L=1000; on a 24 GB card the practical limit
is about `--group_size 5` at L=1000, and roughly 16 at L=100.

# salad scripts documentation

> **One script.** Upstream ships a separate evaluation script per task; this
> port has a single entry point, `scripts/generate.py`, and the task is chosen
> by flags. The options below are upstream's, reproduced faithfully including
> the quirks, followed by the flags that exist only here.
>
> **[`docs/USAGE.md`](docs/USAGE.md) is the full reference** - every flag, the
> environment variables, and the compile-cache setup.

## _De novo_ protein design script
To generate protein structures _de novo_, without structure-editing
or multi-motif scaffolding, you can use the script  `scripts/generate.py`. It allows the use of different
model configurations and checkpoints and allows to apply simple
constraints such as secondary structure, motifs, length and number of chains, etc.

For example, to use a variance-preserving (VP) salad model to generate
a set of 100 amino acid proteins, you could run the following:

```bash
python scripts/generate.py \
    --config default_vp \
    --params params/default_vp-200k.jax \
    --out_path path_to_outputs/ \
    --num_aa 100 \
    --num_steps 500 \
    --out_steps 400 \
    --num_designs 10 \
    --prev_threshold 0.8 \
    --timescale_pos "cosine(t)"
```

If instead you wish to generate a complex of a 100 amino acid protein
and a 8 amino acid cyclic peptide ligand, you merely need to set
`--num_aa 100:c8`.

To change the type of model you want to use, choose a different
`--config` and adjust the selected `--params` and other options
accordingly. For example, to instead run a VE variance-expanding (VE)
model, you could run:
```bash
python scripts/generate.py \
    --config default_ve_scaled \
    --params params/default_ve_scaled-200k.jax \
    --out_path path_to_outputs/ \
    --num_aa 100 \
    --num_steps 500 \
    --out_steps 400 \
    --num_designs 10 \
    --prev_threshold 0.99 \
    --timescale_pos "ve(t)"
```
Here, you need to change the noise schedule (`--timescale_pos`)
to a VE noise schedule and adjust `--prev_threshold` to 0.99 to
prevent over-use of self-conditioning which results in decreased
designability of generated structures. Recommended values for each
model and each setting are described below.

For a VP model with input-dependent variance, you could use:

```bash
python scripts/generate.py \
    --config default_vp_scaled \
    --params params/default_vp_scaled-200k.jax \
    --out_path path_to_outputs/ \
    --num_aa 100 \
    --num_steps 500 \
    --out_steps 400 \
    --num_designs 10 \
    --prev_threshold 0.8 \
    --timescale_pos "cosine(t)" \
    --cloud_std "default(num_aa)"
```
Here, you need to specify the standard deviation of the
noise distribution using `--cloud_std`. The resulting protein
structures will have a standard deviation of atom positions
close to this value. In this example we set `--cloud_std` to
the `default` function, which was implemented to approximate
the distribution of atom position standard deviations for proteins
of a given length `num_aa`.

Further usage examples can be found in the `scripts/` directory,
which contains all generation and benchmarking scripts that were
used for evaluating salad in the corresponding preprint.

Below you can find an in-depth explanation of all parameters that
can be passed to `scripts/generate.py`, as well as their
default and recommended values.

### --config
Model configuration in (default_vp, default_vp_scaled, default_ve_scaled).
* `default_vp` is the configuration for the fixed-variance
  variance-preserving (VP) model with variance 10 Å.
* `multimotif_vp` is a version of `default_vp` with multi-motif
  conditioning for motif scaffolding.
* `default_vp_scaled` is the configuration for input-dependent
  variance VP model. The standard deviation of the noise distribution
  has to be specified by setting `--cloud_std <value in angstroms>`
  (see below).
* `default_ve_scaled` is the configuration for the variance-expanding
  (VE) model. Using this configuration requires setting
  `--timescale_pos "ve(t)"`,
  or another diffusion time scale compatible with VE diffusion.
  In addition, it is highly recommended to set
  `--prev_threshold` to a value between 0.95 and 0.99.

The full list of implemented configurations can be found in salad/modules/config/noise_schedule_benchmark.py.

### --params
Path to a set of parameters compatible with the selected `--config`.

Parameters can be downloaded from Zenodo ([10.5281/zenodo.14711580](https://doi.org/10.5281/zenodo.14711580)), or obtained from the most recent `salad` code release on GitHub. Parameter sets are named after the underlying configuration and the conditions the parameter checkpoint was saved at. E.g. `default_vp-200k.jax` is a checkpoint
for the `default_vp` configuration taken at 200,000 training steps.

Currently, the following checkpoints are available:
| name | config | notes |
|---|---|---|
|default_vp-200k.jax|default_vp| ~|
|default_vp_scaled-200k.jax|default_vp_scaled| ~|
|default_ve_scaled-200k.jax|default_ve_scaled| ~|
|multimotif_vp-200k.jax|multimotif_vp| VP multi-motif scaffolding model|
|default_vp-pdb256-200k.jax|default_vp| trained on proteins in PDB with <= 256 amino acids |
|default_vp-synthete256-200k.jax|default_vp| trained on synthetic data with <= 256 amino acids |

### --out_path
Path to the output directory where the generated PDB files
will be saved. The directory will be created if it does not exist.
Running `salad` multiple times with the same `--out_path` will 
replace the outputs of previous runs.

### --num_aa
Specifies the number of amino acids in each chain that will be
generated by `salad`.

`--num_aa` can be:
* an integer like `--num_aa 100`, to specify a single chain
  of amino acids.
* an integer prefixed with "c", like `--num_aa c100`, to specify
  a cyclic chain of amino acids (cyclised with a peptide bond).
* a list of (prefixed) integers, separated by ":" like
  `--num_aa 100:c50`, to generate a complex with two or
  more chains of amino acids (linear or cyclic).

For instance, generating a 100 amino acid proteins with 8
amino acid cyclic ligands would be specified as:
`--num_aa 100:c8`

### --merge_chains (default: False)
Specifies if different chains should use the same chain_index
during generation and instead be separated by a gap in the
residue_index. This is not relevant for complex generation
unless using a model that has not been exposed to protein
complexes during training (e.g. the pdb256 and synthete256
checkpoints).

### --num_steps (default: 500)
The total number of diffusion steps to subdivide the interval
$[0, 1]$ of diffusion times. E.g. to set up diffusion over
200 total steps, specify `--num_steps 200`.

### --out_steps (default: 400)
The number of diffusion steps after which to write an output
PDB file. Similar to RFdiffusion, we return structures early
by default (at 400 steps of 500 total steps), as the resulting
structure shows very little change after that point.

`--out_steps` can be:
* a single integer step-count, like `--out_steps 400`.
* a comma-separated list of step-counts, like `--out_steps 0,100,200`.
  This writes a PDB file at _each_ of the specified step counts.

### --num_designs (default: 10)
The number of generated PDB files.

### --prev_threshold (default: 0.8)
Diffusion time threshold for using self-conditioning.
A `--prev_threshold` of 0.8 signifies that self-conditioning should
be used only for diffusion times between 0.8 and 1.0, early in the
denoising process.

`--prev_threshold` should be set to:
* **0.8** for `vp` and `vp_scaled` models.
  Models still produce reasonable results for values > 0.6 and
  start losing quality for lower values.
* **0.99** for `ve_scaled` models. Values > 0.95 still produce 
  reasonable results, but lower values result in an overabundance
  of beta sheet structures.

### --timescale_pos (default: "cosine(t)")
Function mapping diffusion time to a noise schedule. By default,
this is set to a cosine noise schedule "cosine(t)" for VP models.
For VE models, it should be set to "ve(t)".

`--timescale_pos` can be:
* any expression depending on diffusion time "t", that can be 
  expressed in raw python + numpy (as np).
* one of the following pre-defined noise schedules:
  * cosine(t): a cosine noise schedule
  * ve(t, sigma_max=80.0, rho=7.0): the EDM noise schedule
    where sigma_max is the starting standard deviation and
    rho is the exponent. Results in the manuscript were
    generated with sigma_max = 80 Å and sigma_max = 100 Å

### --cloud_std (default: "none")
Expression defining the standard deviation of the noise distribution
for input-dependent variance VP models (`default_vp_scaled`).

`--cloud_std` can be:
* "none", when not used with a VP-scaled model.
* any expression dependent on `num_aa`, the number of amino acids
  and evaluating to a positive number. E.g. `"num_aa ** 0.4"`
* use the default standard deviation scale `"default(num_aa)"`. 
  **This is a reasonable default for generating protein structures**
  **with VP-scaled diffusion and was used throughout the preprint.**

### --dssp (default: "none")
Secondary structure specification. Overrides `--num_aa`.
This is a string of letters "L" (for Loops), "H" (for Helices),
"E" (for shEets) and "_" (for unspecified secondary structure).
E.g.:
```
___HHHHHHHH___EEEEEE___EEEEEE___EEEEE___HHHHHHHH____
```
Alternatively, you may set `--dssp random`. This randomly samples
a secondary structure string of the shape specified in `--num_aa`
and can be used to increase the diversity of generated structures.

By default, no secondary structure constraints are applied.

**NOTE: currently non-random dssp cannot be mixed with other**
**options, such as cyclicity constraints or symmetry constraints.**
**We plan to address this in a future release.**

### --template (default: "none")
Specifies an ad-hoc template PDB file for single-motif scaffolding.
Template is a comma-separated tuple of a path to a motif PDB file
and a constraint string:
```
path_to/template.pdb,XXXXXXXXXXFFFFFFXXXXFFFFXXXXXFFFFXXXXXXXXX
```
The constraint string specifies the structure of which amino acids
is specified in the template PDB file. The Nth position marked with
"F" in the constraint string corresponds to the Nth amino acid in the
template PDB file. The constraint string **must** contain exactly as
many "F"s as the number of amino acids in `template.pdb`. "X"s specify
amino acids that can be freely designed.

Please make sure to insert "X"s only at positions where the template
PDB file has gaps in the amino acid chain, otherwise you will generate
nonsensical structures.

**NOTE: as the constraint string specifies the number and sequence**
**positions of freely designable amino acids, this overrides**
**--num_aa and is currently incompatible with --dssp specifications.**
**We plan to address this in a future release.**

**For multi-motif scaffolding, see [below](#multi-motif-scaffolding).**

### --fast

Compiles the denoiser and captures it in a CUDA graph. Every sampling step then
replays one graph instead of dispatching ~700 kernels from Python, which is
about 20x cheaper per step. Costs a one-time compile of roughly five minutes.

GPU only. One fixed length per process - compiling several lengths in the same
process degrades the CUDA graphs. Incompatible with `--sym`.

Point `TORCHINDUCTOR_CACHE_DIR` at persistent shared storage, or you pay the
compile again on every machine and every job; `/tmp` is usually node-local and
discarded. Budget ~3,800 files and ~280 MB per compiled length.

### --preset (default: "exact")

Only meaningful with `--fast`. Chooses how much numerical deviation from the
JAX original is acceptable.

| preset | numerics | L=100 step |
|---|---|---|
| `exact` | matches JAX to 0.00002 A on all four verified checkpoints | 2.53 ms |
| `fast` | TF32, half the sampling schedule, two reassociations | 2.00 ms |
| `deterministic` | as `exact`, plus reproducible reductions | ~2.7 ms |

`fast`'s deviations are validated at 160 designs per arm against `exact` on bond
length, radius of gyration, secondary-structure composition, clash count and
divergence rate. Most of its per-design win is the halved schedule, not the
faster step.

`deterministic` replaces `scatter_add` (whose atomic accumulation order varies
run to run) with a one-hot matmul, which is what makes `--seed` actually pin a
CUDA run. It is **not** parity-neutral: it drifts about 3.2x further from JAX
than `exact` and flips 7 residues in 64 on the sampling fixture. Use it to
reproduce or debug a run, not as a production default.

### --group_size (default: off)

Advances N independent designs through **one** CUDA graph. Needs `--fast`.

This is not a batch axis and not a shared trajectory. Each design keeps its own
buffers, its own noise and its own seed-derived RNG stream, and computes exactly
what it would have computed alone - verified bit-identical, 8/8 designs. The
only thing that changes is that the GPU stops idling between one design's
serialised kernel launches.

`auto` measures the right group size on the device you are actually running on:
it captures at 1, 2, 3, 4, ... and stops when a rung stops paying or free VRAM
says stop. Prefer it to a pinned integer, which will be wrong on a different
card. `--group_size 1` is the grouped machinery with a group of one and
reproduces the ungrouped path.

The win comes from idle SMs, so it is largest on a big GPU with a short protein
and shrinks toward 1x on a small GPU or a long one. Throughput saturates around
16 designs in flight.

### --batch_size (default: 1)

Packs B designs along the flat residue axis so each kernel does B designs' work
in one launch. Composable with `--group_size`.

Unlike `--group_size` this is **not** semantically free: packed designs are not
bit-identical to the same seed generated alone, because a larger `M` changes the
GEMM tiling and accumulation order. Measured at 160 designs per arm the
structural distributions are indistinguishable and the amino-acid composition
shifts by 1-2 percentage points.

Worth +42 % at L=50 and +16 % at L=100 on top of `--group_size`; nothing at
L=400 and it *loses* at L=1000, because the pair tensors grow as B^2 while the
designs grow as B. Rule of thumb: `B ~ 600/L`, rounded to a power of two.

### --sequence_only

Write FASTA to `{out_path}/sequences.fa` instead of PDB files.

### --order_agnostic

The network never sees residue order (`resi=-1`, `chain=0`). The output PDB is
re-threaded by following C-N backbone contacts instead of network indexing.

### --depth_adjust (default: "none")

An early-exit ablation: an integer N not greater than the checkpoint's native
trunk depth (6 for `default_vp`) builds and loads only the first N blocks.
Values above the native depth are unsupported - there are no extra trained
weights to load.

### --seed / --jax_seed

`--seed` seeds the noise draw. **On CUDA this does not by itself pin the
output**: segment reductions use `scatter_add`, whose atomic ordering varies,
the neighbour argsort turns that into a discrete tie flip, and the sampling
steps amplify it. Two identical commands differ in about 48 % of residues. Add
`--preset deterministic` to pin a CUDA run. CPU is already deterministic.

`--jax_seed` is an alias, for command lines copied from the JAX CLI. It does
**not** give identical output across backends - the two PRNGs are unrelated.

### --device, --compile, --timing_json

`--device` defaults to `cuda` when available. `--compile` passes a
`torch.compile` mode for the eager path and is independent of `--fast`, which
manages its own compilation. `--timing_json` writes a timing breakdown.


## Multi-motif scaffolding
To scaffold one or more protein motifs, you can use the scripts
`scripts/generate.py` and `scripts/generate.py`.
The former implements (multi-)motif scaffolding for models which
were trained with multi-motif conditioning, while the latter uses
structure-editing to scaffold motifs without explicit motif conditioning.

Both scripts take the same denoising-process arguments as above
(--num_designs, --num_steps,
--out_steps, --prev_threshold, --config, --params, --timescale_pos).
Instead of providing `--num_aa`, both scripts accept a `--template`
argument, e.g. `--template motif_structure.pdb`.

Here, `motif_structure.pdb` must be a motif-annotated PDB file as
described in [the Genie2 repository](https://github.com/aqlaboratory/genie2?tab=readme-ov-file#format-of-motif-scaffolding-problem-definition-file).
The scripts then generate designs according to the specification in that `motif_structure.pdb` file. For example:
```
python scripts/generate.py \
    --config multimotif_vp \
    --params params/multimotif_vp-200k.jax \
    --out_path path_to_outputs/ \
    --num_steps 500 --out_steps 400 --prev_threshold 0.8 \
    --num_designs 1000 --timescale_pos "cosine(t)" \
    --template motif_structure.pdb
```
Using `scripts/generate.py` requires a multi-motif conditioned
checkpoint, such as `multimotif_vp-200k.jax`. To use _any_ salad
model, you can instead use `scripts/generate.py`:
```bash
config=default_vp
python scripts/generate.py \
    --config $config \
    --params params/${config}-200k.jax \
    --out_path path_to_outputs/ \
    --num_steps 500 --out_steps 400 --prev_threshold 0.8 \
    --num_designs 1000 --timescale_pos "cosine(t)" \
    --template motif_structure.pdb
```
Usage of both scripts is the same.

## Symmetric / repeat protein generation
To generate rotation or superhelical (screw) symmetric repeat proteins,
you can use the script `scripts/generate.py`.
Currently, this script only supports cyclic or screw-axis symmetries.
By specifying an angle of rotation, distance from center and translation
along the symmetry axis, this allows generating superhelical repeat
proteins with a fully specified geometry.

This script takes the usual denoising process settings (--num_designs, --num_steps, --out_steps, --prev_threshold, --config, --params, --timescale_pos),
as well as `--num_aa` compatible with `scripts/generate.py`.
In addition, it requires inputs for `--screw_radius`, which specifies
the repeat unit distance from the symmetry axis in angstroms (center-of-mass radius
of the resulting superhelix); `--screw_angle`, which specifies the
rotation angle around the symmetry axis for successive repeat units;
`--screw_translation`, which specifies the translation along the symmetry
axis in angstroms between two successive repeat units.

For example, generating C3-symmetric repeat proteins works as follows:
```bash
monomer_size=50
repeat_count=3
total_count=$(($monomer_size * $repeat_count))
python scripts/generate.py \
    --config default_ve_scaled \
    --params params/default_ve_scaled \
    --num_aa $total_count \
    --out_path path_to_output/ \
    --num_steps 500 --out_steps "400" \
    --prev_threshold 0.98 --num_designs 20 --timescale_pos "ve(t)" \ 
    --screw_radius 10.0 --screw_angle 120.0 \
    --screw_translation 0.0 --replicate_count 3 \
    --mode "screw" --mix_output couple \
    --sym_noise True
```
To generate a screw-symmetric structure instead, it suffices to set
`--screw_translation` to a non-zero value.

While this script can be used with any model configuration, generating
screw-symmetric proteins works best with VE models.

A detailed explanation of all arguments to `scripts/generate.py` is given below.

### --num_aa
This works as in `scripts/generate.py`. However, care needs
to be taken that the chain lengths specified are compatible with the
applied symmetry operation. E.g. for --replicate_count 3, --num_aa needs
to be divisible by 3, or consist of 3 chains of equal length, e.g.:
```
--num_aa 150
--num_aa 50:50:50
```

### --replicate_count
Number of copies of a repeat. E.g. for a C3-symmetric repeat protein,
one should use --replicate_count 3.

### --screw_translation
Translation along the central axis of rotation between two successive
repeat units. If this is 0.0, this will result in a repeat protein
or assembly with cyclic symmetry. Otherwise it will result in a 
superhelical (screw) symmetry.

### --screw_radius
Radius in angstroms at which repeats should be placed from the central
axis of rotation. Allows to control the radius of a cyclic or 
screw-symmetric repeat protein or assembly.

### --screw_angle
Angle of rotation between two successive repeat units around the central
axis of rotation. For designs with --screw_translation 0.0 this needs to
be less than or equal to 360 / --replicate_count, as it will result in
clashing structures otherwise.

# Things that will bite you

All reproduced deliberately, because the point is to match the released model
rather than to improve it.

**Padding is not safe, and batch position changes the answer.** A `-1` "no
neighbour" sentinel un-masks the last residue, which is then selected twice and
double-counted in attention. The same protein at a different offset in the flat
residue axis comes out up to ~5 A different. This is deterministic, not sampling
noise, and it is bit-identical in JAX - the port is faithful and upstream is
wrong. **Concatenate examples along the residue axis; never pad.**

**Residue numbering starts at 0, and each new chain starts 50 past the previous
residue.** So `--num_aa 100:100` gives `0..99` then `149..248`, not `150..249`.
Both the 0-based start and the off-by-one reach the output PDB.

**`--seed` does not pin a CUDA run.** Segment reductions use `scatter_add`,
whose atomic accumulation order varies; the neighbour argsort turns that into a
discrete tie flip and 500 steps amplify it. Two identical commands differ in
~48% of residues. Use `--preset deterministic` if you need reproducibility. CPU
is already deterministic.

**Sampling trajectories are chaotic.** A single forward matches JAX to 2e-7, but
500 steps of feeding the output back in amplifies fp32 reduction-order
differences. Compare trajectories as a *ratio* to an identical-code rerun, never
against an absolute tolerance.

**`jax.nn.gelu` is the tanh approximation**, while `torch.nn.functional.gelu`
defaults to exact erf. Getting this wrong costs 0.31 A per denoiser call. It is
handled; do not "simplify" `salad_torch/modules/basic.py`.

**DDP needs `find_unused_parameters=True`.** 202 parameters never receive
gradient: Haiku's `Linear` allocates a bias even when `bias=False`, so those
tensors exist in the checkpoint and are never added.

**`--fast` fixes the length.** One compiled graph per shape; generating several
lengths in one process degrades the CUDA graphs badly. Run one length per
process, and point `TORCHINDUCTOR_CACHE_DIR` at persistent shared storage or you
pay the ~5 minute compile on every job (see `docs/USAGE.md`).

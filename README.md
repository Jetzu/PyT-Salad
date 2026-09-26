# salad-torch

A PyTorch port of [**salad**](https://github.com/mjendrusch/salad) (sparse
all-atom denoising), the protein structure generation model from Michael
Jendrusch / EMBL. The original is JAX + dm-haiku; this reimplements it in
PyTorch and loads the released weights unchanged.

> **salad is not FlexCraft.** [`flexcraft`](https://github.com/mjendrusch/flexcraft)
> is a separate repo by the same author that *uses* salad for backbone
> generation and adds ProteinMPNN plus AF2/Boltz/Protenix filtering. salad is
> the generative model, and it is what this ports.

Paper: *Efficient protein structure generation with sparse denoising models*,
bioRxiv 2025.01.31.635780.

## Status

| | |
|---|---|
| geometry layer | exact parity vs JAX (38/38 functions) |
| 6-layer trunk, released weights | 0.0003 A max coordinate deviation |
| **whole denoiser, released weights** | **0.00002 A max, identical predicted sequence** |
| all four released checkpoints | load with `strict=True` |
| sampling loop | generates proteins; verified against the chaos noise floor |
| training losses | all 12 terms implemented, gradients match JAX (cosine 0.99999863) |
| training loop | complete and verified; batches by bin-packing, like upstream |
| real-PDB datasource | implemented, **untested** - no preprocessed `.npz` tree on the dev machine |
| **performance** | **1.09-4.7x FASTER than JAX** (see below) |

## Performance

Measured on an H100 at L=100, against the JAX original on the same node.

| | per step | per design | vs JAX |
|---|---|---|---|
| JAX salad | 4.73 ms | 1.56 s | — |
| `--fast --preset exact` | 3.18 ms | 1.34 s | 1.16x |
| `--fast --preset fast` | 3.01 ms | 0.60 s | 2.6x |
| `--fast --preset fast --group_size auto` | 3.01 ms | **0.334 s** | **4.7x** |

`exact` holds the 0.00002 A parity floor. `fast` adds TF32 and a halved
sampling schedule, validated at 160 designs per arm as statistically
indistinguishable from `exact` on bond geometry, radius of gyration, secondary
structure and clash rate. `--group_size` advances N independent designs through
one CUDA graph and is **bit-identical** to generating them one at a time.

Both presets still sit >12x above this model's measured theoretical floor
(0.258 ms/step): it is kernel-launch-bound, not arithmetic- or bandwidth-bound.
Details and the remaining gap list are in `notes/04_PARITY_AND_PERF.md`.

**[Full flag reference: `docs/USAGE.md`](docs/USAGE.md)**

## Layout

```
salad_torch/
  config.py            the four configs with released weights
  geometry/            frames, neighbour selection, pair features, segment ops, sidechains
  modules/             base layers, sparse IPA, diffusion blocks, decoders, the model
  diffusion/           noise schedules, forward noising, preconditioning, sampling
  losses/              the training loss terms
  io/                  PDB output, residue constants
  convert/             Haiku checkpoint -> PyTorch state dict
tests/                 parity suites (run them, they are the point)
notes/                 architecture write-ups and measured results
notes/specs/           the implementation specs the port was written from
```

## Quick start

Generate eight 100-residue designs:

```bash
python scripts/generate.py \
    --config default_vp --params params/default_vp-200k.jax \
    --num_aa 100 --num_designs 8 \
    --fast --preset fast --group_size auto \
    --out_path outputs/my_designs
```

Every flag — conditioning, symmetry, schedules, reproducibility — is documented
in **[`docs/USAGE.md`](docs/USAGE.md)**.

From Python:

```python
import torch
from salad_torch.config import get_config
from salad_torch.modules.model import StructureDiffusionInference
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz

config = get_config("default_vp")
model = StructureDiffusionInference(config).eval()
load_model(model.predict, load_param_npz("default_vp-200k.npz"), config, strict=True)
```

Checkpoints come from Zenodo record
[14711580](https://doi.org/10.5281/zenodo.14711580). They are pickles of JAX
arrays; `convert.haiku_to_torch.load_param_jax` reads them without JAX
installed, or convert once to `.npz`.

## Things that will bite you

These are all reproduced deliberately, because the point is to match the
released model rather than to improve it.

**Padding is not safe, and batch position changes the answer.** A `-1` "no
neighbour" sentinel un-masks the last residue, which then gets selected twice
and double-counted in attention. The same protein at a different offset in the
flat residue axis, or sitting next to masked padding, comes out up to ~5 A
different. This is deterministic, not sampling noise — verified by re-running
with an identical random field. It is bit-identical in JAX, so the port is
faithful and upstream is wrong. **Concatenate examples along the residue axis;
never pad.**

**Sampling trajectories are chaotic.** A single forward pass matches JAX to
2e-7, but 81 steps of feeding the output back in amplifies fp32 reduction-order
differences. Compare trajectories as a *ratio* to an identical-code rerun, never
against an absolute tolerance.

**DDP needs `find_unused_parameters=True`.** 202 parameters never receive
gradient: Haiku's `Linear` allocates a bias even when `bias=False`, so those
tensors exist in the checkpoint and are never added.

**`jax.nn.gelu` is the tanh approximation**, while `torch.nn.functional.gelu`
defaults to exact erf. Getting this wrong costs 0.31 A per denoiser call. It is
already handled; don't "simplify" `salad_torch/modules/basic.py`.

## Licence

Code Apache 2.0 (EMBL), model parameters CC-BY 4.0 — both permit a derivative
port with attribution. This port carries the same terms.

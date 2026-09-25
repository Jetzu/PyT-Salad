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
| training losses | all 12 terms implemented, gradients clean |
| data pipeline + training loop | in progress |
| performance | 5-9x slower than JAX; see `notes/04_PARITY_AND_PERF.md` |

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

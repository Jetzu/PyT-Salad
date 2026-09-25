# SALAD → PyTorch port: workspace setup

Created 2026-09-25.

## Naming clarification

`salad` and `flexcraft` are **two different repositories** by the same author
(Michael Jendrusch, EMBL — Jendrusch/Korbel lab):

- **salad** (`github.com/mjendrusch/salad`) — "sparse all-atom denoising", the
  generative model itself. JAX + dm-haiku. *This is the port target.*
- **flexcraft** (`github.com/mjendrusch/flexcraft`) — a pipeline library that
  *uses* salad for backbone generation and adds ProteinMPNN sequence design plus
  AF2 / Boltz / Protenix structure prediction and filtering. Mostly JAX too.
- **flexloop** (`github.com/mjendrusch/flexloop`) — the author's training-loop
  library; a salad *training* dependency only.

Paper: "Efficient protein structure generation with sparse denoising models",
bioRxiv 2025.01.31.635780.

## Licensing

- Code: Apache 2.0 (EMBL).
- Model parameters: CC-BY 4.0.

Both permit a derivative PyTorch port with attribution.

## Layout

```
/home/hpc/v181bb/v181bb10/SALAD_torch/
  upstream/salad/       reference clone @ 8348117  (READ-ONLY - never edit)
  upstream/flexcraft/   pipeline library
  upstream/flexloop/    training loop library
  params/      -> /hnvme/workspace/v181bb10-salad/params      (12 salad ckpts)
  ae_params/   -> /hnvme/workspace/v181bb10-salad/ae_params   (8 autoencoder ckpts)
  salad.sif    -> /hnvme/workspace/v181bb10-salad/salad.sif   (12 GB JAX reference)
  notes/
```

Bulk data lives on the `/hnvme` workspace `v181bb10-salad` (allocated 2026-09-25,
90 days, 3 extensions remaining — `ws_extend salad 90` before it lapses).
Rationale: `/home/hpc` is at 84 G/100 G and 446 k/500 k inodes; `/hnvme` has free
block space and the payload is a few large files. Compute nodes see both
`/hnvme` and `/home/hpc`.

## JAX reference environment

Built from the repo's own recipe (`apptainer/salad.def`), in local `/tmp` to
avoid the inode quota, then staged to `/hnvme`:

```bash
apptainer build salad.sif apptainer/salad.def   # ~4 min, 12 GB
```

Verified inside: Python 3.10.21, jax 0.6.2, haiku 0.0.17, `salad` importable.

**Invocation gotcha:** the container's conda env is activated by the *runscript*,
so `apptainer exec ... python3.10` fails with "executable not found". Use
`apptainer run`, which forwards its arguments to `python3.10`:

```bash
apptainer run --nv --bind /hnvme:/hnvme salad.sif my_script.py [args]
```

No GPU on the login node — parity runs need a Slurm job (`h100` / `h200`).

## Checkpoints

Downloaded from Zenodo record 14711580 (`salad_params.tar.gz`, `ae_params.tar.gz`).

The `.jax` files are **Python pickles containing JAX arrays**, so plain
`pickle.load` fails with `ModuleNotFoundError: jax` outside the container.
Converted once, inside the container, to `.npz` + a `.txt` manifest of every
parameter path / shape / dtype (`/tmp/dump_params.py`):

| checkpoint            | modules | leaf arrays | params  |
|-----------------------|---------|-------------|---------|
| `default_vp`          | 132     | 260         | 11.54 M |
| `default_vp_scaled`   | 132     | 260         | 11.54 M |
| `default_ve_scaled`   | 136     | 268         | 11.85 M |
| `multimotif_vp`       | 138     | 272         | 11.86 M |

`default_vp` and `default_vp_scaled` are architecturally identical — they differ
only in coordinate scaling config, so one PyTorch module serves both.

After conversion the PyTorch side needs **no JAX dependency at all**.

## Parameter tree shape (default_vp)

Top-level groups, by module count:

| prefix (under `structure_diffusion/`) | modules |
|---------------------------------------|---------|
| `diffusion/diffusion_stack`           | 96      |
| `~encode/encoder`                     | 59      |
| `diffusion/aa_decoder`                | 57      |
| `diffusion/distogram_decoder`         | 20      |
| `~prepare_condition/condition`        | 12      |
| `diffusion/~prepare_features`         | 10      |
| `diffusion/mlp`, `diffusion/linear`   | 6       |

A `~` prefix is Haiku's scope marker for a method other than `__call__`.

**Main porting gotcha:** scopes containing `__layer_stack_no_state` come from
`hk.LayerStack`, which stacks every layer's weights along a *leading* dimension.
Observed leading dims: 6 (main diffusion stack), 3 (aa_decoder), 2. So e.g.
`(6, 128, 512)` is six separate `128 -> 512` layers, and the loader must slice
`[i]` per layer rather than treating it as one weight.

Width is 128 with a 512 FFN; head dims 32/64.

## Environments available for the PyTorch side

- `miniforge3/envs/triad` — torch 2.5.1+cu121
- `conda_envs/esmfold2` — torch 2.13.0+cu130

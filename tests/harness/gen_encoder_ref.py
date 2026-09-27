"""Dump the UPSTREAM `Encoder` (real `default_vp-200k` weights) on a few
synthetic inputs, for `tests/test_encoder_parity.py`.

The Encoder is config-invariant across all 4 released checkpoints (spec-05:
none of them touch `augment_size`/`encoder_depth`/`resi_dual`/`equivariance`),
so `default_vp`'s weights are representative and this only needs generating
once.

Run inside the container:
    apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
        tests/harness/gen_encoder_ref.py

Originally written as the project notes
symmetry-encoder/r05_encoder_parity_jax.py` (a same-family red-team pass that
found the D1 crash this test exists to catch a regression of - see
`salad_torch/modules/encoder.py`'s `unsqueeze_batch` fix); promoted here
per that pass's own recommendation once a real test harness location existed.
"""
import pickle

import numpy as np
import jax
import jax.numpy as jnp

# XLA:GPU lowers float32 `dot` to TF32 by default, which makes a GPU-generated
# reference a ~1e-3-relative one - see notes/04_PARITY_AND_PERF.md section 5.
# Bit-exact no-op on CPU.
jax.config.update("jax_default_matmul_precision", "highest")

import haiku as hk

import salad.modules.noise_schedule_benchmark as NSB
from salad.modules.config import noise_schedule_benchmark as configs

cfg = configs.default_vp
cfg.eval = True


class SD(hk.Module):
    """Minimal wrapper opening the `structure_diffusion` scope the checkpoint
    uses, so `Encoder`'s params resolve to `structure_diffusion/~encode/
    encoder/...` - matching `StructureDiffusion.encode`'s own call site."""

    def __init__(self, c):
        super().__init__(name="structure_diffusion")
        self.config = c

    def encode(self, data):
        return NSB.Encoder(self.config)(data)


def fwd(data):
    return SD(cfg).encode(data)


def build(rng, n, chains=1):
    t = np.arange(n)[:, None]
    bb = np.concatenate([2.3 * np.cos(t * 1.7), 2.3 * np.sin(t * 1.7), 1.5 * t], -1)
    pos = (bb[:, None, :] + rng.normal(0, 0.8, (n, 5, 3))).astype(np.float32)
    ch = (np.arange(n) // max(1, n // chains)).clip(max=chains - 1).astype(np.int32)
    return dict(pos=pos, residue_index=np.arange(n, dtype=np.int32),
               chain_index=ch, batch_index=np.zeros(n, np.int32),
               mask=np.ones(n, bool))


def main(param_path="params/default_vp-200k.jax",
        out_path="fixtures/encoder_default_vp.npz"):
    f = hk.transform(fwd)
    with open(param_path, "rb") as fh:
        params = pickle.load(fh)

    rng = np.random.default_rng(3)
    out = {}
    for tag, n, ch in (("L64c1", 64, 1), ("L96c3", 96, 3), ("L16c1", 16, 1)):
        d = build(rng, n, ch)
        local, pos = f.apply(params, jax.random.PRNGKey(0),
                             {k: jnp.asarray(v) for k, v in d.items()})
        for k, v in d.items():
            out[f"{tag}|in_{k}"] = np.asarray(v)
        out[f"{tag}|out_local"] = np.asarray(local)
        out[f"{tag}|out_pos"] = np.asarray(pos)
        print(tag, "local", np.asarray(local).shape, "pos", np.asarray(pos).shape,
             "|pos|max", float(np.abs(np.asarray(pos)).max()))
    np.savez(out_path, **out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

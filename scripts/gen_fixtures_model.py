"""Dump a full JAX forward pass of the salad denoiser, with the RNG pinned.

The denoiser is stochastic even at inference: every DiffusionBlock samples
random neighbours via Gumbel-top-k from the threefry stream. To make the
PyTorch port checkable we *record* every neighbour array the JAX model draws,
so the port can replay them and be compared deterministically.

Run inside the container (eagerly - no jit, or the Python-side recording of
traced values would not work):

    apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
        scripts/gen_fixtures_model.py default_vp \
        /hnvme/workspace/v181bb10-salad/params/default_vp-200k.jax \
        fixtures/model_default_vp.npz
"""
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

# ---------------------------------------------------------------------------
# PRECISION PIN - do not remove.
#
# XLA:GPU lowers float32 `dot` to TF32 (10 mantissa bits) by default. A fixture
# generated on a GPU is therefore NOT an fp32 reference: measured on
# default_vp, an H100 run deviates from the identical CPU run by 5.98e-03 A in
# `out_pos` and 1.22e-02 A in the block-0 trajectory - 300x the fp32 parity
# floor - while `aatype` still matches exactly, so nothing looks wrong.
# Two fixtures were generated that way and spent a day being debugged as a port
# bug. `highest` collapses the same GPU-vs-CPU gap to 7.6e-06 A.
jax.config.update("jax_default_matmul_precision", "highest")

import salad.modules.noise_schedule_benchmark as NSB
import salad.modules.utils.geometry as G
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import noise_schedule_benchmark as configs

FIXED_U = None  # (N, N) float32, stands in for jax.random.uniform


def install_fixed_uniform(n, seed=99):
    """Replace the Gumbel-top-k RNG draw with a fixed uniform field.

    The neighbour draw is a data-routing decision, not learned math, and it
    runs inside `hk.scan`+`hk.remat` where Python-side recording only sees
    tracers. Substituting a constant field makes the whole forward pass
    deterministic and reproducible on the PyTorch side, which is what we need
    to verify the network itself. The Gumbel sampler's *distribution* is
    checked separately.
    """
    global FIXED_U
    rng = np.random.default_rng(seed)
    FIXED_U = rng.uniform(1e-6, 1 - 1e-6, size=(n, n)).astype(np.float32)
    u = jnp.asarray(FIXED_U)

    def patched(count):
        def inner(pos, item, mask, neighbours=None):
            distance = None
            if isinstance(pos, jnp.ndarray):
                assert pos.ndim == 2
                if pos.shape[0] == pos.shape[1]:
                    distance = pos
                else:
                    pos = Vec3Array.from_array(pos)
            if distance is None:
                distance = (pos[:, None] - pos[None, :]).norm()
            same_item = item[:, None] == item[None, :]
            weight = -3 * jnp.log(distance + 1e-6)
            gumbel = jnp.log(-jnp.log(u))          # <- fixed, not drawn
            weight = weight - gumbel
            distance = jnp.where(same_item, -weight, jnp.inf)
            pair_mask = mask[:, None] * mask[None, :] * same_item
            return G.get_neighbours(count)(distance, pair_mask, neighbours)
        return inner

    G.get_random_neighbours = patched
    NSB.get_random_neighbours = patched


def build_motif_condition(pos, rng, block=20):
    """Real dmap / dmap_mask conditioning for `multi_motif`.

    Mirrors the shape of `StructureDiffusion.prepare_condition`'s multi-motif
    branch (NSB ~line 284-330): split the chain into contiguous segments and
    assign each segment to one of two groups; only within-group pairs are
    "known". Unlike that training-time code (random segment lengths, random
    group assignment, random drop-out, all via `hk.next_rng_key`), this is a
    plain deterministic construction - it only has to produce a *real*,
    non-degenerate dmap_mask, not replay the stochastic reveal (that is
    task 4's job, in `training/loop.py`).

    `block` is picked so each group ends up bigger than the motif attention's
    top-32 selection (`extract_motif_neighbours(32)`): with `length=80` and
    `block=20` each group has 40 residues (39 same-group candidates per row),
    so the neighbour draw must actually rank by distance instead of admitting
    every candidate through the mask - an all-False or trivially-small-group
    mask would not exercise that ranking at all.
    """
    n = pos.shape[0]
    cb = pos[:, 4]  # atom index 4 = CB by convention (Encoder, prepare_condition)
    dmap = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1).astype(np.float32)
    group = ((np.arange(n) // block) % 2).astype(np.int32)
    dmap_mask = group[:, None] == group[None, :]
    np.fill_diagonal(dmap_mask, False)
    return dmap, dmap_mask


def build_data(length, augment_size, rng, multi_motif=False):
    """A single-chain unconditional system of `length` residues."""
    n = length
    a = 5 + augment_size
    resi = np.arange(n, dtype=np.int32)
    chain = np.zeros(n, np.int32)
    batch = np.zeros(n, np.int32)
    mask = np.ones(n, bool)
    # a plausible starting structure rather than pure noise, so frames are sane
    t = np.arange(n)[:, None]
    bb = np.concatenate([2.3 * np.cos(t * 1.7), 2.3 * np.sin(t * 1.7), 1.5 * t], -1)
    pos = (bb[:, None, :] + rng.normal(0, 1.2, (n, a, 3))).astype(np.float32)
    data = dict(
        pos=pos,
        aa_gt=np.full(n, 20, np.int32),
        seq=np.full(n, 20, np.int32),
        residue_index=resi,
        chain_index=chain,
        batch_index=batch,
        mask=mask,
        cyclic_mask=np.zeros(n, bool),
        t_pos=np.full(n, 0.5, np.float32),
        t_seq=np.full(n, 1.0, np.float32),
    )
    if multi_motif:
        dmap, dmap_mask = build_motif_condition(pos, rng)
        data["dmap"] = dmap
        data["dmap_mask"] = dmap_mask
    return data


def main(config_name, param_path, out_path, length=48):
    rng = np.random.default_rng(1234)
    config = getattr(configs, config_name)
    config.eval = True

    with open(param_path, "rb") as f:
        params = pickle.load(f)

    data = build_data(length, config.augment_size, rng, multi_motif=bool(config.multi_motif))
    prev = dict(
        pos=np.zeros_like(data["pos"]),
        local=np.zeros((length, config.local_size), np.float32),
    )

    out = {f"in_{k}": v for k, v in data.items()}
    out["in_prev_pos"] = prev["pos"]
    out["in_prev_local"] = prev["local"]

    # --- 1. run the noising step alone, so pos_noised becomes a fixed input ---
    def noise_fn(d):
        return NSB.StructureDiffusionNoise(config)(d)

    noise_t = hk.transform(noise_fn)
    key = jax.random.PRNGKey(0)
    jdata = jax.tree_util.tree_map(jnp.asarray, data)
    noised = noise_t.apply({}, key, jdata)
    for k, v in noised.items():
        out[f"noised_{k}"] = np.asarray(v)

    # --- 2. run the predictor with those fixed noised inputs ------------------
    install_fixed_uniform(length)

    fixed = dict(jdata)
    fixed.update(noised)

    def predict_fn(d, p):
        return NSB.StructureDiffusionPredict(config)(d, p)

    predict_t = hk.transform(predict_fn)
    jprev = jax.tree_util.tree_map(jnp.asarray, prev)
    result, new_prev = predict_t.apply(params, jax.random.PRNGKey(7), fixed, jprev)

    for k, v in result.items():
        try:
            out[f"out_{k}"] = np.asarray(v)
        except Exception:
            pass
    out["out_prev_pos"] = np.asarray(new_prev["pos"])
    out["out_prev_local"] = np.asarray(new_prev["local"])

    # --- 3. the pinned uniform field the port must replay --------------------
    out["fixed_uniform"] = FIXED_U

    # --- 4. provenance: what backend/precision produced these numbers --------
    # tests/test_model_parity.py refuses a fixture that is not fp32-exact.
    out["jax_platform"] = np.asarray(
        jax.devices()[0].platform, dtype=object).astype("U")
    out["jax_matmul_precision"] = np.asarray(
        str(jax.config.jax_default_matmul_precision), dtype=object).astype("U")

    np.savez(out_path, **out)
    print(f"wrote {out_path}: {len(out)} arrays  (length={length}, config={config_name})")
    for k in sorted(out):
        v = out[k]
        print(f"   {k:34s} {getattr(v, 'shape', ())} {getattr(v, 'dtype', type(v))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 48)

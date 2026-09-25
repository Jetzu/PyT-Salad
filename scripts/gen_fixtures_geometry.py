"""Dump JAX reference inputs/outputs for salad's pure geometry functions.

Run INSIDE the salad container:

    apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
        scripts/gen_fixtures_geometry.py fixtures/geometry.npz

Everything captured here is a pure function of its inputs (no haiku, no RNG),
so the PyTorch port can be checked against it exactly.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

from salad.aflib.model.geometry import Vec3Array, Rot3Array, Rigid3Array
import salad.modules.utils.geometry as G
import salad.modules.geometric.geometric as GM

OUT = {}


FAILED = []


def put(name, value):
    """Store an array (or a Vec3Array/Rot3Array/Rigid3Array) as numpy."""
    if hasattr(value, "to_array"):
        value = value.to_array()
    OUT[name] = np.asarray(value)


def cap(name, fn):
    """Capture fn()'s result under `name`, recording rather than raising on error."""
    try:
        put(name, fn())
    except Exception as exc:  # noqa: BLE001 - we want the whole sweep regardless
        FAILED.append(f"{name}: {type(exc).__name__}: {exc}")


def main(out_path):
    rng = np.random.default_rng(0)

    # ---- a small, realistic two-chain system -------------------------------
    N, A = 37, 20
    # chain 0: residues 0..20, chain 1: residues 0..15 (indices offset by 50)
    chain = np.concatenate([np.zeros(21, np.int32), np.ones(16, np.int32)])
    resi = np.concatenate([np.arange(21, dtype=np.int32),
                           np.arange(16, dtype=np.int32) + 71]).astype(np.int32)
    batch = np.zeros(N, np.int32)
    mask = np.ones(N, bool)
    mask[-3:] = False                      # exercise masking
    # positions: a coarse helix so frames are well conditioned, + noise
    t = np.arange(N)[:, None]
    backbone = np.concatenate(
        [2.3 * np.cos(t * 1.7), 2.3 * np.sin(t * 1.7), 1.5 * t], axis=-1)
    pos = (backbone[:, None, :] + rng.normal(0, 1.3, (N, A, 3))).astype(np.float32)

    put("in_pos", pos)

    put("in_resi", resi)

    put("in_chain", chain)

    put("in_batch", batch)

    put("in_mask", mask)


    jpos = jnp.asarray(pos)
    vpos = Vec3Array.from_array(jpos)

    # ---- frames ------------------------------------------------------------
    frames, local_pos = G.extract_aa_frames(vpos)
    cap("extract_aa_frames.frames", lambda: frames.to_array())      # (N,3,4)
    cap("extract_aa_frames.local_pos", lambda: local_pos.to_array())  # (N,A,3)

    ref = G.make_transform_from_reference(
        vpos[:, 0], vpos[:, 1], vpos[:, 2])
    cap("make_transform_from_reference", lambda: ref.to_array())

    cap("compute_pseudo_cb", lambda: G.compute_pseudo_cb(jpos))
    cap("positions_to_ncacocb", lambda: np.asarray(G.positions_to_ncacocb(jpos[:, :5])))

    # Rot3Array.from_two_vectors, used by every frame construction
    v0 = Vec3Array.from_array(jnp.asarray(rng.normal(size=(11, 3)).astype(np.float32)))
    v1 = Vec3Array.from_array(jnp.asarray(rng.normal(size=(11, 3)).astype(np.float32)))
    put("in_v0", v0.to_array())

    put("in_v1", v1.to_array())

    cap("from_two_vectors", lambda: Rot3Array.from_two_vectors(v0, v1).to_array())

    # apply / apply_inverse round trip
    rot = Rot3Array.from_two_vectors(v0, v1)
    pts = Vec3Array.from_array(jnp.asarray(rng.normal(size=(11, 3)).astype(np.float32)))
    put("in_pts", pts.to_array())

    cap("rot_apply_to_point", lambda: rot.apply_to_point(pts).to_array())
    cap("rot_apply_inverse_to_point", lambda: rot.apply_inverse_to_point(pts).to_array())
    cap("vec_norm", lambda: vpos.norm())
    cap("vec_norm2", lambda: vpos.norm2())
    cap("vec_normalized", lambda: vpos.normalized().to_array())

    # ---- RBF / binning -----------------------------------------------------
    d = jnp.asarray(np.abs(rng.normal(0, 6, (13, 7))).astype(np.float32))
    put("in_d", d)

    cap("distance_rbf_0_22_64", lambda: G.distance_rbf(d, 0.0, 22.0, 64))
    cap("distance_rbf_0_22_16", lambda: G.distance_rbf(d, 0.0, 22.0, 16))
    cap("distance_rbf_0_10_16", lambda: G.distance_rbf(d, 0.0, 10.0, 16))
    cap("distance_one_hot_0_22_64", lambda: G.distance_one_hot(d, 0.0, 22.0, 64))
    centers, cstep = G.linear_centers(0.0, 22.0, 64)
    cap("linear_centers_0_22_64", lambda: centers)
    cap("linear_centers_step_0_22_64", lambda: cstep)
    cap("gaussian_rbf", lambda: G.gaussian_rbf(d[..., None], centers, cstep))

    # ---- neighbour selection (deterministic stages only) -------------------
    nb_idx = G.get_index_neighbours(16)(resi, chain, batch, mask)
    cap("get_index_neighbours_16", lambda: nb_idx)
    nb_sp = G.get_spatial_neighbours(16)(vpos[:, 1], batch, mask, nb_idx)
    cap("get_spatial_neighbours_16_chained", lambda: nb_sp)
    nb_sp0 = G.get_spatial_neighbours(20)(vpos[:, 1], batch, mask, None)
    cap("get_spatial_neighbours_20_fresh", lambda: nb_sp0)

    # get_neighbours on an explicit distance matrix (covers the ties path)
    dm = jnp.asarray(np.abs(rng.normal(0, 4, (N, N))).astype(np.float32))
    dm = (dm + dm.T) / 2
    pm = jnp.asarray(mask[:, None] & mask[None, :])
    put("in_dm", dm)

    put("in_pm", pm)

    cap("get_neighbours_12", lambda: G.get_neighbours(12)(dm, pm, None))
    cap("get_neighbours_12_chained", lambda: G.get_neighbours(12)(dm, pm, nb_idx))

    # ---- relative position encoding ---------------------------------------
    neighbours = nb_sp
    put("in_neighbours", neighbours)

    cap("seq_relpos_32_onehot", lambda: G.sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, neighbours=neighbours))
    cap("seq_relpos_32_raw", lambda: G.sequence_relative_position(32, one_hot=False)(
            resi, chain, batch, neighbours=neighbours))
    cyc_mask = np.zeros(N, bool)
    cyc_mask[:21] = True
    put("in_cyclic_mask", cyc_mask)

    cap("seq_relpos_32_cyclic", lambda: G.sequence_relative_position(32, one_hot=True, cyclic=True)(
            resi, chain, batch, neighbours=neighbours,
            cyclic_mask=jnp.asarray(cyc_mask)))

    # ---- pair features -----------------------------------------------------
    cap("distance_features", lambda: GM.distance_features(vpos, neighbours, d_max=22.0))
    cap("direction_features", lambda: GM.direction_features(vpos, neighbours))
    cap("position_rotation_features", lambda: GM.position_rotation_features(vpos, neighbours))
    cap("pair_vector_features", lambda: GM.pair_vector_features(vpos, neighbours))
    cap("rotation_features", lambda: GM.rotation_features(frames, neighbours))

    # ---- segment ops -------------------------------------------------------
    data = jnp.asarray(rng.normal(size=(N, 5)).astype(np.float32))
    put("in_segdata", data)

    cap("index_sum", lambda: G.index_sum(data, jnp.asarray(chain), jnp.asarray(mask)[..., None]))
    cap("index_mean", lambda: G.index_mean(data, jnp.asarray(chain), jnp.asarray(mask)[..., None]))
    cap("index_max", lambda: G.index_max(data, jnp.asarray(chain), jnp.asarray(mask)[..., None]))
    cap("index_std", lambda: G.index_std(data, jnp.asarray(chain), jnp.asarray(mask)[..., None]))
    cap("index_count", lambda: G.index_count(jnp.asarray(chain), jnp.asarray(mask)))
    cap("unique_chain", lambda: G.unique_chain(jnp.asarray(chain), jnp.asarray(batch)))

    # ---- time embeddings ---------------------------------------------------
    import salad.modules.utils.diffusion as D
    tt = jnp.asarray(np.linspace(0.02, 1.0, N).astype(np.float32))
    put("in_t", tt)

    cap("fourier_time_embedding_256", lambda: D.fourier_time_embedding(tt, size=256))

    import salad.modules.noise_schedule_benchmark as NSB
    cap("sigma_scale_cosine", lambda: NSB.sigma_scale_cosine(tt))
    cap("sigma_scale_framediff", lambda: NSB.sigma_scale_framediff(tt))

    # ---- sidechain reconstruction -----------------------------------------
    aatype = jnp.asarray(rng.integers(0, 20, N).astype(np.int32))
    angles = rng.normal(size=(N, 7, 2)).astype(np.float32)
    angles /= np.linalg.norm(angles, axis=-1, keepdims=True)
    put("in_aatype", aatype)

    put("in_angles", angles)

    cap("single_protein_sidechains",
        lambda: G.single_protein_sidechains(aatype, frames, jnp.asarray(angles))[0])

    np.savez(out_path, **OUT)
    print(f"wrote {out_path}: {len(OUT)} arrays")
    if FAILED:
        print(f"\n{len(FAILED)} CAPTURES FAILED:")
        for f in FAILED:
            print("   !", f)
    for k in sorted(OUT):
        print(f"   {k:44s} {OUT[k].shape} {OUT[k].dtype}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/geometry.npz")

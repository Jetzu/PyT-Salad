"""Dump every term of salad's JAX `Diffusion.loss` on a fixed input.

Run inside the container:
    apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD --bind <scratch>:<scratch> \
        salad.sif <scratch>/gen_loss_ref.py <params.jax> <out.npz>
"""
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import salad.modules.noise_schedule_benchmark as NSB
import salad.modules.utils.geometry as G
from salad.aflib.common import residue_constants as RC
from salad.aflib.model import all_atom_multimer as AAM
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import noise_schedule_benchmark as configs

_FIELDS = {}


def install_fixed_uniform(name):
    """Patch get_random_neighbours to use a pinned uniform field `name`."""
    u = jnp.asarray(_FIELDS[name])

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
            gumbel = jnp.log(-jnp.log(u))
            weight = weight - gumbel
            distance = jnp.where(same_item, -weight, jnp.inf)
            pair_mask = mask[:, None] * mask[None, :] * same_item
            return G.get_neighbours(count)(distance, pair_mask, neighbours)
        return inner

    G.get_random_neighbours = patched
    NSB.get_random_neighbours = patched


def build_data(rng, augment_size=15):
    """Two batch items, three chains, padding, mixed t_pos."""
    # item 0: one chain, 40 residues, t_pos = 0.30 (late)
    # item 1: two chains (30 + 26), 56 residues, t_pos = 0.80 (not late)
    #         last 6 residues masked out (padding)
    lens = [(0, [40]), (1, [30, 26])]
    resi, chain, batch = [], [], []
    for b, chains in lens:
        for ci, cl in enumerate(chains):
            resi.append(np.arange(cl, dtype=np.int32))
            chain.append(np.full(cl, ci, np.int32))
            batch.append(np.full(cl, b, np.int32))
    resi = np.concatenate(resi)
    chain = np.concatenate(chain)
    batch = np.concatenate(batch)
    n = resi.shape[0]
    a = 5 + augment_size

    mask = np.ones(n, bool)
    mask[-6:] = False                       # padding tail inside item 1

    # a plausible helix-ish backbone per chain, so frames/bonds are sane
    pos = np.zeros((n, a, 3), np.float32)
    atom_pos = np.zeros((n, 14, 3), np.float32)
    off = 0
    for b, chains in lens:
        for ci, cl in enumerate(chains):
            t = np.arange(cl)[:, None]
            ca = np.concatenate(
                [2.3 * np.cos(t * 1.7), 2.3 * np.sin(t * 1.7), 1.5 * t], -1)
            ca = ca + np.array([[40.0 * b, 25.0 * ci, 0.0]])
            blob = ca[:, None, :] + rng.normal(0, 1.2, (cl, a, 3))
            pos[off:off + cl] = blob.astype(np.float32)
            # atom14: N, CA, C, O from idealised offsets around CA + sidechain blob
            ideal = np.array([[-1.46, 0.0, 0.0], [0.0, 0.0, 0.0],
                              [0.53, 1.43, 0.0], [0.63, 2.62, 0.25]], np.float32)
            ap = np.zeros((cl, 14, 3), np.float32)
            ap[:, :4] = ca[:, None, :] + ideal[None]
            ap[:, 4:] = ca[:, None, :] + rng.normal(0, 2.0, (cl, 10, 3))
            atom_pos[off:off + cl] = ap
            off += cl

    aa_gt = rng.integers(0, 20, n).astype(np.int32)
    aa_gt[3] = 20          # unresolved identity
    aa_gt[50] = 20
    aa_gt[7] = RC.restype_order['P']    # a proline, for the C-N bond branch
    aa_gt[8] = RC.restype_order['C']    # cysteines, for the disulfide exclusion
    aa_gt[9] = RC.restype_order['C']

    canonical = np.asarray(RC.restype_atom14_mask)[aa_gt]        # (n, 14)
    drop = rng.random((n, 14)) < 0.10
    atom_mask = (canonical * (~drop)).astype(np.float32)

    t_pos = np.where(batch == 0, 0.30, 0.45).astype(np.float32)

    return dict(
        pos=pos.astype(np.float32),
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        aa_gt=aa_gt,
        aa_mask=(rng.random(n) < 0.2),
        dssp_gt=rng.integers(0, 3, n).astype(np.int32),
        dssp_mask=(rng.random(n) < 0.2),
        seq=np.full(n, 20, np.int32),
        residue_index=resi,
        chain_index=chain,
        batch_index=batch,
        mask=mask,
        cyclic_mask=np.zeros(n, bool),
        t_pos=t_pos,
        t_seq=np.full(n, 1.0, np.float32),
    )


def main(param_path, out_path, config_name="default_vp"):
    rng = np.random.default_rng(20260925)
    config = getattr(configs, config_name)
    config.eval = True

    with open(param_path, "rb") as f:
        params = pickle.load(f)

    data = build_data(rng, config.augment_size)
    n = data["mask"].shape[0]
    _FIELDS["model"] = rng.uniform(1e-6, 1 - 1e-6, (n, n)).astype(np.float32)
    _FIELDS["loss"] = rng.uniform(1e-6, 1 - 1e-6, (n, n)).astype(np.float32)

    out = {f"in_{k}": v for k, v in data.items()}
    out["uniform_model"] = _FIELDS["model"]
    out["uniform_loss"] = _FIELDS["loss"]

    prev = dict(pos=np.zeros_like(data["pos"]),
                local=np.zeros((n, config.local_size), np.float32))

    # ---- 1. noising (fixes pos_noised) -------------------------------------
    noise_t = hk.transform(lambda d: NSB.StructureDiffusionNoise(config)(d))
    jdata = jax.tree_util.tree_map(jnp.asarray, data)
    noised = noise_t.apply({}, jax.random.PRNGKey(0), jdata)
    for k, v in noised.items():
        out[f"noised_{k}"] = np.asarray(v)

    # ---- 2. real denoiser forward with the real checkpoint -----------------
    install_fixed_uniform("model")
    model_in = {k: jdata[k] for k in
                ("pos", "aa_gt", "seq", "residue_index", "chain_index",
                 "batch_index", "mask", "cyclic_mask")}
    model_in.update(noised)
    predict_t = hk.transform(
        lambda d, p: NSB.StructureDiffusionPredict(config)(d, p))
    jprev = jax.tree_util.tree_map(jnp.asarray, prev)
    result, _ = predict_t.apply(params, jax.random.PRNGKey(7), dict(model_in), jprev)
    for k, v in result.items():
        try:
            out[f"res_{k}"] = np.asarray(v)
        except Exception:
            pass

    # ---- 3. the loss, term by term -----------------------------------------
    install_fixed_uniform("loss")
    loss_data = dict(jdata)
    loss_data.update(noised)

    def loss_fn(d, r):
        return NSB.Diffusion(config).loss(d, r)

    loss_t = hk.transform(loss_fn)
    total, losses = loss_t.apply({}, jax.random.PRNGKey(11), loss_data, result)
    out["loss_total"] = np.asarray(total)
    for k, v in losses.items():
        out[f"loss_{k}"] = np.asarray(v)
        print(f"  loss/{k:22s} {float(v): .10f}")
    print(f"  loss/{'TOTAL':22s} {float(total): .10f}")

    # ---- 4. extra probes: the two neighbour draws, the bound tables ---------
    def probe(d, r):
        c = config
        mask = d["mask"]
        batch_ = d["batch_index"]
        pos_gt = jnp.where(mask[:, None, None], d["pos"], 0)
        pos_gt_v = Vec3Array.from_array(pos_gt)
        pm = (batch_[:, None] == batch_[None, :]) * mask[:, None] * mask[None, :]
        dist = (pos_gt_v[:, None, 4] - pos_gt_v[None, :, 4]).norm()
        dist = jnp.where(pm, dist, jnp.inf)
        nb = NSB.get_random_neighbours(c.fape_neighbours)(dist, batch_, mask)
        ln = G.get_neighbours(c.local_neighbours)(dist, pm)
        return dict(fape_neighbours=nb, local_neighbours=ln, cb_distance=dist)

    probe_t = hk.transform(probe)
    pr = probe_t.apply({}, jax.random.PRNGKey(11), loss_data, result)
    for k, v in pr.items():
        out[f"probe_{k}"] = np.asarray(v)

    bounds = RC.make_atom14_dists_bounds(overlap_tolerance=1.5,
                                         bond_length_tolerance_factor=2.0)
    for k, v in bounds.items():
        out[f"bounds_{k}"] = np.asarray(v)
    atomtype_radius = np.array([RC.van_der_waals_radius[nm[0]]
                                for nm in RC.atom_types], np.float32)
    a14to37 = np.asarray(AAM.get_atom14_to_atom37_map(jnp.arange(21)))
    out["atom14_radius_table"] = atomtype_radius[a14to37]
    out["restype_atom14_mask"] = np.asarray(RC.restype_atom14_mask)

    np.savez(out_path, **out)
    print(f"wrote {out_path}: {len(out)} arrays (n={n})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

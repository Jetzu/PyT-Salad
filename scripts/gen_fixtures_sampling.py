"""Run the FULL JAX sampling loop with every random draw pinned, and dump it.

This is the end-to-end reference: a complete de-novo design trajectory, so the
PyTorch port can be checked on the thing that actually matters - the generated
protein - not just one forward pass.

Both sources of randomness are injected from a file so the two backends can be
made to agree exactly:
  * `noise`   (steps, L, A, 3) - the Gaussian draw inside diffuse_atom_cloud
  * `uniform` (steps, L, L)    - stands in for the Gumbel-top-k neighbour draw

Run inside the container:
    apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
        scripts/gen_fixtures_sampling.py fixtures/sampling_default_vp.npz \
        --length 64 --num_steps 100 --out_steps 80
"""
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import salad.modules.noise_schedule_benchmark as NSB
import salad.modules.utils.geometry as G
import salad.modules.utils.diffusion as DIFF
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import noise_schedule_benchmark as configs

PARAMS = "/hnvme/workspace/v181bb10-salad/params/default_vp-200k.jax"

STATE = {"step": 0, "uniform": None, "noise": None}


def install_injectors():
    """Replace both RNG consumers with lookups into the pinned arrays."""

    def patched_random_neighbours(count):
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
            u = jnp.asarray(STATE["uniform"][STATE["step"]])
            weight = -3 * jnp.log(distance + 1e-6) - jnp.log(-jnp.log(u))
            distance = jnp.where(same_item, -weight, jnp.inf)
            pair_mask = mask[:, None] * mask[None, :] * same_item
            return G.get_neighbours(count)(distance, pair_mask, neighbours)
        return inner

    G.get_random_neighbours = patched_random_neighbours
    NSB.get_random_neighbours = patched_random_neighbours


def noise_step(pos, t, batch_index, sigma_data):
    """diffuse_atom_cloud with the Gaussian draw injected (vpfixed path)."""
    target = sigma_data * jnp.asarray(STATE["noise"][STATE["step"]])
    # CA-centroid removal, per batch item
    center = DIFF.index_mean(target[:, 1], batch_index,
                             jnp.ones((target.shape[0], 1), dtype=jnp.bool_))
    target = target - center[:, None, :]
    t = t[:, None, None]
    return pos * jnp.sqrt(jnp.maximum(1 - t ** 2, 0.0)) + target * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--out_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    L, S = args.length, args.num_steps
    config = getattr(configs, "default_vp")
    config.eval = True
    A = 5 + config.augment_size

    rng = np.random.default_rng(args.seed)
    STATE["noise"] = rng.standard_normal((S + 1, L, A, 3)).astype(np.float32)
    STATE["uniform"] = rng.uniform(1e-6, 1 - 1e-6, (S + 1, L, L)).astype(np.float32)

    install_injectors()

    with open(PARAMS, "rb") as fh:
        params = pickle.load(fh)

    def fn(data, prev):
        return NSB.StructureDiffusionPredict(config)(data, prev)

    model = hk.transform(fn).apply
    key = jax.random.PRNGKey(0)

    z = np.zeros(L, np.int32)
    data = dict(
        pos=jnp.zeros((L, A, 3), jnp.float32),
        aa_gt=jnp.full((L,), 20, jnp.int32),
        seq=jnp.full((L,), 20, jnp.int32),
        residue_index=jnp.arange(L, dtype=jnp.int32),
        chain_index=jnp.asarray(z), batch_index=jnp.asarray(z),
        mask=jnp.ones(L, bool), cyclic_mask=jnp.zeros(L, bool),
    )
    init_prev = dict(pos=jnp.zeros((L, A, 3), jnp.float32),
                     local=jnp.zeros((L, config.local_size), jnp.float32))
    prev = init_prev

    out = {"in_length": np.int32(L), "in_num_steps": np.int32(S),
           "in_out_steps": np.int32(args.out_steps),
           "noise": STATE["noise"], "uniform": STATE["uniform"]}

    prev_threshold = 0.8
    traj_pos, traj_t = [], []

    for step in range(S + 1):
        STATE["step"] = step
        raw_t = 1 - step / S
        t = float(NSB.sigma_scale_cosine(jnp.asarray(raw_t)))
        traj_t.append(t)
        data["t_pos"] = jnp.full((L,), t, jnp.float32)
        data["t_seq"] = jnp.ones((L,), jnp.float32)
        if raw_t < prev_threshold:
            prev = init_prev
        data["pos_noised"] = noise_step(data["pos"], data["t_pos"],
                                        data["batch_index"], config.sigma_data)
        data["seq_noised"] = data["seq"]
        result, prev = model(params, key, data, prev)
        data["pos"] = result["pos"]
        data["seq"] = jnp.argmax(result["aa"], axis=-1)
        traj_pos.append(np.asarray(result["pos"]))
        if step == args.out_steps:
            out["final_pos"] = np.asarray(result["pos"])
            out["final_atom_pos"] = np.asarray(result["atom_pos"])
            out["final_aatype"] = np.asarray(result["aatype"])
            out["final_aa"] = np.asarray(result["aa"])
            out["final_local"] = np.asarray(result["local"])
            break

    out["traj_pos"] = np.stack(traj_pos)
    out["traj_t"] = np.array(traj_t, np.float32)
    np.savez(args.out, **out)
    ca = out["final_pos"][:, 1]
    print(f"wrote {args.out}")
    print(f"  steps run: {len(traj_pos)}  final t={traj_t[-1]:.4f}")
    print(f"  CA radius of gyration: {np.sqrt(((ca - ca.mean(0))**2).sum(-1).mean()):.2f} A")
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    print(f"  CA-CA: mean {d.mean():.3f} A, min {d.min():.3f}, max {d.max():.3f}")


if __name__ == "__main__":
    main()

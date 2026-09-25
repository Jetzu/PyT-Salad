"""Time the JAX denoiser, for comparison with scripts/bench_torch.py.

Run inside the salad container:
    apptainer run --nv --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
        scripts/bench_jax.py --lengths 50,100,200,400 --out bench_jax.json

Uses jit (as upstream does at inference), so the reported time excludes
compilation - warmup absorbs it.
"""
import argparse
import json
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import salad.modules.noise_schedule_benchmark as NSB
from salad.modules.config import noise_schedule_benchmark as configs

PARAMS = "/hnvme/workspace/v181bb10-salad/params/default_vp-200k.jax"


def make_data(length, augment_size=15):
    rng = np.random.default_rng(0)
    a = 5 + augment_size
    t = np.arange(length)[:, None]
    bb = np.concatenate([2.3 * np.cos(t * 1.7), 2.3 * np.sin(t * 1.7), 1.5 * t], -1)
    pos = (bb[:, None, :] + rng.normal(0, 1.2, (length, a, 3))).astype(np.float32)
    z = np.zeros(length, np.int32)
    data = dict(
        pos=pos, pos_noised=pos.copy(),
        aa_gt=np.full(length, 20, np.int32),
        seq=np.full(length, 20, np.int32),
        seq_noised=np.full(length, 20, np.int32),
        residue_index=np.arange(length, dtype=np.int32),
        chain_index=z, batch_index=z,
        mask=np.ones(length, bool), cyclic_mask=np.zeros(length, bool),
        t_pos=np.full(length, 0.5, np.float32),
        t_seq=np.ones(length, np.float32),
    )
    prev = dict(pos=np.zeros_like(pos), local=np.zeros((length, 128), np.float32))
    return (jax.tree_util.tree_map(jnp.asarray, data),
            jax.tree_util.tree_map(jnp.asarray, prev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="50,100,200,400")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("devices:", jax.devices())
    config = getattr(configs, "default_vp")
    config.eval = True
    with open(PARAMS, "rb") as fh:
        params = pickle.load(fh)

    def fn(data, prev):
        return NSB.StructureDiffusionPredict(config)(data, prev)

    model = jax.jit(hk.transform(fn).apply)
    key = jax.random.PRNGKey(0)

    results = {}
    for length in [int(x) for x in args.lengths.split(",")]:
        data, prev = make_data(length, config.augment_size)
        for _ in range(args.warmup):
            out = model(params, key, data, prev)
        jax.block_until_ready(out)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            out = model(params, key, data, prev)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        ms = np.array(times) * 1e3
        results[length] = dict(median_ms=float(np.median(ms)),
                               min_ms=float(ms.min()),
                               p90_ms=float(np.percentile(ms, 90)))
        print(f"L={length:<5} median {np.median(ms):8.2f} ms   "
              f"min {ms.min():8.2f} ms   "
              f"-> 500-step design {np.median(ms) * 500 / 1000:7.1f} s")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"backend": "jax", "device": str(jax.devices()[0]),
                       "results": results}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

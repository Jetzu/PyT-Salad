"""End-to-end effect of one candidate change, measured in a FRESH process.

usage: x_e2e.py <variant>[+<variant>...] [--length 100] [--profile]
variants: base tables onehot nopc noaa
Monkeypatches only (no production code touched). Reports compiled+cudagraph
wall time of one denoiser forward AND the deviation of the patched EAGER
output from the unpatched eager output on identical input.
"""
import argparse, sys, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure")
from common import build, make_data, bench, kernel_stats, FastDenoiser

ap = argparse.ArgumentParser()
ap.add_argument("variant")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--profile", action="store_true")
a = ap.parse_args()
V = set(a.variant.split("+"))
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
dev = torch.device("cuda")


def patch_tables():
    import salad_torch.geometry.sidechain as SC
    SC._tables(dev)


def patch_onehot():
    """one-hot matmul -> row gather (bit-exact: 65 exact zeros + one value)."""
    from salad_torch.modules.basic import HkLinear
    import salad_torch.geometry.features as GF
    import salad_torch.modules.decoders as D

    orig_fwd = HkLinear.forward

    def fwd(self, x):
        if not torch.is_floating_point(x):
            out = self.w[x]
            return out + self.b if self.use_bias else out
        return orig_fwd(self, x)
    HkLinear.forward = fwd

    orig_srp = GF.sequence_relative_position

    def srp(count=32, one_hot=True, cyclic=False, **kw):
        return orig_srp(count, one_hot=False, cyclic=cyclic, **kw)
    GF.sequence_relative_position = srp

    def aapf(self, pos, aa, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] & mask[neighbours] & (neighbours != -1)
        relpos = GF.sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, neighbours)
        pair = self.linear_relpos(relpos)
        pair = pair + self.linear_dist(GF.distance_features(pos, neighbours, 0.0, 22.0))
        pair = pair + self.linear_dir(GF.direction_features(pos, neighbours))
        pair = pair + self.linear_rot(GF.position_rotation_features(pos, neighbours))
        pair = pair + self.linear_pairvec(GF.pair_vector_features(pos, neighbours))
        aa_embedding = self.layer_norm_aa(self.linear_aa(aa))
        pair = pair + aa_embedding[neighbours]
        return self.mlp(self.layer_norm(pair)), pair_mask
    D.AADecoderPairFeatures.forward = aapf


def patch_nopc():
    """Skip the dmap / flags pair terms; for de-novo they contribute exactly 0."""
    import salad_torch.modules.blocks as B
    GF = B.GF

    def fwd(self, pos, pair_condition, neighbours, resi, chain, batch, mask,
            cyclic_mask=None, init_pos=None):
        mask = mask.bool()
        pair_mask = mask[:, None] & mask[neighbours] & (neighbours != -1)
        relpos = GF.sequence_relative_position(
            32, one_hot=True, cyclic=cyclic_mask is not None)(
                resi, chain, batch, neighbours, cyclic_mask=cyclic_mask)
        pair = self.linear_relpos(relpos)
        pair = pair + self.linear_dist(
            GF.distance_features(pos, neighbours, 0.0, 22.0))
        pair = pair + self.linear_dir(GF.direction_features(pos, neighbours))
        pair = pair + self.linear_rot(GF.position_rotation_features(pos, neighbours))
        pair = pair + self.linear_pairvec(GF.pair_vector_features(pos, neighbours))
        if self.use_init_pos:
            pair = pair + self.linear_init_dist(
                GF.distance_features(init_pos, neighbours, 0.0, 22.0))
            pair = pair + self.linear_init_rot(
                GF.position_rotation_features(init_pos, neighbours))
        return self.mlp(self.layer_norm(pair)), pair_mask
    B.DiffusionPairFeatures.forward = fwd


def patch_noaa():
    import salad_torch.modules.decoders as D

    def aa_fwd(self, aa, local, pos, resi, chain, batch, mask):
        n = local.shape[0]
        return (torch.zeros(n, 20, device=local.device, dtype=local.dtype),
                torch.zeros(n, local.shape[-1], device=local.device, dtype=local.dtype))
    D.AADecoder.forward = aa_fwd

    def ah_fwd(self, aatype, local, pos):
        n = local.shape[0]
        z = torch.zeros(n, 7, 2, device=local.device, dtype=local.dtype)
        return z, z, torch.zeros(n, 14, 3, device=local.device, dtype=local.dtype)
    D.AngleHead.forward = ah_fwd


def patch_k48():
    """In de-novo mode the block-contact stage of extract_condition_neighbours
    yields 16 all-(-1) columns (measured). Drop it: K 64 -> 48 for the whole
    condition branch. Masked slots contribute exactly 0 to the softmax."""
    import salad_torch.modules.diffusion as DF
    from salad_torch.geometry.neighbours import (get_index_neighbours,
                                                 get_neighbours,
                                                 get_random_neighbours)

    def ecn(num_index, num_spatial, num_random, num_block):
        def inner(pos, pair_condition, resi, chain, batch, mask,
                  uniform=None, uniform2=None):
            nb = get_index_neighbours(num_index)(resi, chain, batch, mask)
            ca = pos[:, 1]
            dist = torch.sqrt(torch.clamp(
                ((ca[:, None] - ca[None, :]) ** 2).sum(-1), min=1e-12))
            dist = torch.where(pair_condition["dmap_mask"],
                               pair_condition["dmap"], dist)
            same_batch = batch[:, None] == batch[None, :]
            pm = mask[:, None] & mask[None, :] & same_batch
            nb = get_neighbours(num_spatial)(dist, pm, nb)
            nb = get_random_neighbours(num_random)(dist, batch, mask, nb,
                                                   uniform=uniform)
            return nb
        return inner
    DF.extract_condition_neighbours = ecn


PATCHES = dict(tables=patch_tables, onehot=patch_onehot,
               nopc=patch_nopc, noaa=patch_noaa, k48=patch_k48)

m, cfg = build()
d, p = make_data(cfg, a.length)
with torch.no_grad():
    ref = m(d, p, compute_distogram=False)[0]["pos"].clone()

for name in ("tables", "onehot", "nopc", "noaa", "k48"):
    if name in V:
        PATCHES[name]()

with torch.no_grad():
    got = m(d, p, compute_distogram=False)[0]["pos"]
    dev_eager = (got - ref).abs().max().item()
    fast = FastDenoiser(m, a.length, device=dev, mode="reduce-overhead").warmup()
    t = bench(lambda: fast(d, p), iters=50, warm=10)
print(f"VARIANT {a.variant:<22} L={a.length:<4} {t:6.3f} ms  "
      f"500-step {t*500/1000:5.2f} s  max|dpos| vs unpatched eager {dev_eager:.3e}")
if a.profile:
    with torch.no_grad():
        us, n, rows = kernel_stats(lambda: fast(d, p))
    print(f"  GPU busy {us/1000:.3f} ms/fwd in {n:.0f} kernels; "
          f"launch gap {t - us/1000:.3f} ms")
    for t_us, cnt, key in rows[:16]:
        print(f"   {t_us:8.1f} us  x{cnt:6.1f}  {key[:66]}")
sys.stdout.flush()

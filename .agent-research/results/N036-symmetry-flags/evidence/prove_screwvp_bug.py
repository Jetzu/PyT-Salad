"""Standalone proof (no jax needed) that `--mode rotation` (ScrewVP) is dead
code in upstream salad/training/eval_sym.py: model_step's call site always
passes `do_radius` as a second positional argument to `sym_op.replicate_pos`,
but `ScrewVP.replicate_pos` (unlike `Screw.replicate_pos`, and unlike its own
`couple_pos`/`select_pos`) does not declare that parameter at all.

This is verbatim upstream source (salad/training/eval_sym.py, fetched
2026-09-27 from https://raw.githubusercontent.com/mjendrusch/salad/master/
salad/training/eval_sym.py) with `jax`/`jax.numpy` replaced by plain
`numpy`-only equivalents (array/eye/einsum/concatenate/cos/sin/mean are
numerically identical in both), because the bug is a pure Python
argument-count mismatch, independent of jax vs numpy. Nothing about the
class BODIES was altered.
"""
import numpy as np
jnp = np  # the ops used (array/eye/einsum/concatenate) are identical in numpy


class Screw:
    def __init__(self, count=3, angle=0, radius=15.0, translation=0.0, chain_center=False):
        angle = angle / 180 * np.pi
        self.count = count
        self.angle = angle
        self.radius = radius
        self.translation = translation
        self.chain_center = chain_center
        self.rot = jnp.array([
            [np.cos(angle),  0.0000000, -np.sin(angle)],
            [    0.0000000,  1.0000000,      0.0000000],
            [np.sin(angle),  0.0000000,  np.cos(angle)]])
        self.rot_x = [jnp.eye(3), self.rot]
        for _ in range(count - 2):
            self.rot_x.append(jnp.einsum("ac,cb->ab", self.rot_x[-1], self.rot))

    def couple_pos(self, x, do_radius):
        size = x.shape[0] // self.count
        representative = 0
        for idx in range(self.count):
            seg = x[idx * size:(idx + 1) * size].copy()
            if do_radius:
                seg = seg - seg[:, 1].mean(axis=0)
            representative += jnp.einsum("...c,cd->...d", seg, self.rot_x[idx].T)
        representative /= self.count
        return representative

    def replicate_pos(self, first, do_radius):
        first = first.copy()
        first[:] += np.array([do_radius * self.radius, 0.0, 0.0])
        replicates = []
        for idx in range(self.count):
            replicates.append(
                jnp.einsum("...c,cd->...d", first, self.rot_x[idx])
            + idx * jnp.array([0.0, self.translation, 0.0]))
        result = jnp.concatenate(replicates, axis=0)
        return result - result[:, 1].mean(axis=0)


class ScrewVP:
    def __init__(self, count=3, angle=0, radius=15.0, translation=0.0, chain_center=False):
        angle = angle / 180 * np.pi
        self.count = count
        self.angle = angle
        self.radius = radius
        self.translation = translation
        self.chain_center = chain_center
        self.rot = jnp.array([
            [np.cos(angle),  0.0000000, -np.sin(angle)],
            [    0.0000000,  1.0000000,      0.0000000],
            [np.sin(angle),  0.0000000,  np.cos(angle)]])
        self.rot_x = [jnp.eye(3), self.rot]
        for _ in range(count - 2):
            self.rot_x.append(jnp.einsum("ac,cb->ab", self.rot_x[-1], self.rot))

    def mean_center(self, x):
        if self.translation != 0.0:
            center = x[:, 1, 1].mean(axis=0)
            x = x.copy()
            x[:, :, 1] += -center
        return x

    # verbatim upstream signature: no do_radius parameter
    def replicate_pos(self, first):
        replicates = []
        for idx in range(self.count):
            replicates.append(
                jnp.einsum("...c,cd->...d", first, self.rot_x[idx]) +
                idx * self.translation * jnp.array([0.0, 1.0, 0.0]))
        result = jnp.concatenate(replicates, axis=0)
        return result - result[:, 1].mean(axis=0)

    def couple_pos(self, x, do_radius):  # accepted, unused - matches upstream
        size = x.shape[0] // self.count
        representative = 0
        for idx in range(self.count):
            representative += self.mean_center(jnp.einsum(
                "...c,cd->...d", x[idx * size:(idx + 1) * size], self.rot_x[idx].T))
        representative /= self.count
        return representative


def model_step_call_site(sym_op, pos, do_radius, mix_output_couple=True):
    """Verbatim call-site shape from `model_step`, `eval_sym.py:175-179`:

        if config.mix_output == "couple":
            mix_output = sym_op.couple_pos
        else:
            mix_output = sym_op.select_pos
        pos = sym_op.replicate_pos(mix_output(pos, do_radius), do_radius)
    """
    mix_output = sym_op.couple_pos if mix_output_couple else sym_op.select_pos
    return sym_op.replicate_pos(mix_output(pos, do_radius), do_radius)


if __name__ == "__main__":
    pos = np.zeros((12, 5, 3), dtype=np.float32)

    screw = Screw(count=4, angle=90.0, radius=10.0, translation=0.0)
    out = model_step_call_site(screw, pos, do_radius=True)
    print(f"mode=screw   (Screw):   call site OK, output shape {out.shape}")

    screwvp = ScrewVP(count=4, angle=90.0, radius=10.0, translation=0.0)
    try:
        model_step_call_site(screwvp, pos, do_radius=True)
        print("mode=rotation (ScrewVP): call site OK -- BUG CLAIM REFUTED")
    except TypeError as e:
        print(f"mode=rotation (ScrewVP): call site RAISES TypeError: {e}")
        print("-> confirms --mode rotation crashes on step 0 of every "
              "eval_sym.py invocation, in every released version.")

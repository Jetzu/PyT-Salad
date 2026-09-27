"""N028/O8: every frame transform is a cuBLAS bmm. Make it a Triton reduction.

`apply_t` / `apply_inverse_t` / `rot_apply` / `rot_apply_inverse` / `rot_compose`
are written as `torch.einsum("...ij,...j->...i", ...)` (and `@`). ATen lowers
ALL of them to `aten.bmm` - measured: 61 of the 303 `extern_kernels.bmm` calls
in the generated module have a `(..., 3, 3)` operand, i.e. they are 3x3 frame
algebra dispatched to cuBLAS. A cuBLAS launch costs 5.98 us for 9 FLOPs, against
a 1.06 us Triton-kernel floor, and the `permute`/`view` chain einsum needs to
reach bmm layout forces a `clone` kernel on top.

Written as an explicit broadcast-multiply + `sum` the same arithmetic becomes a
pointwise+reduction node inductor can FUSE into the surrounding elementwise work
instead of a launch of its own.
"""
import re
import sys

W = sys.argv[1]

# ---- opt.py flag ---------------------------------------------------------
p = W + "/salad_torch/opt.py"
s = open(p).read()
s = s.replace('"IPA_MULSUM", "PAIR_FEAT_MERGE"]',
              '"IPA_MULSUM", "PAIR_FEAT_MERGE", "FRAME_MULSUM"]')
s += '''

#: N028/O8 - how the 3x3 frame algebra contracts. `apply_t`, `apply_inverse_t`,
#: `rot_apply`, `rot_apply_inverse` and `rot_compose` are `torch.einsum`
#: (or `@`), and ATen lowers EVERY one of them to `aten.bmm`: 61 of the 303
#: `extern_kernels.bmm` calls in the generated module have a `(..., 3, 3)`
#: operand. That is a 5.98 us cuBLAS launch for nine multiply-adds, plus the
#: `clone` inductor emits to get the permuted operands into bmm layout.
#:
#: `SALAD_OPT_FRAMEMULSUM=1` writes them as broadcast-multiply + `sum` over the
#: contracted axis, so each becomes an ordinary pointwise+reduction node that
#: inductor can fuse into the surrounding elementwise work. Same class of trade
#: as `IPA_MULSUM`, and the same direction: it removes cuBLAS launches and adds
#: (at most) Triton ones.
#:
#: It is a REASSOCIATION over a 3-element contraction (bitwise identical in
#: eager CPU fp32, but cuBLAS is free to contract with FMA in a different
#: order), so its baseline is whatever the CPU bitwise control and the n=160
#: structural gate say - see N028's RESULTS.md.
FRAME_MULSUM = _flag("SALAD_OPT_FRAMEMULSUM", default=False)
'''
open(p, "w").write(s)

# ---- frames.py -----------------------------------------------------------
p = W + "/salad_torch/geometry/frames.py"
s = open(p).read()
old = '''def apply_t(rot: Tensor, trans: Tensor, points: Tensor, k: int = 0) -> Tensor:
    """``R p + t``, broadcasting over `k` inserted axes."""
    rot, trans = _lift(rot, trans, k)
    return torch.einsum("...ij,...j->...i", rot, points) + trans


def apply_inverse_t(rot: Tensor, trans: Tensor, points: Tensor, k: int = 0) -> Tensor:
    """``R^T (p - t)``, broadcasting over `k` inserted axes."""
    rot, trans = _lift(rot, trans, k)
    return torch.einsum("...ji,...j->...i", rot, points - trans)'''
new = '''def apply_t(rot: Tensor, trans: Tensor, points: Tensor, k: int = 0) -> Tensor:
    """``R p + t``, broadcasting over `k` inserted axes."""
    rot, trans = _lift(rot, trans, k)
    if opt.FRAME_MULSUM:
        # N028/O8 - a 3x3 matvec as a fusable reduction, not a cuBLAS bmm.
        return (rot * points.unsqueeze(-2)).sum(-1) + trans
    return torch.einsum("...ij,...j->...i", rot, points) + trans


def apply_inverse_t(rot: Tensor, trans: Tensor, points: Tensor, k: int = 0) -> Tensor:
    """``R^T (p - t)``, broadcasting over `k` inserted axes."""
    rot, trans = _lift(rot, trans, k)
    if opt.FRAME_MULSUM:
        return (rot * (points - trans).unsqueeze(-1)).sum(-2)
    return torch.einsum("...ji,...j->...i", rot, points - trans)'''
assert old in s
s = s.replace(old, new)
s = s.replace("from .rigid import Rigid, rot_from_two_vectors",
              "from .. import opt\nfrom .rigid import Rigid, rot_from_two_vectors")
open(p, "w").write(s)

# ---- rigid.py ------------------------------------------------------------
p = W + "/salad_torch/geometry/rigid.py"
s = open(p).read()
old = '''def rot_apply(rot: Tensor, point: Tensor) -> Tensor:
    """R @ p  for R (..., 3, 3) and p (..., 3), broadcasting over leading dims."""
    return torch.einsum("...ij,...j->...i", rot, point)


def rot_apply_inverse(rot: Tensor, point: Tensor) -> Tensor:
    """R^T @ p. Rotations are orthonormal, so the inverse is the transpose."""
    return torch.einsum("...ji,...j->...i", rot, point)


def rot_compose(a: Tensor, b: Tensor) -> Tensor:
    return a @ b'''
new = '''def rot_apply(rot: Tensor, point: Tensor) -> Tensor:
    """R @ p  for R (..., 3, 3) and p (..., 3), broadcasting over leading dims."""
    if opt.FRAME_MULSUM:                     # N028/O8
        return (rot * point.unsqueeze(-2)).sum(-1)
    return torch.einsum("...ij,...j->...i", rot, point)


def rot_apply_inverse(rot: Tensor, point: Tensor) -> Tensor:
    """R^T @ p. Rotations are orthonormal, so the inverse is the transpose."""
    if opt.FRAME_MULSUM:                     # N028/O8
        return (rot * point.unsqueeze(-1)).sum(-2)
    return torch.einsum("...ji,...j->...i", rot, point)


def rot_compose(a: Tensor, b: Tensor) -> Tensor:
    if opt.FRAME_MULSUM:                     # N028/O8
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)
    return a @ b'''
assert old in s
s = s.replace(old, new)
assert "\nfrom .. import opt\n" not in s
s = s.replace("from torch import Tensor", "from torch import Tensor\n\nfrom .. import opt", 1)
open(p, "w").write(s)

# ---- features.py: position_rotation_features / rotation_features ---------
p = W + "/salad_torch/geometry/features.py"
s = open(p).read()
old = '''    rot_i = frames.rotation[:, None]                      # (N, 1, 3, 3)
    rot_j = frames.rotation[neighbours]                   # (N, K, 3, 3)
    rel = rot_i.transpose(-1, -2) @ rot_j
    return rel.reshape(rel.shape[0], rel.shape[1], 9)'''
new = '''    rot_i = frames.rotation[:, None]                      # (N, 1, 3, 3)
    rot_j = frames.rotation[neighbours]                   # (N, K, 3, 3)
    rel = rot_compose(rot_i.transpose(-1, -2), rot_j)
    return rel.reshape(rel.shape[0], rel.shape[1], 9)'''
assert old in s
s = s.replace(old, new)
old = '''    rot, _, _ = aa_frames_t(pos)
    rel = rot[:, None].transpose(-1, -2) @ rot[neighbours]
    return rel.reshape(rel.shape[0], rel.shape[1], 9)'''
new = '''    rot, _, _ = aa_frames_t(pos)
    rel = rot_compose(rot[:, None].transpose(-1, -2), rot[neighbours])
    return rel.reshape(rel.shape[0], rel.shape[1], 9)'''
assert old in s
s = s.replace(old, new)
s = s.replace("from .rigid import Rigid, vec_norm, vec_normalized",
              "from .rigid import Rigid, rot_compose, vec_norm, vec_normalized")
open(p, "w").write(s)

# ---- fast.py: preset key + the process-global set -----------------------
p = W + "/salad_torch/modules/fast.py"
s = open(p).read()
s = s.replace("_ENV_IPA_MULSUM = opt.IPA_MULSUM",
              "_ENV_IPA_MULSUM = opt.IPA_MULSUM\n"
              "#: ...and `SALAD_OPT_FRAMEMULSUM=1` forces N028/O8 on (same OR contract).\n"
              "_ENV_FRAME_MULSUM = opt.FRAME_MULSUM")
s = s.replace("        opt.IPA_MULSUM = bool(p.get(\"ipa_mulsum\")) or _ENV_IPA_MULSUM",
              "        opt.IPA_MULSUM = bool(p.get(\"ipa_mulsum\")) or _ENV_IPA_MULSUM\n"
              "        # N028/O8, same contract again.\n"
              "        opt.FRAME_MULSUM = bool(p.get(\"frame_mulsum\")) or _ENV_FRAME_MULSUM")
s = s.replace("        ipa_mulsum=False,\n    ),", "        ipa_mulsum=False,\n        frame_mulsum=False,\n    ),")
s = s.replace("        ipa_mulsum=False,   # N028 candidate - flipped to True only if n=160 clears it\n    ),",
              "        ipa_mulsum=False,   # N028 candidate - flipped to True only if n=160 clears it\n"
              "        frame_mulsum=False, # N028 candidate - same\n    ),")
open(p, "w").write(s)
print("O8 applied to", W)

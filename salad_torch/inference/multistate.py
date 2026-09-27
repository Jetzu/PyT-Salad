"""Multi-state protein design: the tertiary-structure coupling for a
``--dssp "parent/child_1/child_2"`` spec (see `options.py::parse_dssp_split`
for the string-parsing half of this feature).

Ported from `salad/training/eval_split_change.py::couple_part` /
`SplitChange.update_pos` (`notes/specs/07_multistate_dssp.md`). This is the
task described by Lisanza *et al.*, 2024 (Nature Biotechnology): design a
*parent* protein with a specified secondary structure which, split into an
N-terminal and a C-terminal half, gives two *child* proteins that adopt a
*different* secondary structure from the parent - while residues marked
uppercase in the parent's DSSP string keep the same *tertiary* structure
(actual 3-D coordinates, not just a secondary-structure label) in the
corresponding child.

The mechanism is a **hard, one-directional coordinate override** applied at
every denoising step, not a loss term: at each step's incoming ``pos`` (the
previous step's x0 prediction - all-zeros on step 0, matching upstream's
``init_pos``), for every ``same_mask`` residue, a child's coordinates are
*replaced* by the corresponding parent segment's coordinates, rigidly
superimposed onto the child via a Kabsch fit computed **from the masked
residues only** (`salad_torch.geometry.align.index_align`, which gates its
per-item mean/covariance by exactly the `mask` argument given). Unmasked
residues are left as the child's own prediction. The parent's own trajectory
is **not** pulled toward its children - see `update_split_change`'s docstring
for the (deliberately reproduced) asymmetry.

This lives in its own module, not in `options.py`, mirroring how `--sym`'s
string parsing stays in `options.py`-adjacent code while its geometry
(`screw_replicate`) lives in `inference/symmetry.py` - `options.py` is
parsing-only, this module is runtime geometry wired into the sampling loop
by `scripts/generate.py::run_multistate`.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..geometry.align import index_align

__all__ = ["couple_part", "update_split_change"]


def couple_part(x: Tensor, y: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """One child/parent-segment pair. `eval_split_change.py::couple_part`.

    Args:
        x: a CHILD's full positions, `(L, A, 3)`.
        y: the corresponding PARENT segment's positions, `(L, A, 3)` - same
            `L` as `x` (a 1:1 per-residue correspondence is required; see
            `options.py::parse_dssp_split`'s length check, which is what
            guarantees this holds by the time this function is called).
        mask: `(L,)` bool - `same_mask` restricted to this segment. `True`
            marks a residue constrained to share tertiary structure.

    Returns `(x_new, y_new)`. The Kabsch fit inside `index_align` is
    determined **only** from the masked residues (its `mask` argument gates
    `index_kabsch`'s per-item mean/covariance reduction), then the resulting
    rigid transform is applied to *every* atom of the moved tensor - so
    `y_at_x`/`x_at_y` below are full-length, rigidly-moved copies of `y`/`x`,
    and it is the final `torch.where` that actually leaves unmasked residues
    untouched.
    """
    batch = torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
    # "y aligned to x": the parent segment y, rigidly moved into the child
    # x's frame, using x as the fixed Kabsch target.
    y_at_x = index_align(y, x, batch, mask)
    # "x aligned to y": the child x, rigidly moved into the parent segment
    # y's frame, using y as the fixed Kabsch target.
    x_at_y = index_align(x, y, batch, mask)
    m = mask[:, None, None]
    x_new = torch.where(m, y_at_x, x)
    y_new = torch.where(m, 0.5 * y + 0.5 * x_at_y, y)
    return x_new, y_new


def update_split_change(parent_pos: Tensor, child_1_pos: Tensor, child_2_pos: Tensor,
                        same_mask: Tensor, parent_split: int
                        ) -> tuple[Tensor, Tensor, Tensor]:
    """One coupling step. `eval_split_change.py::SplitChange.update_pos`.

    Call this on the incoming (previous-step) ``pos`` of all three states,
    before re-noising/denoising each of them for the current step - exactly
    where upstream's `step()` calls it, right after unpacking
    `data[i]["pos"]` and before `center`/`noise`/`predict`. See
    `scripts/generate.py::run_multistate`'s `step_fn_group` for the call
    site.

    Splits `parent_pos` at `parent_split` (`== child_1_pos.shape[0]`) into
    the segments corresponding to child_1 (`p_1 = parent_pos[:split]`) and
    child_2 (`p_2 = parent_pos[split:]`), couples each child onto its
    parent segment via `couple_part`, and returns the parent **unchanged**.

    That last part is a deliberately reproduced upstream asymmetry, not a
    simplification: `couple_part` computes an updated value for the parent
    segment too (its second return value - `y_new` above), but upstream
    discards it (`c_1, _ = couple_part(c_1, p_1, m_1)`) and reassembles
    `parent = concatenate((p_1, p_2))` from the **original**, uncoupled
    segments. So the constraint is one-directional: at every step, children
    are pulled onto the parent's current tertiary structure at `same_mask`
    positions; the parent's own denoising trajectory is never influenced by
    either child. (The concatenation below is consequently a no-op - it
    reproduces upstream's own reassembly for parity/clarity, not because it
    changes `parent_pos`.)
    """
    p_1, p_2 = parent_pos[:parent_split], parent_pos[parent_split:]
    m_1, m_2 = same_mask[:parent_split], same_mask[parent_split:]
    child_1_new, _ = couple_part(child_1_pos, p_1, m_1)
    child_2_new, _ = couple_part(child_2_pos, p_2, m_2)
    parent_new = torch.cat((p_1, p_2), dim=0)
    return parent_new, child_1_new, child_2_new

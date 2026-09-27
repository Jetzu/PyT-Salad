"""Secondary-structure assignment (`assign_dssp`) and its (dead-for-`default_vp`)
conditioning dropout (`drop_dssp`).

Port of `salad.modules.utils.dssp` per `notes/specs/04_data_and_training.md`
§A.6. This is CPU/numpy-preprocessing territory in spirit (it is a per-residue
sequential recurrence in the original), but the actual reference implementation
runs on-device inside the JAX training step, batched, on the already-centered
atom14 ground truth (`data["atom_pos"]`) - so this module is written against
`torch.Tensor` throughout, not `numpy`, to slot into the training step the same
way. Nothing here needs a gradient (it consumes ground truth only), so callers
should wrap it in `torch.no_grad()`.

Only `assign_dssp`'s first return value, `dssp`, reaches the loss/conditioning
for `default_vp` (spec §0c): `blocks`/`block_adjacency` are dead weight, and
`drop_dssp` is dead code entirely (its outputs are discarded upstream). Both
are still implemented in full below, because `assign_dssp` cannot be spec'd
without them and another config (SS/ADJ-conditioned variants) may need them.

Index convention: `assign_dssp` takes `coord` = ground-truth atom14 (or wider)
positions, batch item ids, and a per-residue VALIDITY mask (not a pair mask -
the pair mask is built internally). Returns `(dssp, blocks, block_adjacency)`
with `dssp[i] in {0=loop, 1=helix, 2=strand}`, or the sentinel `3` for every
residue if the whole flattened batch has fewer than 5 residues (matches the
"no DSSP conditioning provided" sentinel used elsewhere - do not clamp it).
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["get_hbond_map", "assign_dssp", "drop_dssp"]

_N, _CA, _C, _O = 0, 1, 2, 3


def _unfold(x: Tensor, size: int, axis: int) -> Tensor:
    """Sliding window of length `size`, step 1, along `axis`.

    Matches the JAX helper `_unfold` used by `get_hbond_map`/`assign_dssp`:
    the target axis shrinks to `N - size + 1` and a new trailing axis of
    length `size` is appended - exactly `torch.Tensor.unfold`'s contract.
    """
    axis = axis if axis >= 0 else x.dim() + axis
    return x.unfold(axis, size, 1)


def _peptide_bond_h_position(coord: Tensor) -> Tensor:
    """Backbone amide H position for residues ``1..L-1``. ``dssp.py:60-94`` (helper).

    Geometric placement: unit(C[i]->N[i+1]) + unit(CA[i+1]->N[i+1]), normalized,
    then 1.01 A from N[i+1]. Returns ``(L-1, 3)``, row ``i`` = H of residue ``i+1``.
    """
    n_next = coord[1:, _N]
    vec_cn = n_next - coord[:-1, _C]
    vec_cn = vec_cn / (vec_cn.norm(dim=-1, keepdim=True) + 1e-9)
    vec_can = n_next - coord[1:, _CA]
    vec_can = vec_can / (vec_can.norm(dim=-1, keepdim=True) + 1e-9)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / (vec_nh.norm(dim=-1, keepdim=True) + 1e-9)
    return n_next + 1.01 * vec_nh


def get_hbond_map(coord: Tensor, mask: Tensor, cutoff: float = -0.5,
                  margin: float = 1.0) -> Tensor:
    """Kabsch-Sander electrostatic H-bond map, relaxed to ``[0, 1]``.

    ``dssp.py:60-94``. `coord` is ``(L, >=4, 3)`` (only N/CA/C/O, slots 0-3,
    are read). `mask` is a caller-supplied PAIR mask ``(L, L)`` (same-item AND
    both-residues-valid) - this function does not compute one itself.

    The `k=-1`/`k=-2` local-exclusion diagonal (see module docstring) only
    needs to be self-consistent with the transpose `assign_dssp` applies right
    after calling this - it is not meaningful in isolation.
    """
    length = coord.shape[0]
    h = _peptide_bond_h_position(coord)                       # (L-1, 3)
    nmap = coord[1:, None, _N]
    hmap = h[:, None]
    cmap = coord[None, :-1, _C]
    omap = coord[None, :-1, _O]

    def _dist(a: Tensor, b: Tensor) -> Tensor:
        return torch.sqrt(((a - b) ** 2).sum(-1) + 1e-12) + 1e-3

    d_on = _dist(omap, nmap)
    d_ch = _dist(cmap, hmap)
    d_oh = _dist(omap, hmap)
    d_cn = _dist(cmap, nmap)
    e = 0.084 * (1.0 / d_on + 1.0 / d_ch - 1.0 / d_oh - 1.0 / d_cn) * 332.0
    # pad(e, [[1,0],[0,1]]): one zero row on top, one zero column on the right
    e = torch.nn.functional.pad(e, (0, 1, 1, 0))              # -> (L, L)

    eye = torch.eye(length, dtype=torch.bool, device=coord.device)
    diag_m1 = torch.diag(torch.ones(max(length - 1, 0), dtype=torch.bool,
                                    device=coord.device), diagonal=-1) \
        if length > 1 else torch.zeros((length, length), dtype=torch.bool, device=coord.device)
    diag_m2 = torch.diag(torch.ones(max(length - 2, 0), dtype=torch.bool,
                                    device=coord.device), diagonal=-2) \
        if length > 2 else torch.zeros((length, length), dtype=torch.bool, device=coord.device)
    local_mask = ~(eye | diag_m1 | diag_m2)

    hbond_map = torch.clamp(cutoff - margin - e, -margin, margin)
    hbond_map = (torch.sin(hbond_map / margin * torch.pi / 2) + 1) / 2
    hbond_map = torch.where(local_mask, hbond_map, torch.zeros_like(hbond_map))
    hbond_map = torch.where(mask.bool(), hbond_map, torch.zeros_like(hbond_map))
    return hbond_map


def _roll(x: Tensor, shift: int) -> Tensor:
    return torch.roll(x, shifts=shift, dims=0)


def assign_dssp(coord: Tensor, batch_index: Tensor, mask: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
    """3-state secondary structure + block id + (dead, for `default_vp`) block
    adjacency. ``dssp.py:96-166``.

    Args:
        coord: ``(L, >=14, 3)`` ground-truth positions (atom14 or wider); only
            slots 0-3 (N, CA, C, O) and 1 (CA, for `block_adjacency`) are read.
        batch_index: ``(L,)`` which flattened-batch item each residue belongs to.
        mask: ``(L,)`` residue VALIDITY (has >=1 resolved atom) - not a pair mask.

    Returns:
        ``(dssp, blocks, block_adjacency)``: ``dssp`` is ``(L,)`` int in
        ``{0, 1, 2}`` (loop, helix, strand), or the out-of-range sentinel ``3``
        everywhere if ``L < 5`` (degenerate-input guard, matches the
        "no DSSP conditioning" sentinel used elsewhere - never clamped into
        range). ``blocks`` is ``(L,)`` int, a run-length id over `dssp`.
        ``block_adjacency`` is ``(L, L)`` bool; DEAD for `default_vp` (spec
        §0c) - computed here for API completeness / other configs only.
    """
    length = coord.shape[0]
    device = coord.device

    if length < 5:
        dssp = torch.full((length,), 3, dtype=torch.long, device=device)
        blocks = torch.zeros((length,), dtype=torch.long, device=device)
        block_adjacency = torch.zeros((length, length), dtype=torch.bool, device=device)
        return dssp, blocks, block_adjacency

    mask_b = mask.bool()
    pair_mask = (batch_index[:, None] == batch_index[None, :]) & mask_b[:, None] & mask_b[None, :]
    hbmap = get_hbond_map(coord, pair_mask)
    hbmap = hbmap.transpose(-1, -2)

    def _diag(offset: int) -> Tensor:
        return (torch.diagonal(hbmap, offset=offset) > 0).long()

    turn3, turn4, turn5 = _diag(3), _diag(4), _diag(5)

    def _pad(x: Tensor, left: int, right: int) -> Tensor:
        return torch.nn.functional.pad(x, (left, right))

    h3 = _pad(turn3[:-1] * turn3[1:], 1, 3)
    h4 = _pad(turn4[:-1] * turn4[1:], 1, 4)
    h5 = _pad(turn5[:-1] * turn5[1:], 1, 5)

    helix4 = h4 + _roll(h4, 1) + _roll(h4, 2) + _roll(h4, 3)
    not_helix4_or_next = (1 - (_roll(helix4, -1) > 0).long()) * (1 - (helix4 > 0).long())
    h3 = h3 * not_helix4_or_next
    h5 = h5 * not_helix4_or_next
    helix3 = h3 + _roll(h3, 1) + _roll(h3, 2)
    helix5 = h5 + _roll(h5, 1) + _roll(h5, 2) + _roll(h5, 3) + _roll(h5, 4)

    # -- bridge / strand detection: 3x3 sliding windows of hbmap along both axes
    unfoldmap = (_unfold(_unfold(hbmap, 3, axis=-2), 3, axis=-2)) > 0   # (L-2, L-2, 3, 3)
    unfoldmap_rev = unfoldmap.transpose(0, 1)

    def _pad2(x: Tensor) -> Tensor:
        return torch.nn.functional.pad(x, (1, 1, 1, 1))

    p_bridge = (unfoldmap[:, :, 0, 1] & unfoldmap_rev[:, :, 1, 2]) | \
        (unfoldmap_rev[:, :, 0, 1] & unfoldmap[:, :, 1, 2])
    p_bridge = _pad2(p_bridge.long())
    p_bridge = torch.where(pair_mask, p_bridge, torch.zeros_like(p_bridge))

    a_bridge = (unfoldmap[:, :, 1, 1] & unfoldmap_rev[:, :, 1, 1]) | \
        (unfoldmap[:, :, 0, 2] & unfoldmap_rev[:, :, 0, 2])
    a_bridge = _pad2(a_bridge.long())
    a_bridge = torch.where(pair_mask, a_bridge, torch.zeros_like(a_bridge))

    ladder = (p_bridge + a_bridge).sum(-1) > 0

    # NOTE (dead-code, spec A.6.2): upstream computes p_strand/a_strand/p_res/
    # a_res and a first `strand` from them, then immediately overwrites it with
    # `strand = ladder` before it is ever read. That whole computation is
    # skipped here - the final `index` depends only on helix3/4/5 and ladder.
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix) & (~strand)

    # argmax over stack([loop, helix, strand]): ties broken loop(0) > helix(1)
    # > strand(2), i.e. first-True-wins - `torch.stack(...).long().argmax(-1)`
    # reproduces that because argmax returns the FIRST maximal index on ties.
    index = torch.stack([loop, helix, strand], dim=-1).long().argmax(dim=-1)

    # -- block labeling: run-length id over `index` (vectorized `lax.scan`) --
    changed = torch.zeros_like(index, dtype=torch.bool)
    changed[1:] = index[1:] != index[:-1]
    blocks = torch.cumsum(changed.long(), dim=0)

    # -- block_adjacency (dead for default_vp, spec A.6.2) -------------------
    num_blocks = int(blocks.max().item()) + 1
    ca = coord[:, _CA]
    distance = torch.sqrt(((ca[:, None] - ca[None, :]) ** 2).sum(-1) + 1e-12)
    flat_index = blocks[:, None] * num_blocks + blocks[None, :]
    acc = torch.full((num_blocks * num_blocks,), 1e6, dtype=distance.dtype, device=device)
    acc.scatter_reduce_(0, flat_index.reshape(-1), distance.reshape(-1),
                        reduce="amin", include_self=True)
    block_distance_table = acc.view(num_blocks, num_blocks)
    block_distance = block_distance_table[blocks][:, blocks]

    same_block = blocks[:, None] == blocks[None, :]
    # sheet_sheet split is a no-op for default_vp (SHEET_ADJACENCY ==
    # OTHER_ADJACENCY == 8.0); kept as a `where` for parity with configs that
    # might set them differently.
    block_adjacency = torch.where(block_distance <= 8.0,
                                  torch.ones_like(same_block),
                                  torch.zeros_like(same_block))
    zero_out = same_block | loop[:, None] | loop[None, :]
    block_adjacency = torch.where(zero_out, torch.zeros_like(block_adjacency), block_adjacency)

    return index, blocks, block_adjacency.bool()


def drop_dssp(secondary_structure: Tensor, blocks: Tensor, block_adjacency: Tensor,
             p_drop: float = 0.2, *, generator: torch.Generator | None = None
             ) -> tuple[Tensor, Tensor]:
    """Per-block dropout of the DSSP conditioning. ``dssp.py:168-173``.

    **Dead for `default_vp`** (spec §0c): `prepare_condition` discards this
    function's `secondary_structure` return (the loop keeps the UNDROPPED
    `dssp` from `assign_dssp`) and never reads its `block_adjacency` return
    either. Implemented only for RNG-stream parity / other configs.

    Draws one Bernoulli value per residue INDEX (not per block), then gathers
    it through `blocks` - so every residue of a given block sees the same
    draw. That is upstream's literal (slightly wasteful) idiom, reproduced here
    rather than "fixed" into one draw per block.
    """
    draw = torch.rand(secondary_structure.shape[0], generator=generator,
                      device=secondary_structure.device) < p_drop
    drop_mask = draw[blocks]
    drop_pair = drop_mask[:, None] | drop_mask[None, :]
    secondary_structure = torch.where(drop_mask, torch.zeros_like(secondary_structure),
                                      secondary_structure)
    block_adjacency = torch.where(drop_pair, torch.zeros_like(block_adjacency), block_adjacency)
    return secondary_structure, block_adjacency

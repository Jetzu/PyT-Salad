"""The AlphaFold structural-violation loss stack.

Ported per `notes/specs/03_training_losses.md` §7.10, §9.3: `violation_loss`
(`salad/modules/utils/alphafold_loss.py:337-376`) calls
`find_structural_violations` then, since salad always passes
`per_residue=False`, `custom_structural_violation_loss`. Only that path is
implemented - `find_structural_violations`'s per-residue violation mask and
`compute_violation_metrics` (`salad/aflib/model/folding_multimer.py:291`) are
both computed upstream but their outputs are DISCARDED (`violation, _ =
violation_loss(...)`); porting them would be pure wasted compute, so they are
not here.

`stereo_chemical_props.txt` (bundled at `losses/data/`, copied VERBATIM from
`upstream/salad/salad/aflib/common/`) is pure physical literature data: bond
lengths/angles per residue, by measurement. `load_stereo_chemical_props` and
`make_atom14_dists_bounds` below are the standard, publicly documented
AlphaFold table-building algorithm for that file format (law-of-cosines
"virtual bond" derivation with linear error propagation) - not salad-specific
logic, and not derived by reading salad's source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ..geometry.rigid import vec_dot, vec_norm, vec_normalized
from ..io.residue_constants import (RESTYPE_ATOM14_MASK, restype_1to3,
                                    restype_name_to_atom14_names, restype_order,
                                    restypes)

__all__ = [
    "load_stereo_chemical_props", "make_atom14_dists_bounds", "get_atom14_mask",
    "between_residue_bond_loss", "custom_clash_loss", "within_residue_violations",
    "custom_structural_violation_loss", "find_structural_violations", "violation_loss",
    "RESTYPE_ATOM14_RADIUS",
]

_DATA_PATH = Path(__file__).parent / "data" / "stereo_chemical_props.txt"

# literal literature constants, spec §9.3 (residue_constants.py:483-488)
_BETWEEN_RES_BOND_LENGTH_C_N = (1.329, 1.341)              # (non-proline, proline)
_BETWEEN_RES_BOND_LENGTH_STDDEV_C_N = (0.014, 0.016)
_BETWEEN_RES_COS_ANGLES_CA_C_N = (-0.4473, 0.0311)         # (mean, stddev)
_BETWEEN_RES_COS_ANGLES_C_N_CA = (-0.5203, 0.0353)

_VAN_DER_WAALS_RADIUS = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}
_SG_IDX = restype_name_to_atom14_names["CYS"].index("SG")


def _build_atom14_radius() -> Tensor:
    rows = []
    for rt in restypes:
        names = restype_name_to_atom14_names[restype_1to3[rt]]
        rows.append([_VAN_DER_WAALS_RADIUS.get(name[0], 0.0) if name else 0.0
                    for name in names])
    rows.append([0.0] * 14)  # UNK
    return torch.tensor(rows, dtype=torch.float32)


RESTYPE_ATOM14_RADIUS = _build_atom14_radius()  # (21, 14)


def get_atom14_mask(aatype: Tensor) -> Tensor:
    """`RESTYPE_ATOM14_MASK[aatype]` - the CANONICAL per-restype atom mask
    (does this amino acid HAVE this atom), NOT `data["atom_mask"]` (was it
    resolved in the crystal structure) - `losses.structure.x_loss` uses the
    latter, this loss uses the former (spec §7.10)."""
    return RESTYPE_ATOM14_MASK.to(aatype.device)[aatype]


# ---------------------------------------------------------------------------
# stereo_chemical_props.txt parsing (pure data, spec §9.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Bond:
    atom1: str
    atom2: str
    length: float
    stddev: float


@dataclass(frozen=True)
class _Angle:
    atom1: str
    atom2: str
    atom3: str
    angle_rad: float
    stddev_rad: float


def load_stereo_chemical_props(path: Path = _DATA_PATH
                               ) -> tuple[dict[str, list[_Bond]], dict[str, list[_Angle]],
                                         dict[str, list[_Bond]]]:
    """Parse the bond-length / bond-angle table, derive "virtual bond" (2-hop)
    distances from angles via the law of cosines.

    Returns `(residue_bonds, residue_angles, residue_virtual_bonds)`, each a
    dict keyed by 3-letter residue code (plus an empty `"UNK"` entry).
    """
    lines = path.read_text().splitlines()
    it = iter(lines)

    residue_bonds: dict[str, list[_Bond]] = {}
    next(it)  # header
    for line in it:
        if line.strip() == "-":
            break
        bond, resname, length, stddev = line.split()
        a1, a2 = bond.split("-")
        residue_bonds.setdefault(resname, []).append(_Bond(a1, a2, float(length), float(stddev)))
    residue_bonds["UNK"] = []

    residue_angles: dict[str, list[_Angle]] = {}
    next(it)  # blank line
    next(it)  # header
    for line in it:
        if line.strip() == "-":
            break
        bond, resname, angle_deg, stddev_deg = line.split()
        a1, a2, a3 = bond.split("-")
        residue_angles.setdefault(resname, []).append(
            _Angle(a1, a2, a3, math.radians(float(angle_deg)), math.radians(float(stddev_deg))))
    residue_angles["UNK"] = []

    def _key(a: str, b: str) -> str:
        return "-".join(sorted((a, b)))

    residue_virtual_bonds: dict[str, list[_Bond]] = {}
    for resname, angles in residue_angles.items():
        cache = {_key(b.atom1, b.atom2): b for b in residue_bonds[resname]}
        virt = []
        for ang in angles:
            b1 = cache[_key(ang.atom1, ang.atom2)]
            b2 = cache[_key(ang.atom2, ang.atom3)]
            gamma = ang.angle_rad
            length = math.sqrt(b1.length ** 2 + b2.length ** 2
                               - 2 * b1.length * b2.length * math.cos(gamma))
            # linear error propagation, uncorrelated errors
            dl = 0.5 / length
            dl_dgamma = (2 * b1.length * b2.length * math.sin(gamma)) * dl
            dl_db1 = (2 * b1.length - 2 * b2.length * math.cos(gamma)) * dl
            dl_db2 = (2 * b2.length - 2 * b1.length * math.cos(gamma)) * dl
            stddev = math.sqrt((dl_dgamma * ang.stddev_rad) ** 2
                               + (dl_db1 * b1.stddev) ** 2 + (dl_db2 * b2.stddev) ** 2)
            virt.append(_Bond(ang.atom1, ang.atom3, length, stddev))
        residue_virtual_bonds[resname] = virt
    return residue_bonds, residue_angles, residue_virtual_bonds


_BOUNDS_CACHE: dict[tuple[float, float], dict[str, Tensor]] = {}


def make_atom14_dists_bounds(overlap_tolerance: float = 1.5,
                             bond_length_tolerance_factor: float = 15.0
                             ) -> dict[str, Tensor]:
    """`(21, 14, 14)` lower/upper/stddev atom-pair distance bound tables.

    Every non-bonded pair gets a clash-avoidance floor (`vdw1+vdw2-overlap`,
    no ceiling); every bonded pair (including 2-hop "virtual bonds" derived
    from angles) OVERWRITES that with a tight `length +/- factor*stddev` band.
    `bond_length_tolerance_factor`'s OWN default here is 15 (matching the
    function this was ported from), but spec §7.10 requires salad's actual
    call site to override it with `violation_tolerance_factor` (2.0 for
    `default_vp`) - never rely on this function's default.
    """
    key = (overlap_tolerance, bond_length_tolerance_factor)
    cached = _BOUNDS_CACHE.get(key)
    if cached is not None:
        return cached

    residue_bonds, _, residue_virtual_bonds = load_stereo_chemical_props()
    lower = torch.zeros(21, 14, 14)
    upper = torch.zeros(21, 14, 14)
    stddev = torch.zeros(21, 14, 14)
    for ri, rt in enumerate(restypes):
        resname = restype_1to3[rt]
        atoms = restype_name_to_atom14_names[resname]
        for i, a1 in enumerate(atoms):
            if not a1:
                continue
            r1 = _VAN_DER_WAALS_RADIUS.get(a1[0], 0.0)
            for j, a2 in enumerate(atoms):
                if not a2 or i == j:
                    continue
                r2 = _VAN_DER_WAALS_RADIUS.get(a2[0], 0.0)
                lower[ri, i, j] = r1 + r2 - overlap_tolerance
                upper[ri, i, j] = 1e10
        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            i, j = atoms.index(b.atom1), atoms.index(b.atom2)
            lo = b.length - bond_length_tolerance_factor * b.stddev
            hi = b.length + bond_length_tolerance_factor * b.stddev
            lower[ri, i, j] = lower[ri, j, i] = lo
            upper[ri, i, j] = upper[ri, j, i] = hi
            stddev[ri, i, j] = stddev[ri, j, i] = b.stddev

    result = {"lower_bound": lower, "upper_bound": upper, "stddev": stddev}
    _BOUNDS_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# (i) between_residue_bond_loss - spec §7.10(i)
# ---------------------------------------------------------------------------
def between_residue_bond_loss(pred_atom_pos: Tensor, pred_atom_mask: Tensor,
                              residue_index: Tensor, aatype: Tensor,
                              tolerance_factor: float = 2.0) -> dict[str, Tensor]:
    """C-N peptide bond length + the two flanking bond angles, between
    SEQUENCE-ADJACENT residues (atom14 indices: N=0, CA=1, C=2).

    UPSTREAM TRAP: there is NO `batch_index`/`chain_index` parameter here at
    all. The only protection against scoring a fake peptide bond across a
    chain/batch-item boundary is `has_no_gap_mask` (`residue_index` must
    increase by EXACTLY 1). This is an IMPLICIT INVARIANT of salad's data
    pipeline - `residue_index` is expected to reset (not keep incrementing) at
    every chain/item boundary - not an explicit safety mask. Preserve
    `residue_index` exactly as given; do not "clean it up" into one globally
    increasing counter, or this silently starts scoring cross-protein geometry.
    """
    device = pred_atom_pos.device
    this_ca, this_c = pred_atom_pos[:-1, 1], pred_atom_pos[:-1, 2]
    next_n, next_ca = pred_atom_pos[1:, 0], pred_atom_pos[1:, 1]
    this_ca_m, this_c_m = pred_atom_mask[:-1, 1].float(), pred_atom_mask[:-1, 2].float()
    next_n_m, next_ca_m = pred_atom_mask[1:, 0].float(), pred_atom_mask[1:, 1].float()
    has_no_gap = (residue_index[1:] - residue_index[:-1] == 1).float()

    c_n_bond_length = vec_norm(this_c - next_n, eps=1e-6)
    next_is_proline = aatype[1:] == restype_order["P"]
    gt_length = torch.where(
        next_is_proline,
        torch.tensor(_BETWEEN_RES_BOND_LENGTH_C_N[1], device=device),
        torch.tensor(_BETWEEN_RES_BOND_LENGTH_C_N[0], device=device))
    gt_stddev = torch.where(
        next_is_proline,
        torch.tensor(_BETWEEN_RES_BOND_LENGTH_STDDEV_C_N[1], device=device),
        torch.tensor(_BETWEEN_RES_BOND_LENGTH_STDDEV_C_N[0], device=device))
    c_n_error = torch.sqrt(1e-6 + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = F.relu(c_n_error - tolerance_factor * gt_stddev)
    m = this_c_m * next_n_m * has_no_gap
    c_n_loss = (m * c_n_loss_per_residue).sum() / (m.sum() + 1e-6)

    c_ca_unit = vec_normalized(this_ca - this_c, eps=1e-6)
    c_n_unit = (next_n - this_c) / c_n_bond_length[..., None]     # NOT renormalised - already eps-clamped
    n_ca_unit = vec_normalized(next_ca - next_n, eps=1e-6)

    ca_c_n_cos = vec_dot(c_ca_unit, c_n_unit)
    ca_c_n_err = torch.sqrt(1e-6 + (ca_c_n_cos - _BETWEEN_RES_COS_ANGLES_CA_C_N[0]) ** 2)
    ca_c_n_loss_per_residue = F.relu(
        ca_c_n_err - tolerance_factor * _BETWEEN_RES_COS_ANGLES_CA_C_N[1])
    m2 = this_ca_m * this_c_m * next_n_m * has_no_gap
    ca_c_n_loss = (m2 * ca_c_n_loss_per_residue).sum() / (m2.sum() + 1e-6)

    c_n_ca_cos = vec_dot(-c_n_unit, n_ca_unit)
    c_n_ca_err = torch.sqrt(1e-6 + (c_n_ca_cos - _BETWEEN_RES_COS_ANGLES_C_N_CA[0]) ** 2)
    c_n_ca_loss_per_residue = F.relu(
        c_n_ca_err - tolerance_factor * _BETWEEN_RES_COS_ANGLES_C_N_CA[1])
    m3 = this_c_m * next_n_m * next_ca_m * has_no_gap
    c_n_ca_loss = (m3 * c_n_ca_loss_per_residue).sum() / (m3.sum() + 1e-6)

    # per-residue diagnostic only (feeds AlphaFold's per-residue violation
    # mask upstream, which salad never uses - not summed into the scalar loss)
    per_residue = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue = 0.5 * (F.pad(per_residue, (0, 1)) + F.pad(per_residue, (1, 0)))

    return {"c_n_loss_mean": c_n_loss, "ca_c_n_loss_mean": ca_c_n_loss,
           "c_n_ca_loss_mean": c_n_ca_loss, "per_residue_loss_sum": per_residue}


# ---------------------------------------------------------------------------
# (ii) custom_clash_loss - spec §7.10(ii)
# ---------------------------------------------------------------------------
def custom_clash_loss(pred_atom_pos: Tensor, atom_exists: Tensor, atom_radius: Tensor,
                      residue_index: Tensor, chain_index: Tensor, batch_index: Tensor,
                      overlap_tolerance: float = 1.5) -> dict[str, Tensor]:
    """salad's batch/chain-aware monkey-patch of AlphaFold's
    `between_residue_clash_loss` (spec §7.10(ii)).

    Chain safety comes from the `+2000*chain_index` offset making cross-chain
    residue indices non-adjacent, NOT from an explicit chain mask (only
    `batch_index` is masked explicitly, matching the reference).
    """
    # OUT-OF-PLACE `+`, never `+=` - the caller's `residue_index` tensor is
    # shared with every other loss term computed off `data["residue_index"]`
    # this step (e.g. `between_residue_bond_loss`, which must NOT see the
    # +2000*chain offset); an in-place update would silently corrupt it.
    residue_index = residue_index + 2000 * chain_index

    pos_i = pred_atom_pos[:, None, :, None, :]      # (L, 1, 14, 1, 3)
    pos_j = pred_atom_pos[None, :, None, :, :]      # (1, L, 1, 14, 3)
    dists = vec_norm(pos_i - pos_j, eps=1e-10)      # (L, L, 14, 14)

    exists_i = atom_exists[:, None, :, None].float()
    exists_j = atom_exists[None, :, None, :].float()
    dists_mask = exists_i * exists_j

    ri_i = residue_index[:, None, None, None]
    ri_j = residue_index[None, :, None, None]
    dists_mask = dists_mask * (ri_i < ri_j).float()                 # upper triangle only
    bi_i = batch_index[:, None, None, None]
    bi_j = batch_index[None, :, None, None]
    dists_mask = dists_mask * (bi_i == bi_j).float()                # explicit batch gate

    device = pred_atom_pos.device
    c_onehot = torch.zeros(14, device=device); c_onehot[2] = 1.0    # atom14 idx 2 = C
    n_onehot = torch.zeros(14, device=device); n_onehot[0] = 1.0    # atom14 idx 0 = N
    c_n_bonds = ((ri_i + 1 == ri_j).float()
                * c_onehot[None, None, :, None] * n_onehot[None, None, None, :])
    dists_mask = dists_mask * (1.0 - c_n_bonds)                     # exclude the real peptide bond

    sg_onehot = torch.zeros(14, device=device); sg_onehot[_SG_IDX] = 1.0
    disulfide = sg_onehot[None, None, :, None] * sg_onehot[None, None, None, :]
    dists_mask = dists_mask * (1.0 - disulfide)                     # exclude Cys-Cys disulfides

    lower_bound = dists_mask * (atom_radius[:, None, :, None] + atom_radius[None, :, None, :])
    error = dists_mask * F.relu(lower_bound - overlap_tolerance - dists)
    clash_mask = dists_mask * (dists < (lower_bound - overlap_tolerance)).float()
    mean_loss = error.sum() / (1e-6 + clash_mask.sum())

    # diagnostic only - not consumed by `custom_structural_violation_loss`
    per_atom_loss_sum = error.sum(dim=(0, 2)) + error.sum(dim=(1, 3))   # (L, 14)
    per_atom_clash_mask = clash_mask.sum(dim=(0, 2)) + clash_mask.sum(dim=(1, 3))

    return {"mean_loss": mean_loss, "per_atom_loss_sum": per_atom_loss_sum,
           "per_atom_clash_mask": per_atom_clash_mask}


# ---------------------------------------------------------------------------
# (iii) within_residue_violations - spec §7.10(iii)
# ---------------------------------------------------------------------------
def within_residue_violations(pred_atom_pos: Tensor, atom_exists: Tensor,
                              dists_lower_bound: Tensor, dists_upper_bound: Tensor,
                              eps: float = 1e-10) -> dict[str, Tensor]:
    """Purely intra-residue 14x14 atom-pair distance bounds - no cross-residue
    or batch coupling at all, safe by construction (spec §7.10(iii)).
    `dists_lower_bound`/`dists_upper_bound`: `(L, 14, 14)`, already gathered
    per-residue by `aatype` from `make_atom14_dists_bounds`.
    """
    n_atom = pred_atom_pos.shape[-2]
    eye = torch.eye(n_atom, device=pred_atom_pos.device)
    dists_mask = (1.0 - eye)[None] * (atom_exists[:, :, None].float()
                                      * atom_exists[:, None, :].float())
    dists = vec_norm(pred_atom_pos[:, :, None, :] - pred_atom_pos[:, None, :, :], eps=eps)
    low_err = F.relu(dists_lower_bound - dists)
    high_err = F.relu(dists - dists_upper_bound)
    loss = dists_mask * (low_err + high_err)
    per_atom_loss_sum = loss.sum(-2) + loss.sum(-1)                 # (L, 14)

    violations = dists_mask * ((dists < dists_lower_bound) | (dists > dists_upper_bound)).float()
    per_atom_violations = torch.maximum(violations.amax(-2), violations.amax(-1))
    return {"per_atom_loss_sum": per_atom_loss_sum, "per_atom_violations": per_atom_violations}


# ---------------------------------------------------------------------------
# assembly - spec §7.10 top level
# ---------------------------------------------------------------------------
def find_structural_violations(aatype: Tensor, residue_index: Tensor, pred_atom_pos: Tensor,
                               pred_atom_mask: Tensor, res_mask: Tensor, *,
                               clash_overlap_tolerance: float = 1.5,
                               violation_tolerance_factor: float = 2.0,
                               chain_index: Tensor, batch_index: Tensor) -> dict:
    """Runs all three violation sub-losses and packages them AlphaFold-style.
    `pred_atom_mask` (the canonical-times-res_mask atom14 mask the caller
    built, spec §7.10 preamble) serves as `atom_exists` in every sub-loss.
    `res_mask` is unused directly here (each sub-loss derives its own gating
    from `pred_atom_mask`) but accepted to mirror the reference signature.
    """
    del res_mask
    device = pred_atom_pos.device
    radius = RESTYPE_ATOM14_RADIUS.to(device)[aatype]

    bonds = between_residue_bond_loss(pred_atom_pos, pred_atom_mask, residue_index, aatype,
                                      tolerance_factor=violation_tolerance_factor)
    clash = custom_clash_loss(pred_atom_pos, pred_atom_mask, radius, residue_index,
                              chain_index, batch_index,
                              overlap_tolerance=clash_overlap_tolerance)
    bounds = make_atom14_dists_bounds(overlap_tolerance=clash_overlap_tolerance,
                                      bond_length_tolerance_factor=violation_tolerance_factor)
    lower = bounds["lower_bound"].to(device)[aatype]
    upper = bounds["upper_bound"].to(device)[aatype]
    within = within_residue_violations(pred_atom_pos, pred_atom_mask, lower, upper)

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": bonds["c_n_loss_mean"],
            "angles_ca_c_n_loss_mean": bonds["ca_c_n_loss_mean"],
            "angles_c_n_ca_loss_mean": bonds["c_n_ca_loss_mean"],
            "connections_per_residue_loss_sum": bonds["per_residue_loss_sum"],
            "clashes_mean_loss": clash["mean_loss"],
            "clashes_per_atom_loss_sum": clash["per_atom_loss_sum"],
            "clashes_per_atom_clash_mask": clash["per_atom_clash_mask"],
        },
        "within_residues": within,
    }


def custom_structural_violation_loss(violations: dict, pred_atom_mask: Tensor) -> Tensor:
    """spec §7.10 combination formula - a single scalar for the whole
    (flattened, multi-item) batch: no `base_weight`, no per-item
    equalisation anywhere, only the internal `/(count+eps)` guards baked into
    each sub-loss."""
    b = violations["between_residues"]
    w = violations["within_residues"]
    return (b["bonds_c_n_loss_mean"]
           + 0.3 * b["angles_ca_c_n_loss_mean"]
           + 0.3 * b["angles_c_n_ca_loss_mean"]
           + b["clashes_mean_loss"]
           + w["per_atom_loss_sum"].sum() / (pred_atom_mask.sum() + 1e-6))


def violation_loss(aatype: Tensor, residue_index: Tensor, pred_atom_pos: Tensor,
                   pred_atom_mask: Tensor, res_mask: Tensor, *,
                   clash_overlap_tolerance: float = 1.5,
                   violation_tolerance_factor: float = 2.0,
                   chain_index: Tensor, batch_index: Tensor,
                   per_residue: bool = False) -> tuple[Tensor, dict]:
    """spec §7.10 top-level entry point. `default_vp` always calls this with
    `per_residue=False` - the `True` branch (AlphaFold's per-residue violation
    mask output) and `compute_violation_metrics` are never exercised and are
    not ported (their results are discarded upstream: `violation, _ =
    violation_loss(...)`).
    """
    if per_residue:
        raise NotImplementedError(
            "per_residue=True is AlphaFold's per-residue violation-mask "
            "output; default_vp always calls with per_residue=False (spec "
            "§7.10) and its result is discarded even where it IS computed "
            "upstream, so this branch is not ported.")
    violations = find_structural_violations(
        aatype, residue_index, pred_atom_pos, pred_atom_mask, res_mask,
        clash_overlap_tolerance=clash_overlap_tolerance,
        violation_tolerance_factor=violation_tolerance_factor,
        chain_index=chain_index, batch_index=batch_index)
    loss = custom_structural_violation_loss(violations, pred_atom_mask)
    return loss, violations

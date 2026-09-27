"""``--dssp`` / ``--dssp_mean`` string parsing and sequence decoding.

`notes/specs/06_generation_flags.md` §5.2, §5.4. `parse_num_aa` itself already
lives in `salad_torch.diffusion.schedules` (it predates this spec and matches
it exactly - `(num_aa, resi, chain, is_cyclic, cyclic_mask)`); nothing here
duplicates it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from ..io import residue_constants as rc

__all__ = ["DSSP_CODE", "parse_dssp", "parse_dssp_mean", "random_dssp",
          "random_dssp_mean", "decode_sequence",
          "MultiStateDSSP", "parse_dssp_split"]

#: 0=loop, 1=helix, 2=strand, 3=sentinel/unconditioned. Matches
#: `salad_torch/data/dssp.py`'s own convention.
DSSP_CODE = "LHE_"


# ---------------------------------------------------------------------------
# --dssp_mean
# ---------------------------------------------------------------------------
def random_dssp_mean(max_loop: float = 0.5) -> list[float]:
    """Fresh random ``[loop, helix, sheet]`` fractions. §5.4.1.

    **Not** renormalised to sum to 1 - reproduced faithfully; `Condition`
    consumes this triple as-is (`condition.py`'s `dssp_mean` override branch).
    """
    helix = random.random()
    sheet = 1.0 - helix
    loop = random.random() * max_loop
    return [loop, (1 - loop) * helix, (1 - loop) * sheet]


def parse_dssp_mean(spec: str) -> Tensor | Callable[[], Tensor] | None:
    """``"none"`` -> `None`; ``"random"`` -> a callable drawing a fresh triple
    each call; else ``"loop,helix,sheet"`` -> a fixed `(3,)` tensor, parsed
    with **no renormalisation** - the caller's responsibility that it sums to 1.
    """
    text = spec.strip()
    if text == "none":
        return None
    if text == "random":
        return lambda: torch.tensor(random_dssp_mean(), dtype=torch.float32)
    values = [float(v) for v in text.split(",")]
    if len(values) != 3:
        raise ValueError(
            f"--dssp_mean expects 'none', 'random' or 'loop,helix,sheet', got {spec!r}")
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# --dssp
# ---------------------------------------------------------------------------
def _render_segment(kind: str, length: int, p: float) -> str:
    """One structured/loop segment -> a DSSP-code string of that length. §5.4.1."""
    if kind == "loop":
        return "L" * length if random.random() < p else "_" * length
    letter = "H" if kind == "helix" else "E"
    if random.random() < (1 - p):
        return "_" + letter * (length - 2) + "_"
    return "_" * length


def random_dssp(count: int, p: float = 0.5) -> Tensor:
    """Fresh random per-residue DSSP string of length `count`. §5.4.1.

    A topology-diversity generator, not a target-reproduction tool: helix/sheet
    fractions come from a fresh, *independent* `random_dssp_mean()` draw (its
    `loop` fraction is discarded), so `--dssp random` and `--dssp_mean random`
    on the same design are never actually coupled even though both say
    "random".
    """
    loop_frac, helix_frac, sheet_frac = random_dssp_mean()
    del loop_frac

    helix = int(helix_frac * count)
    if helix < 6:
        helix = 0
    min_h, max_h = 1, helix // 6
    min_h = min(min_h, max_h)
    num_helices = 0 if helix == 0 else (
        min_h if min_h == max_h else random.randrange(min_h, max_h))

    sheet = int(sheet_frac * count)
    if sheet < 8:
        sheet = 0
    min_s, max_s = 2, sheet // 4
    min_s = min(min_s, max_s)
    num_sheets = 0 if sheet == 0 else (
        min_s if min_s == max_s else random.randrange(min_s, max_s))

    loop = count - (helix + sheet)

    helices = [6] * num_helices
    sheets = [4] * num_sheets
    loops = [0] * (num_helices + num_sheets + 1)

    while sum(sheets) < sheet:
        sheets[random.randrange(num_sheets)] += 1
    while sum(helices) < helix:
        helices[random.randrange(num_helices)] += 1
    while sum(loops) < loop:
        loops[random.randrange(len(loops))] += 1

    structured = [("helix", n) for n in helices] + [("sheet", n) for n in sheets]
    random.shuffle(structured)

    pieces = [_render_segment("loop", loops[0], p=0.5)]
    for (kind, n), l in zip(structured, loops[1:]):
        pieces.append(_render_segment(kind, n, p=0.5))
        pieces.append(_render_segment("loop", l, p=0.5))
    dssp_string = "".join(pieces)
    return parse_dssp(dssp_string)


def parse_dssp(spec: str) -> Tensor | Callable[[int], Tensor] | None:
    """``"none"`` -> `None`; ``"random"`` -> a callable ``f(num_aa) -> Tensor``;
    else a literal ``"LHE_..."`` string -> `(len(spec),) int` tensor via
    `DSSP_CODE`.

    A literal string's length silently overrides ``--num_aa`` per §0 step 3 -
    that override is the CLI's job, not this parser's; this function only
    turns the string into the index tensor.
    """
    text = spec.strip()
    if text == "none":
        return None
    if text == "random":
        return random_dssp
    return torch.tensor([DSSP_CODE.index(c) for c in text], dtype=torch.long)


# ---------------------------------------------------------------------------
# --dssp "parent/child_1/child_2" (multi-state design, Lisanza et al. 2024)
# ---------------------------------------------------------------------------
@dataclass
class MultiStateDSSP:
    """Parsed multi-state ``--dssp "parent/child_1/child_2"``.

    Ported from `salad/training/eval_split_change.py::parse_parent`. Upstream
    reuses the plain, single-state `--dssp` flag: a literal value containing
    ``"/"`` is three DSSP strings (parent, child_1, child_2) instead of one,
    separated by ``"/"``.

    Two independent things come out of the one string:

    * `parent` / `child_1` / `child_2`: three ordinary per-state DSSP
      conditioning tensors (`DSSP_CODE`-indexed, case-folded via `.upper()` -
      lower/upper case makes no difference to *this* channel), fed as each
      state's own `dssp_condition` exactly like single-state `--dssp`.
    * `same_mask` (`(len(parent),)` bool): **only** derived from which
      characters in the PARENT string are uppercase (`str.isupper()`; `"_"``
      is never upper, so it can never be marked). Where `True`, the
      corresponding residue is constrained to share the parent's actual
      *tertiary* structure with its child, not just its secondary-structure
      label - the crux of the feature, applied every denoising step by
      `salad_torch.inference.multistate.update_split_change` (kept
      deliberately out of this parsing-only module, matching how `--sym`'s
      string parsing lives here in spirit but its geometry lives in
      `inference/symmetry.py`).

    `parent_split` is `len(child_1)`, **not** `len(parent) // 2` - matching
    upstream literally (`parent_split = len(child_1)`). `p_1 = parent[:split]`
    is child_1's counterpart, `p_2 = parent[split:]` is child_2's. Upstream
    never checks `len(parent) == len(child_1) + len(child_2)`; this parser
    does (see below) because the port's `index_align`-based coupling needs
    `p_1`/`p_2` and their child to have matching lengths for the per-residue
    Kabsch correspondence to even be well-shaped, and a shape-mismatch
    `RuntimeError` several calls downstream (inside `couple_part`) would be a
    far more confusing failure than rejecting it here with the actual
    constraint named.
    """

    parent: Tensor
    child_1: Tensor
    child_2: Tensor
    same_mask: Tensor
    parent_split: int


def parse_dssp_split(spec: str) -> MultiStateDSSP:
    """``"parent/child_1/child_2"`` -> `MultiStateDSSP`. See its docstring.

    Each of the three substrings is parsed exactly like a literal (non-
    ``"none"``/``"random"``) single-state `--dssp` value - `parse_dssp` is
    reused unchanged for that, called on the `.upper()`-folded substring, so
    ``"l"``/``"L"`` and ``"h"``/``"H"``/``"e"``/``"E"`` give the identical
    `dssp_condition` tensor regardless of case; only `same_mask` (below) sees
    the original case.
    """
    parts = spec.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"multi-state --dssp expects exactly 'parent/child_1/child_2' "
            f"(two '/' separators), got {len(parts) - 1} in {spec!r}")
    parent_s, child_1_s, child_2_s = parts

    same_mask = torch.tensor([c.isupper() for c in parent_s], dtype=torch.bool)
    parent = parse_dssp(parent_s.upper())
    child_1 = parse_dssp(child_1_s.upper())
    child_2 = parse_dssp(child_2_s.upper())
    parent_split = len(child_1_s)

    if parent.shape[0] != child_1.shape[0] + child_2.shape[0]:
        raise ValueError(
            f"multi-state --dssp: len(parent)={parent.shape[0]} must equal "
            f"len(child_1)+len(child_2)={child_1.shape[0]}+{child_2.shape[0]}"
            f"={child_1.shape[0] + child_2.shape[0]} - the parent string must "
            f"be a literal concatenation of the two child lengths so its "
            f"N-/C-terminal segments have a 1:1 residue correspondence with "
            f"child_1/child_2 for the tertiary-structure coupling")

    return MultiStateDSSP(parent=parent, child_1=child_1, child_2=child_2,
                          same_mask=same_mask, parent_split=parent_split)


# ---------------------------------------------------------------------------
# --sequence_only
# ---------------------------------------------------------------------------
def decode_sequence(aatype: Tensor | np.ndarray) -> str:
    """``aatype`` -> one-letter sequence string. §5.2.

    Upstream's imported `decode_sequence` (`salad.data.allpdb`) is called with
    an extra, always-true mask argument its own signature doesn't accept - a
    hard arity `TypeError` under `--sequence_only`. Dropping that argument is
    behaviour-preserving (it never masked anything), not just "close enough" -
    see the spec for the confirmed-matching restype ordering.
    """
    table = rc.restypes_with_x
    if torch.is_tensor(aatype):
        aatype = aatype.detach().cpu().tolist()
    else:
        aatype = np.asarray(aatype).tolist()
    return "".join(table[i] for i in aatype)

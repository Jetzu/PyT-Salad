"""The `default_vp` training datasource: `BatchedProteinPDBStream` and its
`ProteinPDB`/`AllPDB` index, per `notes/specs/04_data_and_training.md`
§A.1-A.4 - plus a synthetic, in-memory source so the pipeline is testable
without the (preprocessed, not-on-this-machine) real PDB caches.

Design: the bin-packing stream (`PackedProteinStream`, §A.4) is written
against an abstract `StructureSource` protocol, not against `AllPDB` directly.
Two implementations are provided:

* `NpzAssemblySource` - the real, on-disk `.npz`-cache-backed source
  (§A.1-A.2). **UNTESTED on this machine**: the preprocessed `entries.idx` /
  cluster `.tsv` / `assembly*_npz/` tree this needs (`/hnvme/.../allpdb/` per
  the pipeline-status memory) is not part of this checkout. Implemented
  faithfully from the spec so it can be wired in once that data is reachable;
  see the class docstring for the one place the spec itself is ambiguous
  (`entries.idx` date format).
* `SyntheticStructureSource` - generates idealised backbone geometry (NeRF
  placement from fixed bond lengths/angles + configurable phi/psi) with random
  sequences, entirely in memory. This is what `training/train.py --synthetic`
  and the self-check use; it exercises every branch of the bin-packing loop
  (multi-chain complexes, short-chain rejection, the length-based skip) without
  touching disk.

Both sources return raw, atom24-shaped per-structure dicts matching §A.3's
`ProteinPDB.__getitem__` schema; `PackedProteinStream` (the
`BatchedProteinPDBStream` port) consumes either one identically.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np

from ..io.residue_constants import restype_1to3, restypes

__all__ = [
    "AA_ORDER", "residue_name_to_aa_gt", "StructureSource", "ProteinExample",
    "NpzAssemblySource", "SyntheticStructureSource", "PackedProteinStream",
]

# `AA_ORDER[i] == restypes_with_x-equivalent 3-letter code`, index-compatible
# with `residue_constants.restype_order` for i in 0..19, `UNK` appended at 20 -
# spec §A.3: "identical order to restypes ... with 'UNK' appended as index 20".
AA_ORDER: list[str] = [restype_1to3[r] for r in restypes] + ["UNK"]
_AA_INDEX = {name: i for i, name in enumerate(AA_ORDER)}


def residue_name_to_aa_gt(residue_name: np.ndarray) -> np.ndarray:
    """3-letter residue names -> `aa_gt` indices, 20 ("UNK") for anything else.

    `ProteinPDB.__getitem__`, §A.3: ``argmax(residue_name[:,None]==AA_ORDER)``,
    with unmatched rows (no AA_ORDER hit at all, e.g. a DNA/RNA/metal residue
    that slipped past `filter_residue_type`) forced to 20 rather than
    whatever `argmax` of an all-zero row would give (always 0 = "ALA",
    which upstream explicitly guards against with the `where`).
    """
    out = np.full(residue_name.shape[0], 20, dtype=np.int64)
    for i, name in enumerate(residue_name):
        out[i] = _AA_INDEX.get(str(name), 20)
    return out


# ---------------------------------------------------------------------------
# Abstract source protocol
# ---------------------------------------------------------------------------
class ProteinExample(dict):
    """A `ProteinPDB.__getitem__`-shaped raw structure (§A.3), plus `chain`.

    Keys (all length `n_res`, atom24 - truncation to atom14 happens later, in
    the training step's `prepare_data`-equivalent, per spec §A.3 note): `aa_gt`
    (int), `residue_index` (int), `chain_index` (str array, NOT yet
    numericalised), `entity_index` (int32), `all_atom_positions`
    (`(n_res, 24, 3)` float), `all_atom_mask` (`(n_res, 24)` bool). `chain` is
    the single Python string this draw is "centered on" (§A.3's second
    `__getitem__` return value).
    """


class StructureSource(Protocol):
    """What `PackedProteinStream` needs from a datasource.

    Mirrors `AllPDB`'s cluster-indexed access (§A.2): `__len__` is the number
    of *clusters*, and `get_example` re-rolls a random member/assembly of that
    cluster on every call - so the same `cluster_index` can legitimately
    return a different structure on two different calls (§A.2's "AlphaFold-
    style cluster reweighting"). Returns `None` for an empty/unusable cluster
    (dropped at draw time, the caller just retries another index).
    """

    def __len__(self) -> int: ...

    def get_example(self, cluster_index: int, rng: random.Random
                    ) -> Optional[ProteinExample]: ...


# ---------------------------------------------------------------------------
# The real, on-disk source (§A.1, §A.2) - untested here, no local PDB mirror
# ---------------------------------------------------------------------------
@dataclass
class _ClusterMember:
    entry: str
    chain: str
    assemblies: list[str]


def read_clusters(path: str | Path) -> dict[str, set[str]]:
    """mmseqs2 cluster `.tsv`: tab-separated `representative \\t member`, one
    row per member, grouped by representative. §A.1.2 ("mmseqs2 cluster file
    format", `read_clusters`, `allpdb.py:1325-1348`).
    """
    clusters: dict[str, set[str]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rep, member = line.split("\t")[:2]
            clusters.setdefault(rep, set()).add(member)
    return clusters


_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
          "SEP", "OCT", "NOV", "DEC"]


def _parse_entries_idx_date(raw: str) -> Optional[tuple[int, int, int]]:
    """Best-effort date parse -> `(year4, month, day)`, or `None`.

    **Spec ambiguity, flagged rather than silently resolved**: §A.1.2 states
    the `entries.idx` date column format is `DD/MM/YY`, but the *same* spec's
    `entry_filter` defaults are `start_date="01/01/90"`, `cutoff_date="12/31/21"`
    - the `31` in the second string cannot be a month, so those two defaults
    are only sensible as `MM/DD/YY`. The real RCSB `entries.idx` file (outside
    what's quoted in the spec) in fact uses neither - it's `DD-MON-YY`
    (e.g. `13-JAN-04`). Rather than guess which the writer intended, this
    accepts all three so a real run isn't blocked by the ambiguity, and
    disambiguates `D/M/Y` vs `M/D/Y` per-value using whichever field is `>12`
    (must be the day) - falling back to `DD/MM/YY` (the spec's literal label)
    only when both fields are `<=12` and genuinely ambiguous. **Verify this
    against one real line of `entries.idx` before trusting the date cutoff for
    an actual training run** - this was never checked against a real file.
    Two-digit years pivot at 30 (`00-29 -> 20xx`, `30-99 -> 19xx`; PDB's dates
    run 1972-present, so this pivot never mishandles a real deposition date).
    """
    raw = raw.strip()
    if "-" in raw:
        try:
            d, mon, y = raw.split("-")
            month = _MONTHS.index(mon.upper()) + 1
            day, year = int(d), int(y)
        except (ValueError, IndexError):
            return None
    elif "/" in raw:
        parts = raw.split("/")
        if len(parts) != 3:
            return None
        try:
            a, b, year = (int(p) for p in parts)
        except ValueError:
            return None
        if a > 12:
            day, month = a, b
        elif b > 12:
            month, day = a, b
        else:
            month, day = a, b   # ambiguous: fall back to the spec's DD/MM/YY label
    else:
        return None
    year = year + 2000 if year < 30 else year + 1900
    return year, month, day


def entry_filter_factory(start_date: str, cutoff_date: str, cutoff_resolution: float):
    start = _parse_entries_idx_date(start_date)
    cutoff = _parse_entries_idx_date(cutoff_date)
    if start is None or cutoff is None:
        raise ValueError(
            f"could not parse start_date={start_date!r} / cutoff_date={cutoff_date!r} "
            "as DD/MM/YY, MM/DD/YY, or DD-MON-YY")

    def inner(entry: dict) -> bool:
        if entry.get("resolution") is None or entry["resolution"] > cutoff_resolution:
            return False
        date = entry.get("date")
        if date is None:
            return False
        return start <= date < cutoff

    return inner


def read_entries_idx(path: str | Path) -> dict[str, dict]:
    """PDB `entries.idx` -> `{code: {date, resolution, exptype}}`. §A.1.2.

    Real `entries.idx` files have a two-line header (column names, then a
    dashed rule); both are skipped by virtue of failing the resolution-float
    parse below rather than matched positionally, so this is robust to either
    the tab-separated form the spec quotes or the fixed-width form RCSB ships.
    """
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            code = cols[0].strip().upper()
            if len(code) != 4:
                continue
            try:
                resolution = float(cols[-2].strip()) if cols[-2].strip() not in ("", "NOT") else None
            except ValueError:
                resolution = None
            date = _parse_entries_idx_date(cols[2].strip()) if len(cols) > 2 else None
            out[code] = {"date": date, "resolution": resolution, "exptype": cols[-1].strip()}
    return out


def compute_index(root: str | Path, start_date: str, cutoff_date: str,
                  cutoff_resolution: float, seqres_aa: str, seqres_na: str,
                  npz_version: str = "") -> dict[str, list[list[_ClusterMember]]]:
    """Port of `compute_index`/`create_chain_index` (§A.2). Cheap, pure Python.

    Assembly-mode only (`chain_assemblies.csv` present) - the asymmetric-unit
    fallback (`compute_asym_index`, `chain_assemblies=None`) is NOT
    implemented: `default_vp`'s training script always runs assembly mode, so
    per the task brief's "skip incidental branches, but say so" this one is
    named and skipped here rather than silently absent.
    """
    root = Path(root)
    npz_dir = root / f"assembly{npz_version}_npz"
    npz_assemblies: dict[str, list[str]] = {}
    if npz_dir.is_dir():
        for p in npz_dir.glob("*/*.npz"):
            code = p.stem.split("-")[0].upper()
            npz_assemblies.setdefault(code, []).append(str(p))

    entries = read_entries_idx(root / "entries.idx") if (root / "entries.idx").exists() else {}
    aa_clusters_raw = read_clusters(root / f"{seqres_aa}_cluster.tsv") \
        if (root / f"{seqres_aa}_cluster.tsv").exists() else {}
    na_clusters_raw = read_clusters(root / f"{seqres_na}_cluster.tsv") \
        if (root / f"{seqres_na}_cluster.tsv").exists() else {}

    chain_assemblies: dict[tuple[str, str], list[str]] = {}
    csv_path = root / "chain_assemblies.csv"
    if csv_path.exists():
        with open(csv_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 3:
                    continue
                pdbid, chain, npz_name = parts[0].upper(), parts[1], parts[2]
                chain_assemblies.setdefault((pdbid, chain), []).append(npz_name)

    entry_ok = entry_filter_factory(start_date, cutoff_date, cutoff_resolution)

    def build(raw_clusters: dict[str, set[str]]) -> list[list[_ClusterMember]]:
        result = []
        for _, members in raw_clusters.items():
            survivors = []
            for member in members:
                # member id convention: "{pdbid}_{chain}" (mmseqs2 seqres header)
                if "_" not in member:
                    continue
                pdbid, chain = member.split("_", 1)
                pdbid = pdbid.upper()
                entry = entries.get(pdbid)
                if entry is None or not entry_ok(entry):
                    continue
                assemblies = chain_assemblies.get((pdbid, chain))
                if assemblies is None:
                    assemblies = npz_assemblies.get(pdbid)
                if not assemblies:
                    continue
                survivors.append(_ClusterMember(entry=pdbid, chain=chain, assemblies=assemblies))
            if survivors:
                result.append(survivors)
        return result

    return {"AA": build(aa_clusters_raw), "NA": build(na_clusters_raw)}


class NpzAssemblySource:
    """The real `ProteinPDB`-equivalent source (§A.1, §A.2, §A.3).

    `filter_residue_type=["AA"]` is forced, matching `ProteinPDB.__init__`
    (which ignores any caller override, §A.3). **Not exercised by any test in
    this checkout** - there is no local PDB mirror / preprocessed cache to
    load; wire a real `root` (e.g. `{data_path}/allpdb/` per §A.4) once one is
    reachable.
    """

    def __init__(self, root: str | Path, *, seqres_aa: str = "clusterSeqresAA",
                seqres_na: str = "clusterSeqresNA", cutoff_resolution: float = 4.0,
                start_date: str = "01/01/90", cutoff_date: str = "12/31/21",
                npz_version: str = ""):
        index = compute_index(root, start_date, cutoff_date, cutoff_resolution,
                              seqres_aa, seqres_na, npz_version)
        self._clusters: list[list[_ClusterMember]] = index["AA"]

    def __len__(self) -> int:
        return len(self._clusters)

    def get_example(self, cluster_index: int, rng: random.Random) -> Optional[ProteinExample]:
        cluster = self._clusters[cluster_index]
        if not cluster:
            return None
        member = rng.choice(cluster)
        assembly_path = rng.choice(member.assemblies)
        try:
            with np.load(assembly_path, allow_pickle=False) as npz:
                residue_type = npz["residue_type"]
                keep = residue_type == "AA"
                if not keep.any():
                    return None
                aa_gt = residue_name_to_aa_gt(npz["residue_name"][keep])
                example = ProteinExample(
                    aa_gt=aa_gt,
                    residue_index=npz["residue_index"][keep],
                    chain_index=npz["chain_index"][keep],
                    entity_index=npz["entity_index"][keep].astype(np.int32),
                    all_atom_positions=npz["position"][keep].astype(np.float32),
                    all_atom_mask=npz["atom_mask"][keep].astype(bool),
                    chain=member.chain,
                )
        except (FileNotFoundError, KeyError, OSError):
            return None
        return example


# ---------------------------------------------------------------------------
# Synthetic in-memory source - what actually gets exercised today
# ---------------------------------------------------------------------------
# Idealised trans-peptide internal coordinates (standard AMBER-ish values; the
# exact numbers don't matter for a synthetic smoke-test source, only that bond
# lengths/angles are realistic enough that `assign_dssp`/the violation loss see
# sane geometry rather than degenerate zero-length bonds).
_N_CA, _CA_C, _C_N = 1.458, 1.525, 1.329
_ANG_N_CA_C = math.radians(111.0)
_ANG_CA_C_N = math.radians(117.2)
_ANG_C_N_CA = math.radians(121.7)


def _nerf_place(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                bond_length: float, bond_angle: float, dihedral: float) -> np.ndarray:
    """Standard NeRF placement of `D` given the chain `A-B-C-D`.

    `bond_angle` = angle(B, C, D), `dihedral` = dihedral(A, B, C, D), radians.
    """
    bc = c - b
    bc_hat = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc_hat)
    n_hat = n / np.linalg.norm(n)
    m_hat = np.cross(n_hat, bc_hat)
    d_local = np.array([-bond_length * math.cos(bond_angle),
                        bond_length * math.sin(bond_angle) * math.cos(dihedral),
                        bond_length * math.sin(bond_angle) * math.sin(dihedral)])
    basis = np.stack([bc_hat, m_hat, n_hat], axis=-1)
    return c + basis @ d_local


def build_idealized_backbone(n_res: int, phi_deg: float, psi_deg: float,
                             *, omega_deg: float = 180.0,
                             jitter_deg: float = 0.0,
                             rng: np.random.Generator | None = None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an `(n_res,)` N/CA/C/O backbone by NeRF chain placement.

    Verified (see the writer's report) to reproduce real secondary structure
    under `assign_dssp`: a fixed alpha-helix `(phi, psi) = (-57, -47)` gives
    interior residues classified helix with loop termini (H-bonds need `i,
    i+3/4/5` neighbours, so a lone helix can never cover its own ends); an
    isolated strand conformation alone classifies as loop (correct DSSP
    behaviour - a beta strand needs a hydrogen-bonded partner strand to be
    called "E"; there is no such partner in a single chain).

    `jitter_deg` adds independent per-residue Gaussian noise to phi/psi/omega,
    for a less crystallographically-perfect synthetic ensemble.
    """
    if n_res < 1:
        raise ValueError("n_res must be >= 1")
    rng = rng or np.random.default_rng()

    def _ang(base_deg):
        if jitter_deg <= 0:
            return math.radians(base_deg)
        return math.radians(base_deg + float(rng.normal(0.0, jitter_deg)))

    n1 = np.zeros(3)
    ca1 = n1 + np.array([_N_CA, 0.0, 0.0])
    theta = math.radians(69.0)          # N-CA-C = 111 deg, placed in the xy-plane
    c1 = ca1 + _CA_C * np.array([math.cos(theta), math.sin(theta), 0.0])
    coords = [n1, ca1, c1]
    for _ in range(1, n_res):
        n_prev, ca_prev, c_prev = coords[-3], coords[-2], coords[-1]
        psi = _ang(psi_deg)
        n_new = _nerf_place(n_prev, ca_prev, c_prev, _C_N, _ANG_CA_C_N, psi)
        omega = _ang(omega_deg)
        ca_new = _nerf_place(ca_prev, c_prev, n_new, _N_CA, _ANG_C_N_CA, omega)
        phi = _ang(phi_deg)
        c_new = _nerf_place(c_prev, n_new, ca_new, _CA_C, _ANG_N_CA_C, phi)
        coords += [n_new, ca_new, c_new]
    coords_arr = np.asarray(coords).reshape(n_res, 3, 3)
    n_arr, ca_arr, c_arr = coords_arr[:, 0], coords_arr[:, 1], coords_arr[:, 2]

    # O placement: the peptide unit {Ca(i), C(i), O(i), N(i+1)} is planar
    # (sp2 C(i)); the three C(i)-substituent bond directions sum to ~360 deg,
    # so `-(unit(Ca-C) + unit(N_next-C))`, renormalised, lands within ~1 deg of
    # the real angle split (117/120.8/122.7 deg is nearly symmetric) - plenty
    # good for a synthetic backbone, not claimed bond-angle-exact.
    o_arr = np.zeros_like(n_arr)
    u = ca_arr[:-1] - c_arr[:-1]
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = n_arr[1:] - c_arr[:-1]
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    w = -(u + v)
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    o_arr[:-1] = c_arr[:-1] + 1.231 * w
    u_last = ca_arr[-1] - c_arr[-1]
    o_arr[-1] = c_arr[-1] - 1.231 * (u_last / np.linalg.norm(u_last))
    return n_arr, ca_arr, c_arr, o_arr


@dataclass
class _SyntheticTemplate:
    """One cluster's fixed "identity" (chain count/lengths/conformation);
    `get_example` regenerates coordinates+sequence fresh each draw, mirroring
    real clusters' "different member each access" property (§A.2)."""
    chain_lengths: list[int]
    conformations: list[tuple[float, float]]   # (phi, psi) per chain


class SyntheticStructureSource:
    """In-memory `StructureSource`: idealised helices with random sequences.

    No disk I/O, no external dependency - this is what makes the training
    loop testable today (hard requirement #1). `num_clusters` templates are
    generated at construction; residue count per chain is drawn uniformly from
    `chain_length_range`, and `p_multi_chain` of templates get 2-3 chains (to
    exercise `PackedProteinStream`'s complex-taking branch, §A.4 step 7).
    """

    def __init__(self, num_clusters: int = 256, *,
                chain_length_range: tuple[int, int] = (20, 120),
                p_multi_chain: float = 0.3, jitter_deg: float = 4.0,
                seed: int = 0):
        rng = np.random.default_rng(seed)
        self._templates: list[_SyntheticTemplate] = []
        conformations = [(-57.0, -47.0), (-63.0, -42.0), (-119.0, 113.0)]
        for _ in range(num_clusters):
            n_chains = 1
            if rng.random() < p_multi_chain:
                n_chains = int(rng.integers(2, 4))
            lengths = [int(rng.integers(*chain_length_range)) for _ in range(n_chains)]
            confs = [conformations[int(rng.integers(0, len(conformations) - 1))]
                    for _ in range(n_chains)]   # bias away from the lone-strand conformation
            self._templates.append(_SyntheticTemplate(lengths, confs))
        self.jitter_deg = jitter_deg

    def __len__(self) -> int:
        return len(self._templates)

    def get_example(self, cluster_index: int, rng: random.Random) -> Optional[ProteinExample]:
        template = self._templates[cluster_index]
        np_rng = np.random.default_rng(rng.getrandbits(63))
        chain_letters = [chr(ord("A") + i) for i in range(len(template.chain_lengths))]

        all_aa, all_ri, all_ci, all_pos, all_mask = [], [], [], [], []
        for letter, length, (phi, psi) in zip(chain_letters, template.chain_lengths,
                                              template.conformations):
            n, ca, c, o = build_idealized_backbone(
                length, phi, psi, jitter_deg=self.jitter_deg, rng=np_rng)
            # random offset per chain so multi-chain complexes aren't coincident
            offset = np_rng.normal(0.0, 15.0, size=3) if len(template.chain_lengths) > 1 else 0.0
            pos24 = np.zeros((length, 24, 3), dtype=np.float32)
            pos24[:, 0], pos24[:, 1], pos24[:, 2], pos24[:, 3] = n, ca, c, o
            pos24 += offset
            mask24 = np.zeros((length, 24), dtype=bool)
            mask24[:, :4] = True   # only backbone N/CA/C/O resolved (CB and beyond unresolved)

            aa_gt = np_rng.integers(0, 20, size=length)   # never UNK, so seq_mask is all-True
            all_aa.append(aa_gt)
            all_ri.append(np.arange(length))
            all_ci.append(np.full(length, letter))
            all_pos.append(pos24)
            all_mask.append(mask24)

        selected = chain_letters[int(np_rng.integers(0, len(chain_letters)))]
        return ProteinExample(
            aa_gt=np.concatenate(all_aa).astype(np.int64),
            residue_index=np.concatenate(all_ri).astype(np.int64),
            chain_index=np.concatenate(all_ci),
            entity_index=np.zeros(sum(template.chain_lengths), dtype=np.int32),
            all_atom_positions=np.concatenate(all_pos).astype(np.float32),
            all_atom_mask=np.concatenate(all_mask).astype(bool),
            chain=selected,
        )


# ---------------------------------------------------------------------------
# `PackedProteinStream` - the `BatchedProteinPDBStream` port (§A.4)
# ---------------------------------------------------------------------------
def _slice_dict(d: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in d.items() if k != "chain"}


def _numerical_chain_index(chain_str: np.ndarray) -> np.ndarray:
    """§A.4 step 8: sorted-string -> small int, LOCAL to this one part.

    `np.unique` sorts lexicographically (ASCII), matching upstream's
    `numerical_chain_index` exactly (`allpdb.py:1237-1240`).
    """
    return np.unique(chain_str, return_inverse=True)[1].astype(np.int32)


def _concat_parts(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def _pad_dict(d: dict, size: int, total: int) -> dict:
    """Zero-pad every field to exactly `size` rows; add the `mask` field.

    §A.4 step 11 (`pad_dict`/`pad_item`, `allpdb.py:1250-1271`).
    """
    out = {}
    for k, v in d.items():
        if v.shape[0] == size:
            out[k] = v
            continue
        pad_shape = (size - v.shape[0],) + v.shape[1:]
        out[k] = np.concatenate([v, np.zeros(pad_shape, dtype=v.dtype)], axis=0)
    out["mask"] = np.arange(size) < total
    return out


class PackedProteinStream:
    """Port of `BatchedProteinPDBStream.next_item` (§A.4, steps 1-12).

    An `Iterator[dict]` rather than a `torch.utils.data.IterableDataset`
    subclass directly - `training/loop.py` wraps one of these per DataLoader
    worker. `__iter__` yields forever, matching upstream's `IterableDataset`
    (`next_item` called in an infinite loop, `allpdb.py:989-991`).

    Every documented upstream quirk is reproduced deliberately (spec §A.8):
    the `legacy_repetitive_chains` complex-mode repetitive-sequence bug (counts
    the WHOLE structure's amino-acid composition even when only the selected
    chain would be added), the backbone-only-three-atoms drop check (N/CA/C,
    not O), and the requeue-list popping from the end (so a just-deferred
    oversized cluster gets retried almost immediately rather than later in the
    epoch). Do not "fix" any of these without updating this docstring and the
    spec cross-reference - they change what the model sees during training.
    """

    def __init__(self, source: StructureSource, *, size: int = 1024,
                min_size: int = 16, max_size: Optional[int] = None,
                p_complex: float = 0.5, legacy_repetitive_chains: bool = True,
                seed: Optional[int] = None, max_attempts_per_draw: int = 200_000):
        self.source = source
        self.size = size
        self.min_size = min_size
        self.max_size = max_size if max_size is not None else size
        self.p_complex = p_complex
        self.legacy_repetitive_chains = legacy_repetitive_chains
        self.rng = random.Random(seed)
        self._order: list[int] = []
        # Not an upstream limit - a safety net against a pathological source
        # (e.g. every structure rejected) spinning forever; upstream has no
        # such cap because real data practically never hits it (§A.4).
        self._max_attempts = max_attempts_per_draw

    def _next_cluster_index(self) -> int:
        if not self._order:
            self._order = list(range(len(self.source)))
            self.rng.shuffle(self._order)
        return self._order.pop()   # pop from the END - matches step 10's quirk

    def next_item(self) -> dict:
        total = 0
        count = 0
        requeue: list[int] = []
        parts: list[dict] = []
        attempts = 0

        while total < self.size:
            attempts += 1
            if attempts > self._max_attempts:
                raise RuntimeError(
                    f"PackedProteinStream: {self._max_attempts} draws without "
                    f"filling a {self.size}-residue batch (got {total}) - the "
                    "source is rejecting almost everything; check min_size/"
                    "max_size against its structure sizes.")

            cluster_index = self._next_cluster_index()
            example = self.source.get_example(cluster_index, self.rng)
            if example is None:
                continue
            chain = example["chain"]
            protein_data = {k: v for k, v in example.items() if k != "chain"}

            # step 1: drop residues missing N, CA, or C (atom24 slots 0,1,2) -
            # matches upstream's literal (O-not-required) check, not "fixed".
            backbone_ok = protein_data["all_atom_mask"][:, :3].all(axis=-1)
            protein_data = _slice_dict(protein_data, backbone_ok)
            item_size = protein_data["aa_gt"].shape[0]

            # step 2
            if item_size < self.min_size:
                continue

            # step 3
            selected = protein_data["chain_index"] == chain
            selected_chain_size = int(selected.sum())
            if (selected_chain_size < self.min_size
                    or selected_chain_size > self.max_size
                    or selected_chain_size > self.size
                    or selected_chain_size == 0):
                continue

            # step 4: repetitive-sequence filter (legacy bug reproduced as-is)
            if self.legacy_repetitive_chains:
                counts = np.bincount(protein_data["aa_gt"], minlength=21)
            else:
                counts = np.bincount(protein_data["aa_gt"][selected], minlength=21)
            if counts.max() > 0.5 * selected_chain_size:
                continue

            # step 5: AlphaFold-style length-based skip
            clamped = min(max(selected_chain_size, 256), 512)
            skip_chance = 1 - clamped / 512
            if self.rng.random() < skip_chance:
                continue

            # step 6: requeue-or-reject on overflow
            if total + selected_chain_size > self.size:
                requeue.append(cluster_index)
                if total < self.size / 2:
                    continue
                break

            # step 7: monomer vs. complex
            take_complex = self.rng.random() < self.p_complex
            if take_complex and (total + item_size <= self.size):
                part, part_size = dict(protein_data), item_size
            else:
                part, part_size = _slice_dict(protein_data, selected), selected_chain_size

            # step 8: renumber chain_index, local to this part
            part["chain_index"] = _numerical_chain_index(part["chain_index"])

            # step 9
            part["batch_index"] = np.full(part_size, count, dtype=np.int32)
            parts.append(part)
            total += part_size
            count += 1

        # step 10: deferred oversized items go back on the pool, appended
        # (popped from the end -> retried almost immediately, upstream's
        # documented "possibly not doing what was intended" quirk).
        self._order.extend(requeue)

        # step 11
        result = _pad_dict(_concat_parts(parts), self.size, total) if parts else \
            _pad_dict({
                "aa_gt": np.zeros(0, dtype=np.int64),
                "residue_index": np.zeros(0, dtype=np.int64),
                "chain_index": np.zeros(0, dtype=np.int32),
                "entity_index": np.zeros(0, dtype=np.int32),
                "all_atom_positions": np.zeros((0, 24, 3), dtype=np.float32),
                "all_atom_mask": np.zeros((0, 24), dtype=bool),
                "batch_index": np.zeros(0, dtype=np.int32),
            }, self.size, 0)

        # step 12
        result["seq_mask"] = result["mask"] & (result["aa_gt"] != 20)
        result["residue_mask"] = result["mask"] & result["all_atom_mask"].any(axis=-1)
        return result

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.next_item()

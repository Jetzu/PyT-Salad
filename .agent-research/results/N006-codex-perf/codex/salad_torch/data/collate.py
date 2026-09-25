"""Concatenating collate for `PackedProteinStream` draws.

**Padding is not safe** (`notes/04_PARITY_AND_PERF.md` §B1, spec §A.8): salad's
segment ops (`index_mean`, `index_sum`, `extract_neighbours`, ...) all expect a
single flat `(N,)`-indexed array with a companion `batch_index`/`mask` - not a
`(B, N, ...)` tensor with a real leading batch dimension. Introducing one would
silently change every downstream shape-dependent op's semantics. So:
`torch.utils.data.default_collate`'s implicit `torch.stack` is the WRONG
collate here on purpose - use `concat_collate` below, which reproduces
upstream's `np_collate` (`flexloop/data.py:91-116`): `np.concatenate`, never
`np.stack`.

For gradient accumulation / multi-worker throughput, several `PackedProteinStream`
draws (each already a fixed-`size`-length flat array with its OWN local
`batch_index`, starting at 0) get concatenated into one longer flat array. The
caller (`training/loop.py`) MUST split that back into per-draw chunks (by
reshape - each chunk is exactly `size` residues) before running a forward pass,
and must process each chunk with its own local `batch_index` - never merge two
draws' `batch_index` spaces into one segment-op namespace (spec §0f: a
`batch_index` collision across draws would silently fuse unrelated proteins).
`split_chunks` below is that inverse operation.

For a single-GPU port with `rebatch=accumulate=NUM_DEVICES=1` (the actual
`default_vp` defaults), none of this machinery is exercised: one dataset draw
*is* one training step, and `training/loop.py` skips the DataLoader batch
dimension entirely for that common case (spec §A.8's own recommendation).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = ["concat_collate", "split_chunks", "example_to_tensors"]


def concat_collate(items: Sequence[dict]) -> dict:
    """`np.concatenate`-equivalent collate: concatenates every dict field along
    axis 0. Fields missing from any item are dropped (matches upstream's
    `all(key in b for b in items)` guard, `flexloop/data.py:96`).
    """
    if not items:
        return {}
    common_keys = [k for k in items[0] if all(k in item for item in items)]
    out = {}
    for key in common_keys:
        values = [item[key] for item in items]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = np.concatenate(values, axis=0)
    return out


def split_chunks(batch: dict, chunk_size: int) -> list[dict]:
    """Inverse of `concat_collate`: split one `(K*chunk_size, ...)` flat dict
    back into `K` independent `chunk_size`-length dicts, each with `batch_index`
    left exactly as it was drawn (i.e. still local/starting-at-0 per chunk,
    since `PackedProteinStream` never produces cross-chunk batch ids and
    `concat_collate` only concatenates, never renumbers).

    Raises if the leading dimension isn't an exact multiple of `chunk_size` -
    a silent partial chunk would be exactly the "algebraically mixed" hazard
    spec §0f warns against.
    """
    total = next(iter(batch.values())).shape[0]
    if total % chunk_size != 0:
        raise ValueError(f"batch of {total} rows is not a multiple of "
                         f"chunk_size={chunk_size}; did a draw get truncated?")
    n_chunks = total // chunk_size
    return [{k: v[i * chunk_size:(i + 1) * chunk_size] for k, v in batch.items()}
           for i in range(n_chunks)]


# ---------------------------------------------------------------------------
# numpy (dataset output) -> torch (model input)
# ---------------------------------------------------------------------------
#: dtype policy mirrors `train_noise_schedule_benchmark.py`'s explicit
#: `jnp.float32` cast of incoming data (spec §B.5) and the segment ops'
#: int64 requirement (`geometry/segment.py`'s `_normalise_index`).
_INT_KEYS = {"aa_gt", "residue_index", "chain_index", "entity_index", "batch_index"}
_BOOL_KEYS = {"all_atom_mask", "mask", "seq_mask", "residue_mask"}


def example_to_tensors(example: dict, *, device: torch.device | str | None = None) -> dict:
    """One `PackedProteinStream` draw (numpy) -> the torch dict a training
    step consumes. Plain per-key dtype coercion; no reshaping or masking
    logic lives here (that's `training/loop.py`'s `prepare_batch`, which is
    training-loss-shaped, not collation-shaped).
    """
    out: dict[str, Tensor] = {}
    for key, value in example.items():
        if key in _BOOL_KEYS:
            out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.bool)
        elif key in _INT_KEYS:
            out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.long)
        else:
            out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.float32)
    return out

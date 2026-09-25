"""The `default_vp` training step and loop - replaces `flexloop`+`optax`.

Port of `train_noise_schedule_benchmark.py` per
`notes/specs/04_data_and_training.md` Part B, for a single-process (optionally
DDP) torch run. What's genuinely needed vs. incidental cluster machinery is
summarised in the spec's §B.7; the short version, followed here:

* plain `Adam(betas=(0.9, 0.99), eps=1e-9)` + global-norm clip `0.1` +
  linear-warmup-then-cosine-decay LR (§B.2-B.3) - no weight decay, ever;
* EMA of the parameters, decay `0.999`, one step stale by construction (§B.4);
* periodic checkpointing (§B.5) so a multi-day run survives preemption;
* `rebatch`/`accumulate`/`multigpu`/`pmap` collapse to "one dataset draw = one
  training step" for `rebatch=accumulate=NUM_DEVICES=1` (upstream's actual
  defaults) - `accumulate > 1` is still supported below (as `K` independent
  forward/backward calls on `K` separately-local-`batch_index` draws, gradients
  averaged, per spec §0f/§B.5), but `rebatch`/`multigpu`/`pmap` themselves are
  NOT ported - a normal single-process loop, optionally wrapped in
  `DistributedDataParallel(model, find_unused_parameters=True)` by the caller,
  covers the same ground with far less machinery (spec §B.5: the
  `find_unused_parameters=True` flag is NOT optional for this model - 202
  `.b` bias tensors get `grad=None` every step, per
  `notes/04_PARITY_AND_PERF.md` §4).

FORMER GAP, now closed (2026-09-25): `prepare_batch` used to stand in for the
`Encoder` network (which builds the ground-truth 20-atom "augmented" position
tensor from real atom14 structures, `StructureDiffusion.encode`, spec-03 §1.2
step 3) with a deterministic, non-learned placeholder that replicated the
idealised CB point into all 15 augment slots. `salad_torch/modules/encoder.py`
now exists and loads `strict=True` from all four released checkpoints, so
`prepare_batch` calls a real `Encoder(config)` instead (spec-05 "call site"
section). Its weights are jointly optimised with the rest of the model - see
`Trainer`/`training_step` below, which fold `encoder.parameters()` into the
same optimizer/EMA/checkpoint as `model.parameters()`, exactly as spec-05
demands ("do not treat it as a frozen preprocessing step"): upstream carries
both under one Haiku param tree (`structure_diffusion/~encode/encoder/...`
alongside `structure_diffusion/diffusion/...`), and the released checkpoint
already contains the encoder leaves (`config.py`'s "unused for de-novo
generation, but its weights are in the file" comment) - de-novo *sampling*
just never calls it, since there is no atom14 ground truth to encode then.

SECOND FORMER GAP, now closed (2026-09-25): the exact stochastic recipe NSB's
`prepare_condition` uses to decide which residues get their true amino acid /
secondary structure REVEALED to the model as a hard condition during training
was not fully specified in specs 02/03/04 (spec-03 §1.3 explicitly stops
short, citing "prepare_condition, not part of the loss") - `reveal_conditioning`
used to be a plausible-but-approximate best-effort reconstruction. It is now
transcribed directly from `Condition.get_masks` (NSB:2562-2586) and the outer
per-item gate in `StructureDiffusion.prepare_condition` (NSB:241-258); see
that function's docstring for the three concrete ways the old version
differed (uniform range, the two-independent-gates structure, and the
previously entirely-missing `dssp_mean` channel).
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor, nn

from ..data.dssp import assign_dssp
from ..geometry.frames import compute_pseudo_cb, positions_to_ncacocb
from ..geometry.segment import index_mean, unique_chain
from ..diffusion.schedules import sigma_scale_cosine
from ..losses.total import total_loss
from ..modules.encoder import Encoder
from ..modules.model import apply_diffusion
from .ema import EMA
from .schedule import cosine_decay_schedule, warmup_schedule

__all__ = ["TrainConfig", "prepare_batch", "reveal_conditioning",
          "training_step", "save_checkpoint", "load_checkpoint", "Trainer"]


# ---------------------------------------------------------------------------
# Config (spec §B.1's CLI table, real `bool`s - not upstream's string-compared
# `"True"`/`"False"` argparse idiosyncrasy, spec §B.1's own footnote on that)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # -- data pipeline (§A.4; forwarded to `PackedProteinStream`) -----------
    num_aa: int = 1024                 # `size`: residues per draw
    min_size: int = 16
    max_size: Optional[int] = None     # None -> num_aa, matching default_vp
    p_complex: float = 0.5
    legacy_repetitive_chains: bool = True

    # -- optimizer / schedule (§B.2-B.3) -------------------------------------
    lr: float = 1e-3
    decay_lr: float = 1e-5
    warmup_steps: int = 1000
    decay_steps: int = 500_000
    clip: float = 0.1
    b1: float = 0.9
    b2: float = 0.99                   # NOT the Adam-typical 0.999
    adam_eps: float = 1e-9
    schedule_fixed: bool = False        # True -> `warmup_schedule` (flat after warmup)

    # -- EMA (§B.4) -----------------------------------------------------------
    ema_weight: float = 0.999

    # -- gradient accumulation (§0f/§B.5; 1 = default_vp's actual setting) ---
    accumulate: int = 1

    # -- checkpointing (§B.5) -------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000     # steps
    save_interval_seconds: float = 600.0

    # -- misc -----------------------------------------------------------------
    seed: int = 42
    max_steps: Optional[int] = None     # None -> warmup_steps + decay_steps + 1
    log_interval: int = 1

    def __post_init__(self) -> None:
        if self.max_size is None:
            self.max_size = self.num_aa
        if self.max_steps is None:
            self.max_steps = self.warmup_steps + self.decay_steps + 1

    def build_schedule(self) -> Callable[[int], float]:
        if self.schedule_fixed:
            return warmup_schedule(self.lr, self.warmup_steps)
        return cosine_decay_schedule(self.lr, self.decay_lr, self.warmup_steps,
                                     self.decay_steps)


# ---------------------------------------------------------------------------
# Conditioning reveal - transcribed from the actual upstream source
# (`Condition.get_masks`, NSB:2562-2586, and the outer per-item gate in
# `StructureDiffusion.prepare_condition`, NSB:241-258), not reconstructed from
# spec-03 alone (which explicitly stops short of this - see module docstring's
# former "second, smaller gap" note, now closed).
# ---------------------------------------------------------------------------
def _mask_me(group_index: Tensor, num_groups: int, *, p: float, p_min: float,
            p_max: float, generator: torch.Generator | None = None) -> Tensor:
    """Port of `Condition.get_masks`'s inner `mask_me` closure (NSB:2570-2578).

    Three INDEPENDENT draws, exactly as upstream (three separate
    `hk.next_rng_key()` calls there, three separate `torch.rand` calls here):
    a per-GROUP reveal rate ``p_mask ~ U(p_min, p_max)``, a per-RESIDUE
    Bernoulli at that rate (``bare_mask``), and a per-GROUP coinflip at
    probability ``p`` (``keep_mask``, upstream's "global probability" gate -
    NOT the same gate as `prepare_condition`'s own outer `use_condition`
    below, which is a SEPARATE, later multiplication). `group_index` is
    `batch_index` for the aa/dssp channels and `chain_index` for dssp_mean -
    upstream uses different grouping axes for the two (NSB:2582-2584).
    """
    device = group_index.device
    p_mask_g = p_min + (p_max - p_min) * torch.rand(
        num_groups, generator=generator, device=device)
    p_mask = p_mask_g[group_index]
    bare_mask = torch.rand(group_index.shape[0], generator=generator,
                           device=device) < p_mask
    keep_mask_g = torch.rand(num_groups, generator=generator, device=device) < p
    keep_mask = keep_mask_g[group_index]
    return bare_mask & keep_mask


def reveal_conditioning(aa_gt: Tensor, dssp_gt: Tensor, mask: Tensor,
                        chain_index: Tensor, batch_index: Tensor, *,
                        generator: torch.Generator | None = None
                        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """`Condition.get_masks` + `StructureDiffusion.prepare_condition`'s outer
    gate, transcribed (NSB:2562-2586, 241-258) rather than approximated.

    Three channels, not two - the PRIOR version of this function had no
    `dssp_mean` (aggregate secondary-structure-composition) conditioning at
    all, so training never exercised that pathway. What actually differed
    from the old best-effort reconstruction (see git history / N004 report):

    1. The per-item reveal rate is ``U(0.2, 1.0)`` for aa/dssp (NEVER below
       20% local coverage when used), not ``U(0, 1)`` - the old version could
       draw a ~0% rate and reveal nothing even when "in use".
    2. There are TWO independent per-batch-item p=0.5 gates multiplied
       together, not one: `Condition.get_masks`'s own `keep_mask` (inside
       `mask_me`, one draw per channel) AND `prepare_condition`'s outer
       `use_condition` (one draw, SHARED by both aa_mask and dssp_mask). Net
       effect: conditioning survives with probability 0.5*0.5 = 0.25 per
       batch item, not 0.5, and aa/dssp are USE-CORRELATED (whether a batch
       item gets any conditioning at all is the SAME coinflip for both
       channels) - the old version drew that coinflip independently per
       channel.
    3. `dssp_mean` conditioning (grouped by CHAIN, not batch; local range
       ``U(0.8, 1.0)`` - deliberately high, per upstream's comment "mask
       condition ... local coverage between 20% and 80%" for aa/dssp vs this
       higher floor for the aggregate channel) is now included; it feeds
       `salad_torch.modules.condition.prepare_condition`'s `dssp_mean`/
       `dssp_mean_mask` keys, previously always left at their "absent"
       defaults during training.

    Returns `(aa_condition, dssp_condition, aa_mask, dssp_mask, dssp_mean,
    dssp_mean_mask)`: the first two and the last two feed the MODEL (what
    it's told); `aa_mask`/`dssp_mask` ALSO feed `total_loss` (which residues
    are excluded from the corresponding NLL because they were told, not
    asked - spec-03 §1.3).
    """
    num_batch = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    num_chain = int(chain_index.max().item()) + 1 if chain_index.numel() else 0
    mask_b = mask.bool()

    sequence_mask = _mask_me(batch_index, num_batch, p=0.5, p_min=0.2, p_max=1.0,
                             generator=generator)
    sse_mask = _mask_me(batch_index, num_batch, p=0.5, p_min=0.2, p_max=1.0,
                        generator=generator)
    sse_mean_mask = _mask_me(chain_index, num_chain, p=0.5, p_min=0.8, p_max=1.0,
                             generator=generator)

    # outer gate (NSB:248-250): ONE draw, shared by aa_mask and dssp_mask -
    # dssp_mean_mask is NOT gated by it (upstream applies `use_condition` only
    # to the two `Condition.__call__` return masks, not to `sse_mean_mask`,
    # which is folded into `condition` inside `Condition.__call__` itself and
    # is therefore already final by the time `prepare_condition` sees it).
    use_condition = (torch.rand(num_batch, generator=generator,
                                device=batch_index.device) < 0.5)[batch_index]

    aa_mask = use_condition & sequence_mask & mask_b
    dssp_mask = use_condition & sse_mask & mask_b
    dssp_mean_mask = sse_mean_mask & mask_b

    aa_condition = torch.where(aa_mask, aa_gt, torch.full_like(aa_gt, 20))
    dssp_condition = torch.where(dssp_mask, dssp_gt, torch.full_like(dssp_gt, 3))

    # dssp_mean (NSB:2546-2548): the TRUE per-chain mean one-hot dssp
    # composition, computed from ground truth regardless of dssp_mean_mask -
    # the mask is applied later, at the `condition` combination step.
    dssp_onehot = torch.nn.functional.one_hot(dssp_gt, 3).float()
    dssp_mean = index_mean(dssp_onehot, chain_index, mask_b[:, None])

    return (aa_condition, dssp_condition, aa_mask, dssp_mask,
           dssp_mean, dssp_mean_mask)


# ---------------------------------------------------------------------------
# prepare_batch: raw PackedProteinStream draw -> the model+loss's `data` dict
# ---------------------------------------------------------------------------
def prepare_batch(raw: dict, *, augment_size: int, encoder: nn.Module | None = None,
                  generator: torch.Generator | None = None) -> dict:
    """`StructureDiffusion.prepare_data` + `.encode` (spec-03 §1.1-1.2), plus
    this port's `dssp`/conditioning-reveal wiring. `raw` is one
    `PackedProteinStream` draw already converted to torch tensors
    (`data/collate.py: example_to_tensors`) - atom24, un-centered, un-truncated.

    `encoder` is a real `modules.encoder.Encoder` instance (spec-05), run with
    gradients ENABLED - its output becomes `data["pos"]`, the ground truth the
    diffusion trunk is trained to denoise, so a stop-gradient here would
    silently turn off the joint training the released checkpoints were
    actually produced with. Required whenever `augment_size > 0` (every
    released config); may be omitted only for a hypothetical `augment_size=0`
    config, matching the JAX `elif`/`else` in spec-05 §1.

    Returns a `data` dict with every field `StructureDiffusionPredict.forward`
    and `losses.total.total_loss` need (see their docstrings for the exact
    key list); `t_pos`/`t_seq`/`pos_noised`/`seq_noised`/`corrupt_aa` are added
    afterwards by `training_step` (they depend on a fresh noise draw, which
    self-conditioning needs to redo independently - spec §1.4-1.5).
    """
    aa_gt = raw["aa_gt"]
    atom_pos_24 = raw["all_atom_positions"]
    atom_mask_24 = raw["all_atom_mask"]
    chain_index_local = raw["chain_index"]
    batch_index = raw["batch_index"]

    # -- prepare_data steps 1-5 (spec-03 §1.1) -------------------------------
    atom_pos = atom_pos_24[:, :14]
    atom_mask = atom_mask_24[:, :14]
    chain = unique_chain(chain_index_local, batch_index)
    mask = raw["seq_mask"].bool() & raw["residue_mask"].bool() & atom_mask[:, :3].all(dim=-1)

    center = index_mean(atom_pos[:, 1], batch_index, atom_mask[:, 1:2])
    atom_pos = atom_pos - center[:, None, :]
    pseudo_cb = compute_pseudo_cb(atom_pos)
    atom_pos = torch.where(atom_mask[..., None], atom_pos, pseudo_cb[:, None, :])

    # -- encode (spec-03 §1.2): backbone+CB is exact; the extra `augment_size`
    # atoms come from the real, jointly-trained Encoder (spec-05 §1) ---------
    backbone5 = positions_to_ncacocb(atom_pos)   # (L, 5, 3): N, CA, C, O, CB
    if augment_size > 0:
        if encoder is None:
            raise ValueError(
                "prepare_batch: augment_size > 0 requires a real Encoder "
                "(spec-05 §1) - the non-learned CB-replication placeholder "
                "was removed, it silently detached the diffusion trunk's "
                "ground truth from the encoder's gradient")
        enc_data = {"pos": backbone5, "residue_index": raw["residue_index"],
                   "chain_index": chain, "batch_index": batch_index, "mask": mask}
        _, pos20 = encoder(enc_data)
    else:
        pos20 = backbone5

    # -- dssp ground truth (spec §A.6; only `dssp` feeds the loss, §0c) ------
    with torch.no_grad():
        dssp_gt, _, _ = assign_dssp(atom_pos, batch_index, mask)

    # -- conditioning reveal (transcribed from NSB - see reveal_conditioning
    #    docstring for exactly what the prior best-effort version missed) ----
    (aa_condition, dssp_condition, aa_mask, dssp_mask,
    dssp_mean, dssp_mean_mask) = reveal_conditioning(
        aa_gt, dssp_gt, mask, chain, batch_index, generator=generator)

    return {
        "aa_gt": aa_gt, "seq": aa_gt,
        "residue_index": raw["residue_index"], "chain_index": chain,
        "batch_index": batch_index, "mask": mask,
        "atom_pos": atom_pos, "atom_mask": atom_mask,
        "pos": pos20, "dssp_gt": dssp_gt,
        "aa_condition": aa_condition, "dssp_condition": dssp_condition,
        "aa_mask": aa_mask, "dssp_mask": dssp_mask,
        "dssp_mean": dssp_mean, "dssp_mean_mask": dssp_mean_mask,
    }


def _draw_noise_clocks(data: dict, *, generator: torch.Generator | None = None) -> dict:
    """Spec §1.4: `t_pos`/`t_seq` are drawn PER BATCH ITEM, independently, then
    broadcast to residues via `batch_index`. `t_pos` is then REASSIGNED to the
    cosine-schedule sigma - from here on it means noise scale, not raw time.
    """
    batch_index = data["batch_index"]
    device = batch_index.device
    num_items = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    t_pos_raw = torch.rand(num_items, generator=generator, device=device)
    t_seq_raw = torch.rand(num_items, generator=generator, device=device)
    out = dict(data)
    out["t_pos"] = sigma_scale_cosine(t_pos_raw[batch_index])
    out["t_seq"] = t_seq_raw[batch_index]
    return out


def training_step(model: nn.Module, encoder: nn.Module | None, ema: EMA,
                  optimizer: torch.optim.Optimizer,
                  config: TrainConfig, micro_batches: list[dict], *,
                  device: torch.device, generator: torch.Generator | None = None,
                  lr: float, encoder_ema: EMA | None = None) -> dict[str, float]:
    """One optimizer step, over `1..accumulate` independent micro-batches.

    Each element of `micro_batches` is one `PackedProteinStream` draw (already
    `example_to_tensors`-converted), i.e. its OWN local `batch_index` - never
    concatenated into a shared segment-op namespace across micro-batches
    (spec §0f). Gradients are averaged across micro-batches (mean of means,
    matching `optax`'s fixed-`accumulate`-divisor online mean, spec §B.5).

    `encoder` (spec-05) builds `data["pos"]` WITH gradients - it is jointly
    trained, not a frozen preprocessing step (module docstring), so its
    parameters get their own gradient-clip contribution and their own `EMA`
    (`encoder_ema`) - kept a SEPARATE `EMA` instance from `ema` (rather than
    merging both modules into one container) so `ema`'s own key namespace
    stays exactly `model.named_parameters()`, unchanged from before this
    module had an encoder at all - a `checkpoint-ema-*.pt` consumer that only
    ever samples (and so never touches `encoder`) needs nothing new. Both are
    optimized by the SAME `optimizer` (one `Adam` over both modules'
    parameters, spec-05: "part of the same optimizer").

    Order matters (spec §B.2/§B.4): backward -> clip -> `ema.update` -> LR set
    -> `optimizer.step()`, so Adam's moment estimates see the CLIPPED gradient
    and EMA sees the PRE-update parameters (the documented one-step staleness).
    """
    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    terms_sum: dict[str, float] = {}
    n = len(micro_batches)

    for raw in micro_batches:
        data = prepare_batch(raw, augment_size=model.config.augment_size,
                             encoder=encoder, generator=generator)
        data = _draw_noise_clocks(data, generator=generator)

        # -- self-conditioning: 0 or 1 stop-gradient preview passes (§1.5) --
        num_aa = data["pos"].shape[0]
        zero_prev = {"pos": torch.zeros_like(data["pos"]),
                    "local": torch.zeros((num_aa, model.config.local_size),
                                        device=device, dtype=data["pos"].dtype)}
        do_self_cond = bool(torch.randint(0, 2, (1,), generator=generator).item())
        if do_self_cond:
            with torch.no_grad():
                preview_data = dict(data)
                preview_data.update(apply_diffusion(model.config, data, generator=generator))
                _, prev = model(preview_data, zero_prev, predict_aa=False)
        else:
            prev = zero_prev

        # -- the real pass: FRESH noise at the same (t_pos, t_seq), reused
        #    for both the model call and every loss term (spec §1.5) --------
        data.update(apply_diffusion(model.config, data, generator=generator))
        result, _ = model(data, prev, predict_aa=False, compute_distogram=True)

        total, terms = total_loss(model.config, data, result, generator=generator)
        (total / n).backward()

        for key, value in terms.items():
            terms_sum[key] = terms_sum.get(key, 0.0) + float(value.detach()) / n

    clip_params = list(model.parameters())
    if encoder is not None:
        clip_params += list(encoder.parameters())
    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=config.clip)
    ema.update(model)          # BEFORE optimizer.step(): the documented staleness
    if encoder is not None and encoder_ema is not None:
        encoder_ema.update(encoder)
    optimizer.step()
    return terms_sum


# ---------------------------------------------------------------------------
# Checkpointing (spec §B.5) - torch format, atomic write (upstream's raw
# `pickle.dump` to `save.jax` has no such protection against a crash mid-write)
# ---------------------------------------------------------------------------
def _atomic_save(obj: dict, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(obj, fh)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_checkpoint(path: str, model: nn.Module, ema: EMA,
                    optimizer: torch.optim.Optimizer, step: int,
                    encoder: nn.Module | None = None,
                    encoder_ema: EMA | None = None) -> None:
    """`encoder`/`encoder_ema` (spec-05) are saved under their own
    `"encoder_params"`/`"encoder_ema_params"` keys - `encoder` is a separate
    `nn.Module` from `model` (kept that way so `load_model`'s existing
    `strict=True` sampling-only checkpoint load is untouched, see
    `training/loop.py` module docstring), but its weights are jointly trained
    and must survive a checkpoint/resume just like `model`'s."""
    state = {
        "params": {k: v.detach().to(torch.float32).cpu() for k, v in model.state_dict().items()},
        "ema_params": {k: v.to(torch.float32).cpu() for k, v in ema.state_dict().items()},
        "opt_state": optimizer.state_dict(),
        "step": step,
        "rng_state": torch.get_rng_state(),
    }
    if encoder is not None:
        state["encoder_params"] = {k: v.detach().to(torch.float32).cpu()
                                   for k, v in encoder.state_dict().items()}
    if encoder_ema is not None:
        state["encoder_ema_params"] = {k: v.to(torch.float32).cpu()
                                       for k, v in encoder_ema.state_dict().items()}
    _atomic_save(state, path)


def load_checkpoint(path: str, model: nn.Module, ema: EMA,
                    optimizer: torch.optim.Optimizer | None = None,
                    encoder: nn.Module | None = None,
                    encoder_ema: EMA | None = None) -> int:
    """Returns the resumed step id, or 0 if `path` doesn't exist yet.

    Upstream's `load_loop_state` swallows ANY exception and starts fresh
    (spec §B.5) - deliberately NOT reproduced here (a torch port should not
    silently eat a corrupted-checkpoint error); only a missing file is treated
    as "start fresh". If `encoder` is given but the checkpoint predates it
    (no `"encoder_params"` key), that IS treated as an error: silently keeping
    a randomly-initialised encoder while resuming a trained trunk would train
    against a moving, uncoordinated ground truth.
    """
    if not os.path.exists(path):
        return 0
    # weights_only=False: the checkpoint also carries optimizer state and the
    # RNG state, not just tensors - both need the full unpickler.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"])
    ema.load_state_dict(state["ema_params"])
    if encoder is not None:
        if "encoder_params" not in state:
            raise ValueError(f"{path}: no 'encoder_params' but an encoder was given "
                             "(checkpoint predates the Encoder wiring)")
        encoder.load_state_dict(state["encoder_params"])
    if encoder_ema is not None:
        if "encoder_ema_params" not in state:
            raise ValueError(f"{path}: no 'encoder_ema_params' but an encoder_ema was given "
                             "(checkpoint predates the Encoder wiring)")
        encoder_ema.load_state_dict(state["encoder_ema_params"])
    if optimizer is not None and "opt_state" in state:
        optimizer.load_state_dict(state["opt_state"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])
    return int(state["step"])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
class Trainer:
    """Drives `training_step` over a stream of raw dataset draws.

    `batches` yields raw `PackedProteinStream` numpy-dict draws forever (e.g.
    `iter(PackedProteinStream(...))`, converted to tensors here via
    `data/collate.py:example_to_tensors` - kept out of the dataset/collate
    layer so this module owns the device placement). `accumulate > 1` pulls
    that many draws per optimizer step (spec §B.5's "K independent
    forward/backward calls").

    Owns a real `Encoder` (spec-05) whenever `model.config.augment_size > 0`
    (every released config) - constructed once here and threaded through
    `prepare_batch`/`training_step`, with its own parameters in the SAME
    `optimizer` as `model`'s and its own `EMA`/checkpoint entry (see
    `training_step`/`save_checkpoint` docstrings for why it is a second `EMA`
    object rather than folded into `self.ema`).
    """

    def __init__(self, model: nn.Module, config: TrainConfig, batches: Iterable[dict], *,
                device: torch.device | str = "cpu",
                tb_writer=None, generator: torch.Generator | None = None):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.batches = iter(batches)
        self.tb_writer = tb_writer
        self.generator = generator

        if model.config.augment_size > 0:
            self.encoder: nn.Module | None = Encoder(model.config).to(device)
        else:
            self.encoder = None

        params = list(model.parameters())
        if self.encoder is not None:
            params += list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=config.lr, betas=(config.b1, config.b2),
            eps=config.adam_eps, weight_decay=0.0)   # NEVER weight decay (spec §B.2)
        self.ema = EMA(model, decay=config.ema_weight)
        self.encoder_ema = (EMA(self.encoder, decay=config.ema_weight)
                           if self.encoder is not None else None)
        self.schedule = config.build_schedule()
        self.step = 0
        self._last_save_time = time.time()

    def _next_micro_batches(self) -> list[dict]:
        from ..data.collate import example_to_tensors
        out = []
        for _ in range(self.config.accumulate):
            raw = next(self.batches)
            out.append(example_to_tensors(raw, device=self.device))
        return out

    def resume(self, path: str) -> None:
        self.step = load_checkpoint(path, self.model, self.ema, self.optimizer,
                                    encoder=self.encoder, encoder_ema=self.encoder_ema)

    def run_step(self) -> dict[str, float]:
        micro_batches = self._next_micro_batches()
        lr = self.schedule(self.step)
        terms = training_step(self.model, self.encoder, self.ema, self.optimizer,
                              self.config, micro_batches, device=self.device,
                              generator=self.generator, lr=lr,
                              encoder_ema=self.encoder_ema)
        terms["lr"] = lr
        self.step += 1

        if self.tb_writer is not None:
            for key, value in terms.items():
                self.tb_writer.add_scalar(key if key == "total" or key == "lr"
                                         else f"{key}_loss", value, self.step)

        if self.step % self.config.checkpoint_interval == 0:
            ckpt_dir = self.config.checkpoint_dir
            save_checkpoint(os.path.join(ckpt_dir, f"checkpoint-{self.step}.pt"),
                            self.model, self.ema, self.optimizer, self.step,
                            encoder=self.encoder, encoder_ema=self.encoder_ema)
            # the EMA-only file is what sampling/eval should load (spec §B.4)
            ema_only = {"ema_params": {k: v.to(torch.float32).cpu()
                                       for k, v in self.ema.state_dict().items()},
                       "step": self.step}
            if self.encoder_ema is not None:
                ema_only["encoder_ema_params"] = {
                    k: v.to(torch.float32).cpu()
                    for k, v in self.encoder_ema.state_dict().items()}
            _atomic_save(ema_only,
                        os.path.join(ckpt_dir, f"checkpoint-ema-{self.step}.pt"))

        now = time.time()
        if now - self._last_save_time >= self.config.save_interval_seconds:
            save_checkpoint(os.path.join(self.config.checkpoint_dir, "save.pt"),
                            self.model, self.ema, self.optimizer, self.step,
                            encoder=self.encoder, encoder_ema=self.encoder_ema)
            self._last_save_time = now

        return terms

    def run(self, *, on_step: Callable[[int, dict[str, float]], None] | None = None) -> None:
        while self.step < self.config.max_steps:
            terms = self.run_step()
            if on_step is not None:
                on_step(self.step, terms)

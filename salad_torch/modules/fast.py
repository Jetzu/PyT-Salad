"""Compiled / CUDA-graph fast path for inference.

Why this file exists
--------------------
The denoiser is **launch-bound**, not compute-bound. Measured on an H100 at
L=100: one `DiffusionBlock` costs ~8 ms while JAX's *entire* forward is ~3.1 ms,
and the eager forward takes the same 75 ms whether L is 50 or 800. The work is
tiny; the ~1500 kernel launches per step are the cost.

So the fix is not a better algorithm, it is fewer launches: `torch.compile` to
fuse, and CUDA graphs to replay the whole step as one submission. That needs

* a **tensor-in / tensor-out** region - Dynamo does not faithfully trace the
  python-dict shuffling the ergonomic API does (it drops the `condition` key),
  hence `Diffusion.forward_core`;
* **static input buffers** - CUDA graphs replay against fixed addresses, so the
  caller must copy into the same tensors each step rather than pass new ones;
* **cloned outputs** - a graph's output buffers are overwritten by the next
  replay, which is exactly the "accessing tensor output of CUDAGraphs that has
  been overwritten" error an earlier attempt hit.

Compilation is a one-time cost (tens of seconds); the per-design cost is what
this is for.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .. import gemm_exact, opt
from .diffusion import uniform_draw_count

__all__ = ["FastDenoiser", "GroupFastDenoiser", "PRESETS", "memory_bound_group"]

#: `SALAD_OPT_DETSEG=1` forces the deterministic segment reduction on for every
#: preset; captured here because `FastDenoiser` writes `opt.DET_SEGMENT`.
_ENV_DET_SEGMENT = opt.DET_SEGMENT
#: ...and `SALAD_OPT_BATCHCOND=0` forces P7 off for every preset.
_ENV_BATCH_COND = opt.BATCH_COND
#: ...and `SALAD_OPT_PAIRFEAT=0` forces O6 off for every preset (same
#: veto-only, `batch_cond_pair` contract - the preset decides, the env var can
#: only turn it off).
_ENV_PAIR_FEAT_MERGE = opt.PAIR_FEAT_MERGE
#: ...and `SALAD_OPT_FRAMEMULSUM=1` forces O8 ON for every preset. This
#: one is `DET_SEGMENT`'s OR contract, NOT `BATCH_COND`'s AND one - see
#: `opt.FRAME_MULSUM`'s docstring for why (it fails eager `test_model_parity`
#: outright, so a bare default of True would break every raw parity-script
#: invocation, not just `fast`).
_ENV_FRAME_MULSUM = opt.ENV_FRAME_MULSUM
#: ...and `SALAD_OPT_IPAMULSUM=1` forces O2 ON for every preset (same
#: OR contract, same reason - see `opt.IPA_MULSUM`).
_ENV_IPA_MULSUM = opt.ENV_IPA_MULSUM

#: Inference presets. **"exact" is the default and must stay so.**
#:
#: "exact" holds the port's measured parity floor against JAX salad
#: (0.00002 A on default_vp / _vp_scaled / _ve_scaled, 0.00020 A on
#: multimotif_vp): every switch it turns on is bit-exact, or - for the
#: K 64 -> 48 condition branch - drops only terms that are exactly 0.0, so
#: the sole change is the shape of a reduction tree.
#:
#: A preset that is allowed to move the numbers belongs in this table too, as
#: its own entry; do not loosen "exact".
try:
    from torch._functorch.compile_utils import fx_graph_cse as _fx_graph_cse
except (ImportError, AttributeError):  # pragma: no cover - private torch path
    # A version bump that moves it must degrade to the plain compile, never fail
    # the denoiser. After a torch upgrade, check whether the newer inductor CSEs
    # on its own - if it does, delete this and the pass below.
    _fx_graph_cse = None


def _cse_graph_pass(graph):
    """O1's CSE pass. MODULE level on purpose - see `FastDenoiser.__init__`.

    Inductor pickles `joint_custom_post_pass` into its on-disk FX-graph cache
    key. A closure is unpicklable, inductor swallows the resulting
    "Can't pickle local object" and silently disables that cache.
    """
    graph.owning_module.graph = _fx_graph_cse(graph)


def _cse_graph_pass_gemm_exact(graph):
    """Identical to `_cse_graph_pass`; the NAME is the point - see below."""
    graph.owning_module.graph = _fx_graph_cse(graph)


def _cse_graph_pass_gemm_stable(graph):
    """Identical to `_cse_graph_pass`; the NAME is the point - see below."""
    graph.owning_module.graph = _fx_graph_cse(graph)


_cse_graph_pass = _cse_graph_pass if _fx_graph_cse is not None else None
_CSE_PASS_BY_MODE = {} if _fx_graph_cse is None else {
    "exact": _cse_graph_pass_gemm_exact,
    "stable": _cse_graph_pass_gemm_stable,
}


PRESETS: dict[str, dict] = {
    "exact": dict(
        # skip the AA decoder + dssp + angle head on steps whose outputs are
        # never read (P2). A pure skip: `pos`/`local`/`trajectory` are
        # bit-identical, and the written PDB is byte-identical.
        lean_steps=True,
        # drop the dead dmap/flags pair terms and the all-`-1` block-contact
        # neighbour stage when the design has no pair conditioning (P4 + P5).
        # Gated at runtime on the real tensors, never assumed.
        skip_pair_cond=True,
        # the dssp head + the distogram neighbour draw. Left ON: removing them
        # was measured SLOWER (6.48 -> 6.9 ms at L=100), the extra wrapper and
        # the lost fusion costing more than the (N, N) Gumbel saved.
        need_aux=True,
        # fp32 matmuls stay fp32, and the sampler keeps its 500/400 schedule.
        tf32=False,
        steps_scale=1.0,
        # inductor scheduler knob, -1.4 % at L=100 / -2.4 % at L=50, and
        # BIT-IDENTICAL on CPU *and* on the H100 under the deterministic
        # segment reduction. Not a numerics trade at all.
        aggressive_fusion=True,
        # P7, the batched condition-branch prologue: -9.2 % of a lean step.
        # Reassociation only (bmm vs six mm); see `batch_cond_pair` below.
        batch_cond_pair=True,
        # O6 - the pair-feature sharing (aa_frames_t + neighbour gather
        # reused between `direction_features` and `pair_vector_features`).
        # BIT-EXACT (see `opt.PAIR_FEAT_MERGE`) - takes every preset, same as
        # `aggressive_fusion` above.
        pair_feat_merge=True,
        # inductor's matmul templates (`max_autotune_gemm`) - REJECTED
        # from every preset. -14% at L=100 (eroding to -6.4% at L=400) but NOT
        # bit-exact: whole-forward H100/DETSEG control measured pos
        # max|d|=5.84e-05 at L=100 / 7.87e-04 at L=400, templates-on vs off
        # (both above the 2e-05 A floor), and templates-on is not even
        # reproducible against itself across independent cold-cache compiles
        # (up to pos 8.87e-05 between two compiles of the SAME config - the
        # autotune benchmark can pick a different kernel at a shape where
        # candidates are within noise). See opt.py's GEMM_TEMPLATES docstring
        # and the project notes
        # `SALAD_OPT_GEMMTPL=1` still works as an opt-in for experimentation;
        # it is a `fast`-preset candidate pending an n=160 structural
        # validation (earlier method) that has not been run.
        #
        # tried to rescue this for `exact` with a per-shape bitwise-exact
        # candidate filter (`SALAD_OPT_GEMMTPL=exact`, `gemm_exact.py`) and it
        # did NOT work: even restricted to Triton candidates individually
        # verified bitwise-equal to cuBLAS at their own shape, the WHOLE
        # FORWARD is still not bit-exact (pos max|d|=5.44e-05, barely better
        # than the unfiltered arm's 5.84e-05) and the verified-candidate SET
        # is itself not stable across cold compiles - see RESULTS.md section 4.
        gemm_templates=False,
        # inductor's `decompose_mm_pass` (see opt.DECOMPOSE_SMALL_BMM).
        # -15.5 % at L=100 on top of P7 but a reassociation (pos
        # max|d|=4.96e-05 A, above the 2e-05 floor), so False here too -
        # `exact` never takes it.
        decompose_small_bmm=False,
        # O8 - the 3x3 frame-algebra mul+sum rewrite. NEVER in `exact`:
        # it is a reassociation over a 3-element contraction and the whole
        # compiled `forward_core` CPU control disagrees on 9/11 outputs.
        # See `opt.FRAME_MULSUM`.
        frame_mulsum=True,
        # O2 - the IPA mul+sum rewrite. NEVER in `exact`: it is a
        # reassociation of four reductions and `test_sampling_parity` fails at
        # 7/64 residues. See `opt.IPA_MULSUM`.
        ipa_mulsum=False,
    ),
    # ------------------------------------------------------------------
    # "fast" - maximum speed, ALLOWED to move the numbers. It is NOT a
    # different model: same checkpoint, same architecture, same neighbourhood
    # sizes, same sampler. What it gives up is mantissa bits in the matmuls and
    # half the trajectory's resolution - and nothing else, because nothing else
    # that was tried paid for itself. Measured at L=100 on an H100, production
    # path, `x_preset.py` (the project notes):
    #
    # reporting step lean step per design
    # exact 4.381 ms 3.568 ms 1.43 s
    # exact + TF32 only 3.719 ms 2.996 ms 1.20 s
    # fast without TF32 4.385 ms 3.563 ms 0.72 s
    # fast 3.614 ms 2.894 ms 0.58 s
    #
    # i.e. TF32 is -16 to -19 % of the step and halving the schedule is the
    # other 2x. JAX salad is 1.56 s per design, so this is ~2.7x JAX and
    # ~2.5x the exact path.
    #
    # QUALITY: validated at n=160 designs per arm, not at the n=8 this preset
    # shipped on (+ 903296; L=100, seeds 42 and 1234 pooled).
    # Halving the schedule is the only knob here that changes the TRAJECTORY
    # rather than the arithmetic, so it was factorised out and tested on its
    # own. Converged designs, `exact` in brackets:
    #
    # CA-CA (A) Rg (A) helix strand diverged
    # exact 3.7628 +- 0.0137 12.89 +- 1.76 0.566 0.353 2/160
    # TF32 only 3.7615 +- 0.0123 12.89 +- 1.46 0.572 0.351 0/160
    # schedule only 3.7627 +- 0.0141 12.89 +- 1.56 0.575 0.345 0/160
    # fast 3.7613 +- 0.0131 12.90 +- 1.89 0.580 0.335 0/160
    #
    # Welch vs exact: every metric p >= 0.24. Two-sample KS (the shape test a
    # t-test cannot do): every metric p >= 0.13. Brown-Forsythe on the spread:
    # nothing survives Bonferroni. The specific worry - half the steps means
    # less time to converge, so LESS secondary structure or MORE extended
    # designs - is refuted with power: at n=160 this would have detected a
    # 0.106 shift in helix fraction, 0.095 in strand, 0.51 A in Rg and 0.0041 A
    # in CA-CA; the largest observed shift is 0.013 / 0.019 / 0.00 / 0.0015.
    # Helix moves UP, not down.
    #
    # The n=8 table this preset shipped on (the project notes, section 4) read helix 0.605 ->
    # 0.591 and Rg 12.50 -> 12.92, i.e. exactly the feared direction. Both were
    # noise, and so was an n=40 rerun that put the fast arm's Rg sd at 2.3x
    # exact's - it came from 3 designs and vanished at n=160. Per-design SSE
    # variance here is sd ~0.35, so anything below n~100 says nothing.
    #
    # DIVERGED designs (CA-CA mean > 4.5 A) are a failure mode of the MODEL,
    # not of this preset: 2/160 on `exact` (worst CA-CA 28.3 A, Rg 408 A),
    # 0/160 on all three deviating arms, Fisher p=0.50. Upstream JAX produced
    # 1/40 (the project notes, section 10). Filter on CA-CA downstream, on either preset.
    #
    # REFUTED and deliberately absent - each measured, none of them free:
    # * K 64 -> 32 on both attention branches: -10 % of the step, but coil
    # 0.076 -> 0.109 and CA-CA clashes 0.1 -> 1.1 per design. The only
    # candidate that visibly degraded the structures. Dropped.
    # * one shared self-conditioning Gumbel draw for all six blocks:
    # 2.894 vs 2.882 ms, i.e. NOTHING, measured with the switch verified
    # live (6.45 A change in `pos`). A deviation with no payoff.
    # * bypassing the conditioning-cache key: 3.563 vs 3.568 ms, nothing -
    # and after S1 a weakened cache key is a bug, not a trade.
    # * `topk` for the stable argsort (+5.6 %), `need_aux=False` (no change),
    # bf16 inside the GEMM via matmul_precision("medium") (2.856 vs 2.760,
    # i.e. slower than TF32). See RESULTS.md section 5.
    "fast": dict(
        lean_steps=True,
        skip_pair_cond=True,
        need_aux=True,
        # TF32: 10 mantissa bits instead of 24 in every fp32 matmul. The single
        # largest lever in the preset, and the one the exact path must refuse -
        # measured TF32 moving coordinates by ~6e-3 A while leaving
        # `aatype` untouched, i.e. it looks fine and is not fp32. Here that is
        # the point. For scale: CUDA's own run-to-run nondeterminism on this
        # model is 2.2e-04 A , so TF32 is a real perturbation, ~1500x the
        # noise floor - but three orders of magnitude smaller than what
        # shrinking the neighbourhood does.
        tf32=True,
        # halve the sampler: 250 steps reporting at 200 instead of 500/400.
        # The schedule is `raw_t = 1 - step/num_steps`, so halving BOTH keeps
        # the trajectory's endpoints (raw_t 1.0 -> 0.2) and only coarsens it.
        # `scripts/generate.py` applies this, and only when the command line
        # did not set --num_steps/--out_steps itself.
        #
        # 0.5 STAYS, on earlier n=160-per-arm test (see QUALITY above): the
        # schedule-only arm is indistinguishable from `exact` on mean, spread
        # and distribution shape, so there is nothing to buy by backing off to
        # 0.75, and nothing to isolate by splitting this into a second preset.
        # A "faster" preset at 125/100 is NOT offered: the project notes, section 4 found it
        # passes the same structural gate, but a 4x cut deserves a
        # designability check (ProteinMPNN + a folding model), which no arm in
        # this port has run. Structural gates cannot answer designability.
        steps_scale=0.5,
        aggressive_fusion=True,
        batch_cond_pair=True,
        # O6 - bit-exact, takes every preset. See `opt.PAIR_FEAT_MERGE`.
        pair_feat_merge=True,
        # /matmul TEMPLATES (`max_autotune_gemm`) stay rejected
        # everywhere, including `fast` - the per-shape bitwise-exact filter
        # (`SALAD_OPT_GEMMTPL=exact`) doesn't buy exactness or determinism
        # (see the `exact` preset's comment above) and only gets -11% at
        # L=100, worse than `decompose_small_bmm` below, for a ~5-40x
        # compile-time tax.
        gemm_templates=False,
        # inductor's `decompose_mm_pass` (see opt.DECOMPOSE_SMALL_BMM).
        # -15.5 % at L=100 on top of P7 (: 3.0721 -> 2.5946 ms), a
        # reassociation (pos max|d|=4.96e-05 A, above the 2e-05 floor, so
        # `exact`/`deterministic` never get it) but PERFECTLY REPRODUCIBLE
        # across cold compiles (, unlike `gemm_templates`) - and the
        # n=160 design-quality gate is in for the ACTUAL shipped `fast`
        # preset: 0/160 diverged, CA-CA 3.7638 A / Rg 12.96 vs baseline's
        # 3.7625 A / 13.04 . Ships here.
        decompose_small_bmm=32,
        # O8 - the 3x3 frame-algebra mul+sum rewrite. Reassociation, not
        # bit-exact (CPU compiled control disagrees on 9/11 outputs - see
        # `opt.FRAME_MULSUM`), but passes the n=160 structural gate at -18.3%
        # (RESULTS.md section 4 + the landing re-verification section 9).
        frame_mulsum=True,
        # O2 - see the verdict recorded in `opt.IPA_MULSUM`.
        # candidate - same, flipped to True only if n=160 clears it.
        ipa_mulsum=True,
    ),
    # ------------------------------------------------------------------
    # "deterministic" - `exact`, plus a segment reduction that does not
    # use atomics, so **`--seed` actually pins a CUDA run**. Measured at L=100
    # on an H100 (`x_floor.py`): 40 identical forwards produce
    # **1 distinct bit pattern instead of 40**, and the step costs 3.634 ms
    # lean / 4.457 ms full against `exact`'s 3.558 / 4.384 in the same job,
    # i.e. **+2.1 %**.
    #
    # It is NOT the default because it changes the summation order of every
    # `index_sum`, so it is its own baseline - and that baseline is NOT free.
    # It is a TRADE: a reproducible CUDA run, paid for in divergence from JAX.
    # Measured with `tests/test_sampling_parity.py` , reproduced on the
    # live tree AND independently on an isolated snapshot ; the two
    # agree to every digit, and the CPU run reproduces the GPU one - so this is
    # the reassociation itself, not the atomics:
    #
    # DETSEG=0 DETSEG=1
    # final aatype 1/64 residues **7/64 residues** <- the real cost
    # final_aa rel 2.046e-02 6.019e-02
    # final structure 0.06166 A 0.19861 A (~3.2x the drift)
    #
    # **Lead with the aatype count.** Seven of sixty-four residues take a
    # DIFFERENT AMINO ACID - a discretely different design, which no rounding
    # argument reaches. `tests/test_sampling_parity.py` FAILS under this preset.
    # (It also reports a 30.6x final/noise-floor ratio; do NOT lean on that
    # number. Its denominator is a self-divergence floor measured on the ATOMICS
    # path, and driving self-divergence to zero is this preset's whole purpose,
    # so the denominator no longer describes the configuration under test. The
    # raw deviations and the aatype count are the defensible statements.)
    #
    # What DOES still hold at the usual floors: the four checkpoint parity
    # configs (0.00002 A / 0.00020 A multimotif), geometry 38/38, stack and
    # encoder. Only sampling moves.
    #
    # So: a legitimate preset to offer - reproducibility is worth real money for
    # debugging and for anything that has to rerun and match - but it has **no
    # recorded sampling baseline of its own yet**. Record one before reporting
    # any result from it; never read it against `exact`'s baseline.
    # Nothing in `exact` or `fast` is affected: `DET_SEGMENT` defaults False and
    # neither preset sets it.
    #
    # `torch.use_deterministic_algorithms(True)` is the obvious alternative
    # and is NOT usable here: it makes `scatter_add` deterministic (measured:
    # 17 -> 170 us for one (100, 512) call) but RAISES under CUDA graph
    # capture - "operation not permitted when stream is capturing" - so the
    # whole fast path is unavailable under it.
    "deterministic": dict(
        lean_steps=True,
        skip_pair_cond=True,
        need_aux=True,
        tf32=False,
        steps_scale=1.0,
        aggressive_fusion=True,
        batch_cond_pair=True,
        # O6 - bit-exact, takes every preset. See `opt.PAIR_FEAT_MERGE`.
        pair_feat_merge=True,
        # rejected everywhere - see the `exact` preset's comment above.
        gemm_templates=False,
        det_segment=True,
        # reassociation, never in the bitwise-reproducible preset.
        decompose_small_bmm=False,
        # O8 - reassociation, never in the bitwise-reproducible preset.
        frame_mulsum=False,
        # O2 - same.
        ipa_mulsum=False,
    ),
}

#: the tensor arguments of `Diffusion.forward_core`, in order
_CORE_ARGS = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
              "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt",
              "pos_gt", "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")
_OUT_NAMES = ("pos", "local", "trajectory", "aa", "aa_features", "dssp",
              "distogram_neighbours", "raw_angles", "angles", "angle_pos", "aatype")
#: what `forward_core(need_aa=False)` returns - the three tensors the sampler
#: feeds back, and nothing else
_OUT_NAMES_LEAN = ("pos", "local", "trajectory")


class FastDenoiser(nn.Module):
    """Wraps a `StructureDiffusionPredict` with a compiled, graph-captured core.

    The signature deliberately mirrors the eager model so it can be swapped in:

        fast = FastDenoiser(model, length=100, device="cuda").warmup()
        result, prev = fast(data, prev)

    `mode` is passed to `torch.compile`, and `manual_graph` decides who owns
    the CUDA graph:

    * `manual_graph=True` (default, and the fast one): compile with
      ``mode="default"`` and capture the WHOLE forward in one explicit
      `torch.cuda.CUDAGraph`, then replay it. 4.79 ms at L=100.
    * `manual_graph=False`: let inductor manage cudagraphs via
      ``mode="reduce-overhead"``. 6.5-7.0 ms.

    The two cannot be combined - wrapping an explicit capture around a
    reduce-overhead callable raises "Offset increment outside graph capture",
    because inductor is already capturing. `mode="default"` does not, which is
    what makes the manual capture legal. Capturing the whole forward once is
    what removes the remaining ~1400 per-launch costs; it is only possible
    because every RNG draw is hoisted out of the region (see `uniform`).
    """

    def __init__(self, model, length: int, device="cuda",
                 mode: str = "reduce-overhead", dtype=torch.float32,
                 capture: bool = True, manual_graph: bool = True,
                 preset: str = "exact", batch_size: int = 1):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
        self.preset = preset
        p = PRESETS[preset]
        self.model = model
        self.config = model.config
        self.length = length
        # - PACKED batching. `length` stays the per-design residue count;
        # every buffer in this instance is sized for the FLAT axis, which is
        # `batch_size` designs concatenated. salad has no leading batch
        # dimension, so a batch IS a longer flat axis plus a `batch_index`
        # (see `salad_torch.diffusion.sampling.pack_designs`). batch_size=1
        # is the ordinary single-design instance, unchanged in every respect.
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.flat_length = int(length) * self.batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.capture = capture and self.device.type == "cuda"
        self.manual_graph = manual_graph
        if manual_graph and mode == "reduce-overhead":
            # Mutually exclusive: inductor is already capturing in
            # reduce-overhead, so an explicit capture on top raises
            # "Cannot prepare for replay during capturing stage". The manual
            # graph needs a plain compile underneath it.
            mode = "default"
        # PROCESS-GLOBAL, and applied before the compile so inductor selects
        # kernels for the right precision. Set in BOTH directions on purpose:
        # building an "exact" denoiser after a "fast" one must put the process
        # back on the parity path rather than silently inherit TF32.
        torch.set_float32_matmul_precision("high" if p.get("tf32") else "highest")
        # Same contract as the line above, and for the same reason: a
        # PROCESS-GLOBAL switch, set in BOTH directions so an "exact" denoiser
        # built after a "deterministic" one goes back to the default reduction
        # rather than silently inheriting it. It has to be set BEFORE the
        # compile, because `index_sum` reads it inside the traced region and
        # Dynamo constant-folds it into a guard.
        opt.DET_SEGMENT = bool(p.get("det_segment")) or _ENV_DET_SEGMENT
        # Inductor's scheduler knob, likewise process-global and likewise set
        # before the compile because that is when the scheduler reads it.
        # Unlike TF32 this one is BIT-EXACT - measured on CPU and, under the
        # deterministic reduction, on the H100 itself - so `exact` takes it.
        import torch._inductor.config as _ind
        _ind.aggressive_fusion = bool(p.get("aggressive_fusion")) and opt.AGGRESSIVE_FUSION
        # . Turning `max_autotune_gemm` on makes inductor BENCHMARK a
        # Triton template against cuBLAS at every distinct GEMM shape in the
        # graph, at compile time, on the machine that is compiling, and keep
        # the winner. So the selection is per-shape AND per-device - the
        # narrow projections (K <= 128) go to Triton, the wide ones (`dist`
        # K=400, `proj_out` K=1024) stay on cuBLAS, and on a different card the
        # same measurement decides again. That adaptivity is the point: a
        # consumer Ada/Blackwell card has a different cuBLAS heuristic and a
        # different SM count, and nothing here is tuned to this node.
        #
        # This is reachable at `MM_MODE`'s DEFAULT (`einsum` -> batch-1 `bmm`)
        # via inductor's `tuned_bmm` template path - NO change to how
        # `HkLinear` contracts is needed. An earlier brief assumed batch-1
        # `bmm` forecloses inductor's matmul templates; that premise is
        # REFUTED (measured: `einsum` unchanged + templates on is -14 % at
        # L=100, bigger than the `mm` rewrite's -0.3 %, and with none of its
        # parity cost - see RESULTS.md). `opt.MM_MODE` and `GEMM_TEMPLATES` are
        # independent knobs for that reason.
        #
        # ATEN stays in the candidate list, so a shape where the template loses
        # keeps exactly the kernel it has today. Compile time goes up by the
        # benchmarking pass (one per distinct shape). REJECTED from every
        # preset (see PRESETS above) - not bit-exact end to end, so this line
        # is always `False and ...` today; kept live (not deleted) so a future
        # `fast`-only landing, once validated, is a one-line preset change.
        #
        # changes what "on" means. `gemm_templates` is now a MODE, not a
        # bool - `False`/`"off"`, `"all"` (earlier rejected fastest-wins race),
        # `"exact"` (only candidates BITWISE equal to cuBLAS at that shape,
        # measured NOT sufficient either - RESULTS.md section 4) or `"stable"`
        # (fastest, deterministic tie-break, still not equal to cuBLAS). See
        # `salad_torch/gemm_exact.py` and `opt.resolve_gemm_templates`.
        _tpl_mode = opt.resolve_gemm_templates(p.get("gemm_templates"))
        _ind.max_autotune_gemm = _tpl_mode != "off"
        if _ind.max_autotune_gemm:
            _ind.max_autotune_gemm_backends = "ATEN,TRITON"
        if _tpl_mode in ("exact", "stable"):
            gemm_exact.install(mode=_tpl_mode)
        else:
            gemm_exact.uninstall()
        # - `decompose_mm_pass`, see `opt.DECOMPOSE_SMALL_BMM`. A config
        # value, so unlike the candidate filter it DOES take part in the
        # FX-graph cache key and needs no salt. The pass module reads both
        # thresholds at import time, so they are re-set on the module too for
        # the case where a second denoiser is built in the same process.
        _dec = opt.resolve_decompose_small_bmm(p.get("decompose_small_bmm"))
        _pgf = dict(_ind.post_grad_fusion_options)
        _pgf.pop("decompose_mm_pass", None)
        if _dec:
            _pgf["decompose_mm_pass"] = {
                "min_first_dimension_decomposition": int(_dec),
                "max_other_dimention_decomposition": 32}
        _ind.post_grad_fusion_options = _pgf
        try:
            from torch._inductor.fx_passes import decompose_mem_bound_mm as _dmm
            if _dec:
                _dmm.min_first_dimension_decomposition = int(_dec)
                _dmm.max_other_dimention_decomposition = 32
        except Exception:
            pass
        # O1. Inductor's inference pipeline has no CSE pass: `joint_graph_passes`
        # runs remove_noop_ops, the pattern matcher and constant folding, and that is
        # all. So the denoiser rebuilds the SAME Gram-Schmidt frame 61 times per
        # forward (`aa_frames_t` -> `rot_from_two_vectors`, 4-5 launches each on an
        # (N, 3) tensor): three times inside every `DiffusionPairFeatures`, again in
        # `Update`, again in `UpdatePositions`, and `attn_2`'s `prev_pos/sigma` frame
        # is identical in all six blocks. 31 are exact duplicates. Running
        # `fx_graph_cse` over the post-dispatch graph deletes them: -50 kernels at
        # EVERY length (the duplication is a property of the graph, not the shape),
        # i.e. -2.5 to -2.9 % of a lean step at L=50/100/200.
        #
        # BIT-IDENTICAL, not approximately: CSE deletes re-executions of pure ops
        # with identical arguments and reassociates nothing. Verified bitwise on CPU
        # (deterministic by construction) over all eleven `forward_core` outputs at
        # L=64 and L=100 in two independent jobs - max|d| = 0.0 exactly. That is the
        # right control here: the four checkpoint parity tests run the EAGER path and
        # cannot see a compile-time graph change at all.
        #
        # Do NOT substitute `_ind.freezing = True`. It runs this same CSE and measures
        # identically (677 kernels, 3.05-3.07 ms) and it is the DOCUMENTED knob, but
        # it also constant-folds the parameters and runs its own layout passes, and
        # those DO move the numbers: measured pos 8.8e-06, raw_angles 1.5e-03 on the
        # same CPU control. Both sit UNDER the 2e-05 A parity floor, so freezing
        # would pass `test_model_parity` while quietly not being fp32-exact.
        # `_cse_graph_pass` is defined at MODULE level, not as a closure here, and
        # that is load-bearing: inductor pickles `joint_custom_post_pass` as part
        # of the on-disk FX-graph cache key. A closure raises "Can't pickle local
        # object" during that hash, inductor swallows it, and the FX-graph cache
        # is silently disabled - every `--fast` process then pays a full 4-8 minute
        # recompile with no cross-process reuse (caught this; the lower-level
        # Triton kernel cache still helps on a repeat length, which is why it only
        # shows up as slow cold starts rather than as an error).
        if _cse_graph_pass is not None:
            # . The GEMM-candidate filter is a monkeypatch on inductor's
            # lowerings, NOT a config value, so it does NOT participate in the
            # FX-graph cache key: without a salt, an `exact` compile would be
            # served a cached `all` artifact (or vice versa) and the difference
            # would silently vanish. `joint_custom_post_pass` DOES participate
            # (that is how earlier unpicklable-closure bug was found), and a
            # module-level function pickles as its qualified NAME - so pointing
            # it at a differently-named function with the same body salts the
            # key. Distinct name per mode, one line, no new config.
            _ind.joint_custom_post_pass = _CSE_PASS_BY_MODE.get(
                _tpl_mode, _cse_graph_pass)
        # P7, same contract: the preset chooses, `SALAD_OPT_BATCHCOND=0` can
        # only veto. Also process-global, and also set before the compile.
        opt.BATCH_COND = bool(p.get("batch_cond_pair")) and _ENV_BATCH_COND
        # O6, same veto-only contract: the preset chooses, the env var
        # can only turn it off. Process-global, set before the compile.
        opt.PAIR_FEAT_MERGE = bool(p.get("pair_feat_merge")) and _ENV_PAIR_FEAT_MERGE
        # O8 - `DET_SEGMENT`'s contract, not `BATCH_COND`'s: `or`, not
        # `and` (env can force ON for a measurement against any preset, but
        # cannot silently turn on outside one - default is False). It is read
        # inside the traced region (`geometry/frames.py`, `geometry/rigid.py`),
        # so Dynamo folds it into a guard - it must be set before `torch.compile`.
        opt.FRAME_MULSUM = bool(p.get("frame_mulsum")) and _ENV_FRAME_MULSUM
        # O2, same `DET_SEGMENT` OR contract.
        opt.IPA_MULSUM = bool(p.get("ipa_mulsum")) and _ENV_IPA_MULSUM
        # P7's stacked condition-branch weights are a CACHE of the per-block
        # parameters, normally refreshed by a `load_state_dict` post-hook.
        # Refresh once more here so a model whose weights arrived some other
        # way (a hand-assigned `.w.data`, a converter that bypasses
        # `load_state_dict`) cannot be served stale ones. It costs one stack of
        # ~1.9 MB, once per denoiser.
        _stack = getattr(model.diffusion, "diffusion_stack", None)
        if getattr(_stack, "cond_pair_stack", None) is not None:
            _stack.cond_pair_stack.sync(_stack.blocks)
        self.n_draws = uniform_draw_count(model.config)
        # NB skipping the inference-dead aux heads (dssp, distogram neighbours)
        # was tried and MEASURED SLOWER (6.48 -> 6.9 ms at L=100, variance
        # ~0.1 ms): the extra wrapper and the lost fusion cost more than the
        # (N, N) Gumbel it avoided. forward_core still takes `need_aux`, but the
        # fast path leaves it on.
        self._compiled = torch.compile(model.diffusion.forward_core,
                                       mode=mode, dynamic=False)
        self._static_in: dict[str, Tensor] = {}
        self._cond_cache = (None, None)
        self.need_aux = bool(p["need_aux"]) and opt.NEED_AUX
        self.lean_steps = bool(p["lean_steps"]) and opt.LEAN_STEPS
        # `opt.SKIP_PAIR_COND` is the benchmark override; it can only ever
        # turn the preset's choice OFF.
        self.allow_pair_cond_skip = bool(p["skip_pair_cond"]) and opt.SKIP_PAIR_COND
        #: whether this design actually HAS pair conditioning. None until the
        #: first real forward has seen the conditioning tensors; the blank
        #: warmup inputs must never be allowed to answer this.
        self._pc_active: bool | None = None
        #: (need_aa, pair_cond) -> (graph, static outputs), captured on demand
        self._graphs: dict[tuple[bool, bool], tuple] = {}

    # ------------------------------------------------------------------
    def _blank_inputs(self) -> dict[str, Tensor]:
        # the FLAT axis: `batch_size` designs of `self.length` residues each
        L = self.flat_length
        per = self.length
        A = self.model.diffusion.atom_count
        ls = self.config.local_size
        d, f = self.device, self.dtype
        return dict(
            pos_noised=torch.zeros(L, A, 3, device=d, dtype=f),
            prev_pos=torch.zeros(L, A, 3, device=d, dtype=f),
            prev_local=torch.zeros(L, ls, device=d, dtype=f),
            condition=torch.zeros(L, ls, device=d, dtype=f),
            dmap=torch.zeros(L, L, device=d, dtype=f),
            omap=torch.zeros(L, L, 9, device=d, dtype=f),
            dmap_mask=torch.zeros(L, L, device=d, dtype=torch.bool),
            flags=torch.zeros(L, L, 3, device=d, dtype=f),
            t_pos=torch.full((L,), 0.5, device=d, dtype=f),
            t_seq=torch.ones(L, device=d, dtype=f),
            seq_noised=torch.full((L,), 20, device=d, dtype=torch.long),
            aa_gt=torch.full((L,), 20, device=d, dtype=torch.long),
            pos_gt=torch.zeros(L, A, 3, device=d, dtype=f),
            # the packed layout, not a single 0..L-1 run: `resi` restarts per
            # design and `batch` labels them, so the blank inputs the capture
            # and the autotune timing run on have the same neighbour topology
            # the real ones will. (Every one of these is overwritten by
            # `_copy_in` before the first real replay; getting the blanks right
            # only matters for what warmup MEASURES.)
            resi=torch.arange(per, device=d).repeat(self.batch_size),
            chain=torch.zeros(L, device=d, dtype=torch.long),
            batch=torch.arange(self.batch_size, device=d,
                               dtype=torch.long).repeat_interleave(per),
            mask=torch.ones(L, device=d, dtype=torch.bool),
            # A cyclic config MUST get a real buffer here, not None. The graph
            # is captured against these addresses, so a `cyclic_mask` that
            # first appears at call time has nowhere to be written: before
            # S2 it was silently dropped (the capture had traced the
            # None branch), and after it, it raises. Capturing an all-False
            # mask and copying the real one in is the fix - it changes no
            # arithmetic, `sequence_relative_position` reads the buffer.
            cyclic_mask=(torch.zeros(L, device=d, dtype=torch.bool)
                         if self.config.cyclic else None),
            # every Gumbel draw, pre-drawn: CUDA graphs cannot capture an RNG op
            uniform=torch.rand(self.n_draws, L, L, device=d, dtype=f)
                         .clamp_(1e-6, 1 - 1e-6),
        )

    def _args(self, need_aa: bool, pair_cond: bool) -> list:
        """`forward_core`'s positional argument list.

        The three trailing flags are plain Python bools, so Dynamo specialises
        on them and each combination becomes its own graph - which is the whole
        point: a branch that is a *runtime* value would have to compute both
        sides.
        """
        return ([self._static_in.get(k) for k in _CORE_ARGS]
                + [self.need_aux, need_aa, pair_cond])

    def _build(self, need_aa: bool, pair_cond: bool, iters: int = 3) -> None:
        """Compile, and on CUDA capture, one (need_aa, pair_cond) variant."""
        key = (need_aa, pair_cond)
        if key in self._graphs:
            return
        args = self._args(need_aa, pair_cond)
        with torch.no_grad():
            for _ in range(iters):
                out = self._compiled(*args)
            if self.capture and self.manual_graph:
                # Compile in default mode and capture the entire forward once.
                # Inductor-managed reduce-overhead graphs cannot be nested.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out = self._compiled(*args)
                self._graphs[key] = (graph, static_out)
            elif self.capture:
                # NOTE: do NOT wrap an explicit torch.cuda.CUDAGraph around a
                # mode="reduce-overhead" callable. Inductor already manages
                # cudagraphs internally, and capturing on top of it raises
                # "Offset increment outside graph capture" from the RNG
                # bookkeeping. Extra warmup iterations are all that is needed.
                for _ in range(2):
                    out = self._compiled(*args)
                torch.cuda.synchronize()
                self._graphs[key] = (None, None)
            else:
                self._graphs[key] = (None, None)

    def warmup(self, iters: int = 3, pair_cond: bool | None = None
               ) -> "FastDenoiser":
        """Compile and, on CUDA, capture the graphs. One-time cost.

        `pair_cond` is a **hint only** - which variants to pre-build so the
        first sampling step does not stall on a compile. It is deliberately
        NOT stored as the gate: `forward` always re-derives that from the real
        conditioning tensors and builds the variant it actually needs, so a
        wrong hint costs compile time and can never cost correctness.
        """
        self._static_in = self._blank_inputs()
        want_pc = True if pair_cond is None else bool(pair_cond)
        self._build(True, want_pc, iters=iters)
        if self.lean_steps:
            self._build(False, want_pc, iters=iters)
        return self

    # ------------------------------------------------------------------
    def _copy_in(self, inputs: dict[str, Tensor]) -> None:
        """Write `inputs` into the persistent buffers the graph was captured on.

        Factored out of `_run` unchanged so `GroupFastDenoiser` can drive N
        members' buffers before a single grouped replay . The single-
        design path calls it from exactly where the loop used to live, so its
        behaviour - including the S2 length refusal and the clone-on-
        first-sight - is byte-for-byte what it was.
        """
        if self._static_in:
            # write into persistent buffers so the input ADDRESSES are stable.
            # Fresh tensors each call show up as
            # `cudagraph_recorded_non_static_inputs` and force a copy per input.
            for k in _CORE_ARGS:
                src, dst = inputs.get(k), self._static_in.get(k)
                if src is not None and dst is not None and dst.shape == src.shape:
                    dst.copy_(src)
                elif src is not None:
                    if self._graphs:
                        # `self._graphs`, not the former single `self._graph`:
                        # keeps one captured graph per
                        # (need_aa, pair_cond).
                        # A captured graph reads the addresses fixed at capture
                        # time, so swapping a buffer here does nothing and the
                        # replay returns the OLD length's output (red team
                        # S2: a length-48 instance fed length-49 returned
                        # 48 coordinates). Refuse rather than silently lie.
                        raise ValueError(
                            f"FastDenoiser was captured for a flat axis of "
                            f"{self.flat_length} residues "
                            f"({self.batch_size} x {self.length}) but got "
                            f"{k!r} with shape {tuple(src.shape)}; build a new "
                            f"FastDenoiser per (length, batch_size)")
                    # CLONE: adopting the caller's tensor as the persistent
                    # buffer means the next call's copy_ silently mutates the
                    # caller's data (observed: cyclic_mask flipped 0/64 -> 64/64).
                    self._static_in[k] = src.clone()

    def _run(self, inputs: dict[str, Tensor], need_aa: bool = True,
             pair_cond: bool = True) -> tuple[Tensor, ...]:
        if self.capture and not self.manual_graph:
            # tells inductor a new iteration has begun, so it may reuse the
            # cudagraph's buffers; without it, holding on to a previous output
            # raises "accessing tensor output of CUDAGraphs that has been
            # overwritten by a subsequent run"
            torch.compiler.cudagraph_mark_step_begin()
        if self._static_in:
            self._copy_in(inputs)
            args = self._args(need_aa, pair_cond)
        else:
            args = ([inputs.get(k) for k in _CORE_ARGS]
                    + [self.need_aux, need_aa, pair_cond])
        key = (need_aa, pair_cond)
        if key not in self._graphs and self._static_in:
            # a variant this warmup did not anticipate (e.g. the first design
            # turned out to carry pair conditioning): build it now, once
            self._build(need_aa, pair_cond)
        graph, static_out = self._graphs.get(key, (None, None))
        if self.manual_graph and graph is not None:
            graph.replay()
            out = static_out
        else:
            out = self._compiled(*args)
        # clone for the same reason: these buffers are reused next call
        return tuple(o.clone() if torch.is_tensor(o) else o for o in out)

    def invalidate_condition_cache(self) -> None:
        """Drop the cached conditioning; call if `data`'s condition keys change."""
        self._cond_cache = (None, None)
        self._pc_active = None
        self._cond_pinned = False

    def pin_condition(self, data: dict) -> bool:
        """Compute `prepare_condition(data)` once and stop re-keying it .

        The content hash in `_prepare_inputs` runs `.cpu()` on every
        conditioning tensor **on every step**, i.e. a device->host sync per
        step per design. That is affordable against one 3.5 ms forward and it
        is not affordable N times inside a grouped step, where the whole point
        is that the host must stay ahead of a single 13 ms replay.

        Pinning is exactly equivalent, not an approximation: `sample()` writes
        only ``pos``/``seq``/``atom_pos``/``aatype``/``t_pos``/``t_seq``/
        ``latent`` into `data`, and the hash keys on none of those - so from
        step 1 on the cache always hits anyway and the hash is pure cost. It
        is therefore safe for exactly as long as the caller promises not to
        change the conditioning without calling `invalidate_condition_cache`,
        which is the same promise `_cond_cache` already documents.

        Returns the P4/P5 `pair_cond` gate for this design.
        """
        cond = self.model.prepare_condition(data)
        self._cond_cache = (cond, "__pinned__")
        pc = cond["pair_condition"]
        self._pc_active = bool(self.config.multi_motif) or bool(
            pc["dmap_mask"].any().item() or pc["flags"].any().item())
        self._cond_pinned = True
        return self._pc_active or not self.allow_pair_cond_skip

    def forward(self, data: dict, prev: dict, *, predict_aa: bool = True,
                need_aa: bool = True,
                uniform: Tensor | None = None,
                compute_distogram: bool = False,
                generator: torch.Generator | None = None) -> tuple[dict, dict]:
        """`need_aa=False` (P2) returns only ``pos``/``local``/``trajectory``.

        The caller must be a sampling step whose ``aa``/``aatype``/``atom_pos``
        are never read - `salad_torch.diffusion.sampling.sample` decides that
        from `out_steps`. The keys are ABSENT rather than stale, so a consumer
        that needs them fails with a KeyError instead of silently reading the
        previous step's sequence.

        `predict_aa` is unrelated and unchanged: it selects whose amino acids
        the sidechains are built from.
        """
        inputs, pair_cond = self._prepare_inputs(
            data, prev, uniform=uniform, generator=generator)
        out = self._run(inputs, need_aa=need_aa, pair_cond=pair_cond)
        return self._postprocess(out, data, need_aa=need_aa,
                                 predict_aa=predict_aa)

    # ------------------------------------------------------------------
    # forward() split in three so `GroupFastDenoiser` can interleave
    # N members' host-side work around ONE grouped replay. The bodies are
    # verbatim what `forward` used to run, in the same order.
    # ------------------------------------------------------------------
    def _prepare_inputs(self, data: dict, prev: dict, *,
                        uniform: Tensor | None = None,
                        generator: torch.Generator | None = None,
                        ) -> tuple[dict, bool]:
        if uniform is None:
            # drawn OUTSIDE the graph, then copied into the static buffer
            # (draws, N, N) over the FLAT axis: with batch_size=B this is
            # B^2 times the field a single design draws, which is one of the
            # two terms that make packed batching superlinear (the other is
            # the (N, N) stable argsort in `extract_neighbours`).
            uniform = torch.rand(self.n_draws, self.flat_length, self.flat_length,
                                 device=self.device, dtype=self.dtype,
                                 generator=generator).clamp_(1e-6, 1 - 1e-6)
        if getattr(self, "_cond_pinned", False):
            # `pin_condition` has already computed this design's conditioning
            # and its P4/P5 gate; skip the per-step hash entirely .
            cond = self._cond_cache[0]
            pc = cond["pair_condition"]
            pair_cond = self._pc_active or not self.allow_pair_cond_skip
            return self._pack_inputs(data, prev, cond, pc, uniform), pair_cond
        # prepare_condition is LOOP-INVARIANT: it depends only on the
        # conditioning inputs and the chain/batch layout, none of which change
        # across the 500 denoising steps of one design. Running it every step
        # cost more than the whole compiled model. Cached per design; call
        # invalidate_condition_cache() if the conditioning changes.
        # KEY the cache on the conditioning CONTENT.
        #
        # This used to sum each tensor, which collides on ANY permutation:
        # swapping two DSSP labels leaves the sum identical, so the second
        # design silently reused the first design's conditioning and came out
        # 2.56 A wrong while looking healthy (red team S1).
        # A content hash of the raw bytes cannot collide that way. The cost is
        # one hash per step over a few (L, L) tensors, which is negligible
        # against the ~5 ms forward and is still far cheaper than recomputing
        # prepare_condition.
        import hashlib

        digest = hashlib.blake2b(digest_size=16)
        for k in ("aa_condition", "dssp_condition", "dssp_mean", "dssp_mean_mask",
                  "dmap", "dmap_mask", "omap", "chain_contacts",
                  "chain_index", "batch_index", "mask", "residue_index"):
            v = data.get(k)
            digest.update(k.encode())
            if v is None:
                digest.update(b"\x00")
            else:
                digest.update(str(tuple(v.shape)).encode())
                digest.update(v.detach().cpu().contiguous().numpy().tobytes())
        key = digest.hexdigest()

        cond, cached_key = getattr(self, "_cond_cache", (None, None))
        if cond is None or cached_key != key:
            cond = self.model.prepare_condition(data)
            self._cond_cache = (cond, key)
            self._pc_active = None
        pc = cond["pair_condition"]
        if self._pc_active is None:
            # P4/P5's runtime gate, derived from the REAL tensors once per
            # design (two host syncs, against 500 steps). `flags` channel 0 is
            # "these two residues are on different chains", so anything
            # multi-chain, any template/motif dmap and any explicit
            # chain_contacts all switch the skip off by themselves.
            self._pc_active = bool(self.config.multi_motif) or bool(
                pc["dmap_mask"].any().item() or pc["flags"].any().item())
        pair_cond = self._pc_active or not self.allow_pair_cond_skip
        return self._pack_inputs(data, prev, cond, pc, uniform), pair_cond

    def _pack_inputs(self, data: dict, prev: dict, cond: dict, pc: dict,
                     uniform: Tensor) -> dict:
        return dict(
            pos_noised=data["pos_noised"], prev_pos=prev["pos"],
            prev_local=prev["local"], condition=cond["condition"],
            dmap=pc["dmap"], omap=pc["omap"],
            dmap_mask=pc["dmap_mask"], flags=pc["flags"],
            t_pos=data["t_pos"], t_seq=data["t_seq"],
            seq_noised=data["seq_noised"], aa_gt=data["aa_gt"],
            pos_gt=data["pos"], resi=data["residue_index"],
            chain=data["chain_index"], batch=data["batch_index"], mask=data["mask"],
            cyclic_mask=data.get("cyclic_mask") if self.config.cyclic else None,
            uniform=uniform,
        )
        return inputs, pair_cond

    def _postprocess(self, out: tuple, data: dict, *, need_aa: bool,
                     predict_aa: bool) -> tuple[dict, dict]:
        if not need_aa:
            result = dict(zip(_OUT_NAMES_LEAN, out))
            result["pos_noised"] = data["pos_noised"]
            result["aa_gt"] = data["aa_gt"]
            return result, {"pos": result["pos"], "local": result["local"]}
        result = dict(zip(_OUT_NAMES, out))
        result["atom_pos"] = result.pop("angle_pos")
        result["pos_noised"] = data["pos_noised"]
        result["aa_gt"] = data["aa_gt"]
        if "aa_condition" in data:
            result["aatype"] = torch.where(
                data["aa_condition"] != 20, data["aa_condition"], result["aatype"])
        if not predict_aa or "aa_condition" in data:
            # atom_pos was built from the predicted aatype inside the graph, so
            # it no longer matches; rebuild it rather than return an
            # inconsistent (aatype, atom_pos) pair (measured 7.7 A apart).
            if not predict_aa:
                result["aatype"] = data["aa_gt"]
            with torch.no_grad():
                raw, ang, apos = self.model.diffusion.angle_head(
                    result["aatype"], result["local"], result["pos"])
            result["raw_angles"], result["angles"], result["atom_pos"] = raw, ang, apos
        return result, {"pos": result["pos"], "local": result["local"]}


# ======================================================================
# - N independent designs in ONE CUDA graph
# ======================================================================
#: Hard ceiling on the auto-chosen group size. It is a *safety* cap on the
#: search, not a tuned value: `--group_size N` overrides it in both
#: directions, and `autosize` almost always lands below it. Kept modest so a
#: first run on an unknown card cannot spend minutes capturing a graph that
#: will not fit.
_AUTOSIZE_CAP = 16
#: Fraction of FREE device memory `autosize` is willing to spend on the
#: grouped graph's activation pool. The rest is headroom for the sampler's own
#: per-design tensors, the snapshots, fragmentation and whatever else shares
#: the card.
_AUTOSIZE_MEM_FRACTION = 0.6


def memory_bound_group(per_member_bytes: int | None, device=None,
                       cap: int = _AUTOSIZE_CAP) -> int:
    """How many members FREE device memory can hold. Portable, no card in it.

    ``1 + free_bytes * _AUTOSIZE_MEM_FRACTION / per_member_bytes``, clamped to
    `cap`. `per_member_bytes` is MEASURED on the real model by
    `GroupFastDenoiser._probe_member_cost`, never estimated from a shape, and
    the fraction leaves the rest of the card to the sampler's own per-design
    tensors, the snapshots, fragmentation and anything else sharing the GPU.

    On a 24 GB consumer card this, not occupancy, is normally what decides the
    group size - which is the whole reason it is measured.
    """
    if not per_member_bytes or per_member_bytes <= 0:
        return cap
    try:
        free_bytes, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return 1
    return max(1, min(cap, 1 + int(free_bytes * _AUTOSIZE_MEM_FRACTION
                                   // per_member_bytes)))


#: The group sizes `auto` will try, in order. A ladder rather than every
#: integer because each rung costs a capture, and the curve is smooth.
_AUTOSIZE_LADDER = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
#: `auto` stops climbing when a rung improves per-design time by less than
#: this. Not a performance constant - a stopping rule.
_AUTOSIZE_MIN_GAIN = 0.03


class GroupFastDenoiser(nn.Module):
    """`group_size` independent designs advanced through ONE captured graph.

    This is **not** a batch axis. salad's flat residue layout would let you
    concatenate designs along it with `batch_index`, and that is a different
    thing with different arithmetic. Here each member keeps its own complete
    set of static input buffers and its own copy of `forward_core`; the
    members are recorded into one `torch.cuda.CUDAGraph` on
    ``group_size`` separate streams, so a single `replay()` submits all of
    them and the driver overlaps whichever of their kernels it likes.

    Each member therefore computes **exactly** what it would compute alone -
    same kernels, same shapes, same inputs, no shared reduction - and the only
    thing that changes is that the GPU is no longer idle between one design's
    serialised launches. measured a single L=100 forward leaving an H100
    roughly 55 % idle.

    Portability, because this is squarely the kind of optimisation that gets
    over-fitted to one card:

    * `group_size` is a runtime parameter, never a constant, and
      ``group_size="auto"`` MEASURES it on the device in hand (`_autotune`)
      rather than evaluating a formula fitted to one card;
    * ``group_size=1`` captures a single-stream body byte-for-byte identical
      to `FastDenoiser`'s, so degrading to "no grouping" is the same code
      path, not a fallback branch;
    * everything here is plain `torch.cuda.Stream` + `torch.cuda.CUDAGraph` +
      `torch.compile`. No architecture-gated kernels, no PTX, nothing Hopper-
      specific. Inductor autotunes the member forward for whatever card it
      lands on;
    * a capture that runs out of memory halves the group and retries, down to
      1, rather than failing the run.
    """

    def __init__(self, model, length: int, group_size: int | str = "auto",
                 device="cuda", mode: str = "default",
                 dtype=torch.float32, preset: str = "exact",
                 autosize_cap: int = _AUTOSIZE_CAP, batch_size: int = 1):
        super().__init__()
        self.length = int(length)
        # each MEMBER may itself be a packed batch of `batch_size`
        # designs. The two axes compose and are orthogonal - `batch_size`
        # makes each kernel bigger, `group_size` runs more kernels at once -
        # so one replay advances ``group_size * batch_size`` designs.
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GroupFastDenoiser needs a CUDA device")
        self.preset = preset
        self._requested = group_size
        self._autosize_cap = int(autosize_cap)
        if mode == "reduce-overhead":
            # Same mutual exclusion `FastDenoiser.__init__` documents, and it
            # bites harder here: in reduce-overhead inductor manages its own
            # cudagraphs, and recording an explicit capture on top raises
            # "Cannot prepare for replay during capturing stage". The group
            # OWNS the capture, so the member forward must be a plain compile.
            mode = "default"
        # The FIRST member does all the process-global setup (TF32, DETSEG,
        # aggressive_fusion, BATCH_COND) and owns the ONE compiled callable;
        # every other member borrows it, so inductor compiles once and the
        # members differ only in which buffers they point at.
        self._head = FastDenoiser(model, self.length, device=self.device,
                                  mode=mode, dtype=dtype, capture=False,
                                  manual_graph=False, preset=preset,
                                  batch_size=self.batch_size)
        self.model = model
        self.config = model.config
        self.members: list[FastDenoiser] = [self._head]
        self.group_size = 1
        self._graphs: dict[tuple[bool, bool], tuple] = {}
        #: measured, per member: activation high-water of one compiled forward
        self.per_member_bytes: int | None = None
        #: filled in by `warmup`, reported by `scripts/generate.py`
        self.peak_bytes: int | None = None

    @property
    def designs_per_replay(self) -> int:
        """How many designs one `replay()` advances: group_size x batch_size."""
        return self.group_size * self.batch_size

    # ------------------------------------------------------------------
    def _add_members(self, n: int) -> None:
        head = self._head
        while len(self.members) < n:
            m = FastDenoiser(head.model, self.length, device=self.device,
                             mode="default", dtype=head.dtype, capture=False,
                             manual_graph=False, preset=self.preset,
                             batch_size=self.batch_size)
            # ONE compiled callable for the whole group: a second
            # `torch.compile` wrapper would get its own code cache and
            # recompile the identical graph from scratch.
            m._compiled = head._compiled
            m._static_in = m._blank_inputs()
            self.members.append(m)
        del self.members[n:]
        self.group_size = n

    def _probe_member_cost(self, pair_cond: bool = True) -> int:
        """Activation high-water of ONE compiled forward, measured not guessed.

        Takes the `pair_cond` the run will actually use: probing the other
        variant would compile a third graph nobody replays (~a minute).

        It also has a second job, and the order in `warmup` is load-bearing
        because of it: this is the FULL-AA path (``need_aa=True``), so running
        it first populates `salad_torch.geometry.sidechain._DEVICE_CACHE`,
        which is lazily filled on first use and - being a python global read
        inside the traced region - would otherwise flip a Dynamo guard and
        trigger a recompile the first time the angle head ran. A recompile
        during a `torch.cuda.graph` capture is a raise or garbage. `_autotune`
        and every `_capture` run after this, so by then the table exists.
        """
        head = self._head
        args = head._args(True, bool(pair_cond))
        with torch.no_grad():
            # compile + autotune first: measuring the FIRST call would fold
            # inductor's benchmarking scratch into the per-member footprint
            # and propose an absurdly small group.
            head._compiled(*args)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.no_grad():
            head._compiled(*args)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(self.device)
        # a floor, so a mis-measurement can never divide by ~0 and propose a
        # thousand-member group
        return max(int(peak - base), 1 << 20)

    # ------------------------------------------------------------------
    def _capture(self, need_aa: bool, pair_cond: bool, iters: int = 3):
        """Record `group_size` member forwards into one graph, one per stream."""
        compiled = self._head._compiled
        argsets = [m._args(need_aa, pair_cond) for m in self.members]
        n = self.group_size
        streams = ([torch.cuda.Stream(device=self.device) for _ in range(n)]
                   if n > 1 else None)

        def body():
            if streams is None:
                # group_size == 1: the SAME single-stream body `FastDenoiser`
                # captures. Not a special case for speed - a guarantee that
                # "no grouping" is not a different computation.
                return [compiled(*argsets[0])]
            cur = torch.cuda.current_stream()
            for s in streams:
                s.wait_stream(cur)
            outs = []
            for i in range(n):
                with torch.cuda.stream(streams[i]):
                    outs.append(compiled(*argsets[i]))
            for s in streams:
                cur.wait_stream(s)
            return outs

        with torch.no_grad():
            # THREE warm calls before the side-stream warm, the same count
            # `FastDenoiser._build` uses and for the same reason: a partially
            # warmed callable captures a *recompile*, which is either a raise
            # or silent garbage. Every member is warmed, not just the head -
            # they share one compiled callable but each has its own buffers.
            for _ in range(iters):
                body()
            torch.cuda.synchronize()
            # CUDA graph capture of multi-stream work must itself be warmed up
            # on a non-default stream first, or the capture sees allocator
            # state it did not record.
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                body()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # DELIBERATELY not sharing one memory pool across the
            # (need_aa=True, need_aa=False) variants, although that would
            # roughly halve the graph's footprint and footprint is the binding
            # limit on a 24 GB consumer card. Graphs that share a pool are
            # only safe when they are replayed in their capture order, and
            # this sampler replays the lean variant ~200 times and the full
            # one once, interleaved - exactly the pattern the shared-pool
            # idiom does not cover. Silent corruption is not worth the GiB;
            # `warmup` reports the real peak so a caller can size the group
            # against what it actually costs.
            with torch.cuda.graph(graph):
                outs = body()
        return graph, outs

    def _build(self, need_aa: bool, pair_cond: bool) -> None:
        key = (need_aa, pair_cond)
        if key in self._graphs:
            return
        while True:
            try:
                self._graphs[key] = self._capture(need_aa, pair_cond)
                return
            except RuntimeError as exc:
                # `torch.cuda.OutOfMemoryError` is a RuntimeError subclass,
                # but a capture that runs out of memory inside the graph's
                # private pool has also been seen to surface as a plain
                # RuntimeError carrying the driver's text - and the text is
                # all there is to go on. Anything else re-raises untouched.
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and\
                        "out of memory" not in str(exc).lower():
                    raise
                if self.group_size <= 1:
                    raise
                new_n = max(1, self.group_size // 2)
                print(f"  group capture OOM at group_size={self.group_size}; "
                      f"retrying at {new_n}")
                # every previously captured variant was recorded against the
                # old member count, so they all have to go with it
                self._graphs.clear()
                self._add_members(new_n)
                torch.cuda.empty_cache()

    def warmup(self, pair_cond: bool | None = None,
               need_aa_variants: Sequence[bool] = (True, False)
               ) -> "GroupFastDenoiser":
        """Choose the group size, compile once, capture the variants."""
        self._head._static_in = self._head._blank_inputs()
        want_pc = True if pair_cond is None else bool(pair_cond)
        self.per_member_bytes = self._probe_member_cost(want_pc)
        if isinstance(self._requested, str):
            if self._requested != "auto":
                raise ValueError(f"group_size must be an int or 'auto', "
                                 f"got {self._requested!r}")
            n = self._autotune(want_pc)
        else:
            n = max(1, int(self._requested))
        return self._capture_variants(n, pair_cond, need_aa_variants)

    def _autotune(self, pair_cond: bool, reps: int = 30) -> int:
        """MEASURE the group size on this device. No formula, no calibration.

        The win is spare GPU capacity, and how much of that a card has depends
        on its SM count, its clocks, its memory system and the driver - not on
        any one number that can be written down here. An analytic rule fitted
        to the H100 this was developed on is precisely the H100
        specialisation the design is supposed to avoid, and it would be wrong
        in an unknown direction on a 4090, a 5090 or whatever comes next.

        So `auto` climbs `_AUTOSIZE_LADDER`, captures the graph at each rung
        (a capture, NOT a recompile - the compiled forward is shared, so a
        rung costs seconds), times the replay that the sampler actually
        spends ~99 % of its steps in, and stops when a rung buys less than
        `_AUTOSIZE_MIN_GAIN` per design or when memory says stop. The whole
        search is a handful of seconds against a one-time compile of a
        minute or more, and it adapts to the card by construction.

        Degrading to 1 is a normal outcome, not a failure: on a card with no
        spare capacity - or at a length where one design already fills it -
        the first rung wins and the caller gets the ordinary single-design
        computation.
        """
        cap = min(self._autosize_cap,
                  memory_bound_group(self.per_member_bytes, self.device,
                                     cap=self._autosize_cap))
        # time the variant the sampler actually spends ~99 % of its steps in
        probe_need_aa = not self._head.lean_steps
        best_n, best_per = 1, None
        for n in _AUTOSIZE_LADDER:
            if n > cap:
                break
            try:
                self._graphs.clear()
                self._add_members(n)
                torch.cuda.empty_cache()
                graph, _outs = self._capture(probe_need_aa, pair_cond)
            except RuntimeError as exc:
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and\
                        "out of memory" not in str(exc).lower():
                    raise
                break
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(reps):
                graph.replay()
            e.record()
            torch.cuda.synchronize()
            # per DESIGN, not per member: with packed batching one member
            # carries `batch_size` designs.
            per = s.elapsed_time(e) / reps / (n * self.batch_size)
            del graph, _outs
            if best_per is None or per < best_per * (1.0 - _AUTOSIZE_MIN_GAIN):
                best_n, best_per = n, per
            else:
                # this rung did not pay for itself; the curve is monotone
                # enough that the next one will not either
                break
        self._graphs.clear()
        torch.cuda.empty_cache()
        return best_n

    def resize(self, n: int, pair_cond: bool | None = None,
               need_aa_variants: Sequence[bool] = (True, False)
               ) -> "GroupFastDenoiser":
        """Re-capture at a different group size, reusing the compiled forward.

        The trailing partial group of a `--num_designs` run needs this: a
        group of 3 cannot replay a graph recorded for 8. It costs one capture
        (seconds), not one compile (a minute) - `self._head._compiled` and
        every member's buffers survive.
        """
        return self._capture_variants(max(1, int(n)), pair_cond,
                                      need_aa_variants)

    def _capture_variants(self, n: int, pair_cond, need_aa_variants
                          ) -> "GroupFastDenoiser":
        self._graphs.clear()
        self._add_members(n)
        torch.cuda.empty_cache()
        want_pc = True if pair_cond is None else bool(pair_cond)
        torch.cuda.reset_peak_memory_stats(self.device)
        for need_aa in need_aa_variants:
            if need_aa or self._head.lean_steps:
                self._build(bool(need_aa), want_pc)
        self.peak_bytes = int(torch.cuda.max_memory_reserved(self.device))
        return self

    # ------------------------------------------------------------------
    def invalidate_condition_cache(self) -> None:
        for m in self.members:
            m.invalidate_condition_cache()

    def pin_conditions(self, datas: Sequence[dict]) -> bool:
        """Pin each member's conditioning and return the group's `pair_cond`.

        The gate is OR-ed over the members because one graph carries one
        variant of it. That is not a numerical compromise: `pair_cond=True` is
        the FULL computation and P4/P5 (`salad_torch.opt.SKIP_PAIR_COND`) are
        defined as dropping terms that are exactly zero for a design without
        pair conditioning - so a member that would have run with the skip gets
        the same answer without it, in a different reduction order over K.
        With ``group_size=1`` the OR is the member's own gate and nothing
        changes at all.
        """
        gates = [m.pin_condition(d) for m, d in zip(self.members, datas)]
        return any(gates)

    def forward(self, datas: Sequence[dict], prevs: Sequence[dict], *,
                predict_aa: bool = True, need_aa: bool = True,
                pair_cond: bool | None = None,
                generators: Sequence[torch.Generator] | None = None,
                ) -> tuple[list[dict], list[dict]]:
        n = self.group_size
        if len(datas) != n or len(prevs) != n:
            raise ValueError(f"expected {n} designs, got {len(datas)} / "
                             f"{len(prevs)}")
        gens = list(generators) if generators is not None else [None] * n
        gate = pair_cond
        for i, m in enumerate(self.members):
            inputs, g = m._prepare_inputs(datas[i], prevs[i], generator=gens[i])
            gate = g if gate is None else (gate or g)
            m._copy_in(inputs)
        key = (need_aa, bool(gate))
        if key not in self._graphs:
            self._build(need_aa, bool(gate))
            # a capture-time OOM may have shrunk the group under us
            if self.group_size != n:
                raise RuntimeError(
                    f"group size dropped {n} -> {self.group_size} while "
                    f"capturing {key}; restart the run at the smaller size")
        graph, outs = self._graphs[key]
        graph.replay()
        results, new_prevs = [], []
        for i, m in enumerate(self.members):
            # clone: the next replay overwrites these buffers
            out = tuple(o.clone() if torch.is_tensor(o) else o for o in outs[i])
            r, p = m._postprocess(out, datas[i], need_aa=need_aa,
                                  predict_aa=predict_aa)
            results.append(r)
            new_prevs.append(p)
        return results, new_prevs

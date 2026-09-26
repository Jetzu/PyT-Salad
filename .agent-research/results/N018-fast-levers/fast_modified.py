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

import torch
from torch import Tensor, nn

from .. import opt
from .diffusion import uniform_draw_count

__all__ = ["FastDenoiser", "PRESETS"]

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
    ),
    # ------------------------------------------------------------------
    # "fast" (N011) - maximum speed, ALLOWED to move the numbers. It is NOT a
    # different model: same checkpoint, same architecture, same neighbourhood
    # sizes, same sampler. What it gives up is mantissa bits in the matmuls and
    # half the trajectory's resolution - and nothing else, because nothing else
    # that was tried paid for itself. Measured at L=100 on an H100, production
    # path, `x_preset.py` (.agent-research/results/N011-fast-preset/):
    #
    #                        reporting step   lean step   per design
    #   exact                    4.381 ms      3.568 ms     1.43 s
    #   exact + TF32 only        3.719 ms      2.996 ms     1.20 s
    #   fast without TF32        4.385 ms      3.563 ms     0.72 s
    #   fast                     3.614 ms      2.894 ms     0.58 s
    #
    # i.e. TF32 is -16 to -19 % of the step and halving the schedule is the
    # other 2x. JAX salad is 1.56 s per design, so this is ~2.7x JAX and
    # ~2.5x the exact path.
    #
    # QUALITY: validated at n=160 designs per arm, not at the n=8 this preset
    # shipped on (N015, jobs 903264 + 903296; L=100, seeds 42 and 1234 pooled).
    # Halving the schedule is the only knob here that changes the TRAJECTORY
    # rather than the arithmetic, so it was factorised out and tested on its
    # own. Converged designs, `exact` in brackets:
    #
    #                      CA-CA (A)        Rg (A)      helix  strand  diverged
    #   exact         3.7628 +- 0.0137  12.89 +- 1.76   0.566   0.353    2/160
    #   TF32 only     3.7615 +- 0.0123  12.89 +- 1.46   0.572   0.351    0/160
    #   schedule only 3.7627 +- 0.0141  12.89 +- 1.56   0.575   0.345    0/160
    #   fast          3.7613 +- 0.0131  12.90 +- 1.89   0.580   0.335    0/160
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
    # The n=8 table this preset shipped on (N011 section 4) read helix 0.605 ->
    # 0.591 and Rg 12.50 -> 12.92, i.e. exactly the feared direction. Both were
    # noise, and so was an n=40 rerun that put the fast arm's Rg sd at 2.3x
    # exact's - it came from 3 designs and vanished at n=160. Per-design SSE
    # variance here is sd ~0.35, so anything below n~100 says nothing.
    #
    # DIVERGED designs (CA-CA mean > 4.5 A) are a failure mode of the MODEL,
    # not of this preset: 2/160 on `exact` (worst CA-CA 28.3 A, Rg 408 A),
    # 0/160 on all three deviating arms, Fisher p=0.50. Upstream JAX produced
    # 1/40 (N001 section 10). Filter on CA-CA downstream, on either preset.
    #
    # REFUTED and deliberately absent - each measured, none of them free:
    #   * K 64 -> 32 on both attention branches: -10 % of the step, but coil
    #     0.076 -> 0.109 and CA-CA clashes 0.1 -> 1.1 per design. The only
    #     candidate that visibly degraded the structures. Dropped.
    #   * one shared self-conditioning Gumbel draw for all six blocks:
    #     2.894 vs 2.882 ms, i.e. NOTHING, measured with the switch verified
    #     live (6.45 A change in `pos`). A deviation with no payoff.
    #   * bypassing the conditioning-cache key: 3.563 vs 3.568 ms, nothing -
    #     and after N010/S1 a weakened cache key is a bug, not a trade.
    #   * `topk` for the stable argsort (+5.6 %), `need_aux=False` (no change),
    #     bf16 inside the GEMM via matmul_precision("medium") (2.856 vs 2.760,
    #     i.e. slower than TF32). See RESULTS.md section 5.
    "fast": dict(
        lean_steps=True,
        skip_pair_cond=True,
        need_aux=True,
        # TF32: 10 mantissa bits instead of 24 in every fp32 matmul. The single
        # largest lever in the preset, and the one the exact path must refuse -
        # N005 measured TF32 moving coordinates by ~6e-3 A while leaving
        # `aatype` untouched, i.e. it looks fine and is not fp32. Here that is
        # the point. For scale: CUDA's own run-to-run nondeterminism on this
        # model is 2.2e-04 A (N010), so TF32 is a real perturbation, ~1500x the
        # noise floor - but three orders of magnitude smaller than what
        # shrinking the neighbourhood does.
        tf32=True,
        # halve the sampler: 250 steps reporting at 200 instead of 500/400.
        # The schedule is `raw_t = 1 - step/num_steps`, so halving BOTH keeps
        # the trajectory's endpoints (raw_t 1.0 -> 0.2) and only coarsens it.
        # `scripts/generate.py` applies this, and only when the command line
        # did not set --num_steps/--out_steps itself.
        #
        # 0.5 STAYS, on N015's n=160-per-arm test (see QUALITY above): the
        # schedule-only arm is indistinguishable from `exact` on mean, spread
        # and distribution shape, so there is nothing to buy by backing off to
        # 0.75, and nothing to isolate by splitting this into a second preset.
        # A "faster" preset at 125/100 is NOT offered: N011 section 4 found it
        # passes the same structural gate, but a 4x cut deserves a
        # designability check (ProteinMPNN + a folding model), which no arm in
        # this port has run. Structural gates cannot answer designability.
        steps_scale=0.5,
    ),
    # ------------------------------------------------------------------
    # N018 EXPERIMENTS (this worktree only, NOT landed in the shared tree) -
    # re-test N011's "K 64 -> 32" rejection at adequate n, plus a milder cut this
    # task proposes. Both stack on top of "fast"'s TF32; steps_scale is left
    # at 1.0 here on purpose so a speed comparison against these presets
    # isolates the neighbourhood lever from the schedule lever (the brief's
    # explicit requirement) - callers wanting the schedule cut too still pass
    # --num_steps/--out_steps or use "fast" itself.
    "fast_halfnbr": dict(
        lean_steps=True, skip_pair_cond=True, need_aux=True, tf32=True,
        steps_scale=1.0, nbr_mode="half",
    ),
    "fast_randhalf": dict(
        lean_steps=True, skip_pair_cond=True, need_aux=True, tf32=True,
        steps_scale=1.0, nbr_mode="randhalf",
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
                 preset: str = "exact"):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
        self.preset = preset
        p = PRESETS[preset]
        self.model = model
        self.config = model.config
        self.length = length
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
        # N018 EXPERIMENT (this worktree only): same PROCESS-GLOBAL / set-in-
        # BOTH-directions pattern as the TF32 line above, for the same reason
        # - a preset without "nbr_mode" must put the process back on "full"
        # rather than silently inherit a previous preset's shrink. Read by
        # `Diffusion.forward_core` at trace time (see opt.NBR_MODE), so it has
        # to be set before `torch.compile` below.
        opt.NBR_MODE = p.get("nbr_mode", "full")
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
        L = self.length
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
            resi=torch.arange(L, device=d),
            chain=torch.zeros(L, device=d, dtype=torch.long),
            batch=torch.zeros(L, device=d, dtype=torch.long),
            mask=torch.ones(L, device=d, dtype=torch.bool),
            # A cyclic config MUST get a real buffer here, not None. The graph
            # is captured against these addresses, so a `cyclic_mask` that
            # first appears at call time has nowhere to be written: before
            # N010/S2 it was silently dropped (the capture had traced the
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
    def _run(self, inputs: dict[str, Tensor], need_aa: bool = True,
             pair_cond: bool = True) -> tuple[Tensor, ...]:
        if self.capture and not self.manual_graph:
            # tells inductor a new iteration has begun, so it may reuse the
            # cudagraph's buffers; without it, holding on to a previous output
            # raises "accessing tensor output of CUDAGraphs that has been
            # overwritten by a subsequent run"
            torch.compiler.cudagraph_mark_step_begin()
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
                        # N009 keeps one captured graph per
                        # (need_aa, pair_cond).
                        # A captured graph reads the addresses fixed at capture
                        # time, so swapping a buffer here does nothing and the
                        # replay returns the OLD length's output (red team
                        # N010/S2: a length-48 instance fed length-49 returned
                        # 48 coordinates). Refuse rather than silently lie.
                        raise ValueError(
                            f"FastDenoiser was captured for length {self.length} "
                            f"but got {k!r} with shape {tuple(src.shape)}; build "
                            f"a new FastDenoiser per length")
                    # CLONE: adopting the caller's tensor as the persistent
                    # buffer means the next call's copy_ silently mutates the
                    # caller's data (observed: cyclic_mask flipped 0/64 -> 64/64).
                    self._static_in[k] = src.clone()
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
        if uniform is None:
            # drawn OUTSIDE the graph, then copied into the static buffer
            uniform = torch.rand(self.n_draws, self.length, self.length,
                                 device=self.device, dtype=self.dtype,
                                 generator=generator).clamp_(1e-6, 1 - 1e-6)
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
        # 2.56 A wrong while looking healthy (red team N010/S1).
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
        inputs = dict(
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
        out = self._run(inputs, need_aa=need_aa, pair_cond=pair_cond)
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

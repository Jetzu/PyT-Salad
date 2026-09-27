# N031 · P7 fusion miscompile: exact kernel + tested fix

type: task · date: 2026-09-27 · status: backfilled (done, fix not yet integrated)

Backfilled by work-lint 2026-09-27: `results/N031-p7-fusion-rootcause/codex/`
(34 MB, no file >5 MB) existed with a full `RESULTS.md` but no INDEX row or
item file. The folder was previously named `N026-p7-fusion-rootcause` -
`codex/RESULTS.md` self-identifies as "N026 Codex root cause and fix round",
but that ID was already taken by N026 (Kale feasibility) by the time this
work needed to register - the same concurrent-agent ID race documented in
N030's item file (whose brief was also originally labelled "N027"). Renamed
to N031 (the next free ID) and backfilled here; log/script contents inside
`codex/` still say "N026" in places (job labels, filenames) and were left
untouched as the historical record of what was actually run.

This is a Codex red-team/fix-round investigation, per the four-role
workflow: writes confined to `codex/`, no production file edited, no
commit/PR. It directly extends N025 (which identified P7's
`SALAD_OPT_BATCHCOND` prologue as the whole regression and an inductor
fusion miscompile as the mechanism, but left the exact bad kernel
unidentified - see TODO T9).

## Goal
Find the exact miscompiled fusion inside N025's already-narrowed compiled
`forward_core` region, and produce a tested, minimal fix.

## Evidence
Rebuilt N025's real step-30 L=100 state (`DET_SEGMENT=1`, fp32,
`torch.compile(mode="default", dynamic=False)`, no CUDA graph). An
accepted-fusion prefix bisect found the exact first bad decision: prefix 14
gives `pos` error 1.86e-05 A, prefix 15 gives **9.586393 A**
(`bisect_907076.log`). The 15th fusion decision is `op195 + op207`: `op195`
writes an int64 `[600,48]` six-block condition-neighbour `stack`, `op207`
reads it as indirect residue/chain/batch indices for tiled relative-position
bins. Vetoing only that pair in an otherwise fully-fused graph gives
1.81e-05 A. In the fused kernel
(`triton_poi_fused_add_clamp_eq_full_like_index_repeat_stack_sub_where_72`),
three indirect `tl.load`s are guarded by a mask ANDing `tmp4` (`row < 100`)
with `tmp36` (`100 <= row < 200`) - always false - so on every lane the wrong
integer bin is read; the unfused kernel's equivalent loads use only `xmask`.
Standalone replay of the exact captured fused vs. unfused kernels on real
captured inputs confirms it bit-for-bit: relative-position bins differ in
**23,070/28,800** positions, max error 64 bins, everything else (the stack
output itself) matches exactly. A no-SALAD minimal PyTorch reproducer
(`minimal_conditional.py`) gets 22,222/28,800 int64 bins wrong by default and
0 wrong at `max_fusion_size=1` - an upstream Inductor bug, not
SALAD-specific (`UPSTREAM_ISSUE.md` drafted).

## Outcome
**Root cause: PyTorch 2.5.1 Inductor miscompiles the fusion of `op195`
(stack) with `op207` (indirect gather)**, not the LayerNorm TODO T9
speculated as the "prime suspect" (that guess is now superseded - the actual
site is the stack+gather pair, unrelated to the prologue's LayerNorm).
**Tested fix**: a single `torch._dynamo.graph_break()` right after
`cond_nbr_stack = torch.stack(...)` in `Diffusion.forward_core`, before
`StackedCondPair` computes relative positions (`fix.patch`, checked against
the branch index). With the graph break: `pos` max abs error 1.53e-05 A
vs eager P7-on (matches the P7-off compile floor of ~2e-05 A). Required GPU
quality gate **0/160 diverged** at L=100 `--preset fast`, CA-CA 3.7631 A, Rg
13.05 A. Whole-forward lean step at L=100 (3x30-iter reps): P7 off 2.8606 ms,
broken P7 on 2.6660 ms, **fixed P7 on 2.6755 ms** - the fix is 6.47% faster
than P7-off and keeps 95.1% of P7's measured launch-saving benefit, at 0.36%
cost vs. the broken (wrong) version. CPU checkpoint parity is INCOMPLETE:
three of four configs passed under `DET_SEGMENT=1` (a setting that differs
from the production checkpoint-parity gate and changes reduction order);
`default_ve_scaled` missed only on `atom_pos` tolerance, and a
`DET_SEGMENT=0` control run was never completed.

## Outputs
`results/N031-p7-fusion-rootcause/codex/`: `RESULTS.md` (full writeup),
`fix.patch` (the tested one-line graph-break fix, not yet applied to any
production file), `UPSTREAM_ISSUE.md` (draft PyTorch bug report),
`minimal_conditional.py` (no-SALAD repro), `bisect_forward.py`/
`capture_kernel.py`/`standalone_kernel.py` (the bisection and standalone
kernel-replay tooling), `on.kernels.tsv`/`off.kernels.tsv`/
`fusion_decisions.diff`, `real_kernel_io.pt` (captured kernel I/O), per-step
`.sbatch`/`.log` pairs, `trace_on.tar.gz`/`trace_off.tar.gz`/
`trace_lim15.tar.gz`.

## Next
**Fix not yet integrated** - `codex/RESULTS.md`'s own integration note says
the live tree gained separate, uncommitted P7 edits while these jobs ran, so
`fix.patch` should not be applied blindly and needs reconciling with current
HEAD before a Claude integrator lands it (per the Codex protocol: reproduce
first-hand before any code change). Needs: (1) apply/reconcile `fix.patch`
against current HEAD, (2) complete the CPU checkpoint-parity control at
`DET_SEGMENT=0` to match the production gate exactly, (3) re-run the GPU
quality/benchmark gates on the integrated tree, (4) file the upstream
PyTorch issue if still reproducible on the pinned version, (5) update TODO
T9 (done by this lint - see TODO.md) to stop pointing at the LayerNorm as
the suspect kernel.

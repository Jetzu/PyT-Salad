# N002 · Generation-loop CLI flags (spec 06): symmetry, template, renoise, order-agnostic

type: task · date: 2026-09-25 · status: done

## Goal

Implement and verify the `scripts/generate.py` flags specified in
`notes/specs/06_generation_flags.md`: `--sym`/`--sym_mean`/`--sym_threshold`/
`--replicate_count`/`--screw_*` (priority 1), `--dssp`/`--dssp_mean`/
`--merge_chains`/seeding/`--sequence_only` (priority 2), `--template`/
`--template_aa` (priority 3), `--order_agnostic`/`--renoise`/`--sample_final`/
`--depth_adjust` (priority 4). WRITER role in the reader/writer pair; the
reader's spec is self-contained (no JAX source read).

## Plan

A previous WRITER session (same task) was cut off by a rate limit after
writing essentially all of the code (`salad_torch/inference/{symmetry,
template,postprocess,options}.py`, the CLI wiring in `scripts/generate.py`,
and `AADecoder.sample` in `salad_torch/modules/decoders.py`) but before
running any verification. This session: (1) read all 4 owned files + the full
1038-line spec, (2) code-reviewed every hook against the spec section by
section, (3) ran CPU-only unit checks (precedence chain, `screw_replicate` on
a toy tensor, `parse_template`/`read_pdb` on a synthetic PDB, `reorder_amino_
acids` on a synthetic backbone), (4) ran the real CLI end-to-end on the H100
checkpoint for every flag, (5) fixed 2 real bugs the GPU runs surfaced, (6)
re-ran to a clean pass.

## Changes

- `scripts/generate.py`: added light type hints (`StepFn`, `Tensor`,
  `argparse.Namespace` annotations) that the cut-off session's draft was
  missing. Fixed two real bugs found by the verification runs (see Evidence):
  (a) `--order_agnostic`'s `write_result_pdb` now catches `reorder_amino_
  acids`'s documented `RuntimeError` (backbone has isolated residues - a real,
  fairly common outcome with this checkpoint at L=32, not a fluke) and falls
  back to writing the network's own residue order instead of aborting the
  whole `--num_designs` run; (b) that fallback also repairs `residue_index`
  (`np.arange(L)`) - without it, every residue keeps the all-`-1` value
  `--order_agnostic` feeds the network, `to_pdb` writes duplicate PDB residue
  numbers in one chain, and a standard reader (including this port's own
  `read_pdb`) silently merges same-numbered residues on read-back, losing
  residues.
- No changes needed to `salad_torch/inference/{symmetry,template,postprocess,
  options}.py` or `salad_torch/modules/decoders.py::AADecoder.sample` - all
  correct as the cut-off session left them, confirmed by full CPU + GPU
  verification below.
- Skipped per spec (comments already in place, confirmed correct): `config.
  symm`/`update_positions(symm=...)` (dead code), `salad/inference/symmetry.
  py::Screw` (unused by the eval script), the `use_omap` trunk embedding
  (unreachable for every config this port supports).

## Evidence

All against the real checkpoint `/hnvme/workspace/v181bb10-salad/
default_vp-200k.npz`, `--num_steps 30` unless noted. Jobs: 901321 (first full
run, found the order_agnostic crash), 901334 (15-combination sweep: steps
30/60/100/150/300 x seeds 1/2/3, ALL 15 hit the same RuntimeError - ruled out
"just needs more steps"), 901344 (harness fix to survive one failed scenario),
901357 (post try/except fix - surfaced the residue_index bug via a shape-
mismatch/NaN in the diff check), 901360 (final, clean: **25 CLI runs, 0
unexpected outcomes, 69 checks, 0 failed**), 901372 (targeted confirmation
that the residue_index fix produces a well-formed 32-residue PDB when the
fallback path actually fires). Total GPU time across all 6 jobs: ~13 minutes
(~0.2 GPU-hours, 1x H100 each) - no approval threshold implications.

**C4 symmetry acceptance test** (superimpose chain i vs chain 0 rotated by
the *known* i*90 deg, not a fitted rotation), 3 independent generations,
25:25:25:25 residues:
| variant | overall RMSD | max per-chain residual |
|---|---|---|
| `--sym c4` (mean=False, default) | 0.041-0.070 A | 0.06-0.12 A |
| `--sym c4 --sym_mean True` | 0.068-0.078 A | 0.08-0.14 A |
| `--sym c4 --sym_threshold 0.5` | 0.035-0.054 A | 0.04-0.13 A |

Small and reproducible across 3 independent runs - the symmetriser is
correctly wired end to end (`sym_noise` on `pos_noised`, the post-hoc
`post_step_fn` on the x0 prediction, `init_pos`). The residual is NOT exactly
0 because it's measured on `atom_pos` (sidechain-reconstructed via
`AngleHead`, itself NOT re-symmetrised - only `data["pos"]` is, per spec
section 1.6); a toy-tensor unit test of `screw_replicate` alone gives ~1e-7
(pure floating point), confirming the small residual comes specifically from
one un-symmetrised `AngleHead` pass, not a broken symmetriser.

All other flags: ran end to end, finite output, and visibly changed the
result vs. the closest no-flag baseline (mean CA displacement, or sequence-
identity fraction for `--sample_final` since it doesn't touch the backbone -
see Outcome). `--template_aa`: 100% of conditioned positions carry the
template's amino acid exactly. `--depth_adjust 8` (> the checkpoint's native
depth 6) fails exactly as documented: `IndexError: index 6 is out of bounds
for axis 0 with size 6`.

## Outcome

All spec-listed flags implemented and verified working. Two real bugs found
and fixed (both `--order_agnostic`-specific, both in `scripts/generate.py`,
neither in the owned inference/ modules). One test-design mistake caught and
fixed in the verification harness itself (comparing `--sample_final` against
its baseline by CA position instead of sequence identity - `--sample_final`
only changes the decoded sequence/sidechains, not the backbone, since
`AngleHead`'s backbone atoms derive from `pos`, which `--sample_final`
doesn't touch).

Interesting, not-a-bug finding: `--order_agnostic` (zero sequence-order
information into the model) reliably produces backbones with 2-11 (of 32)
residues that never form a clean C-N peptide-bond-like link, **independent of
how many diffusion steps are run** (30 through 300 all reproduce it, 15/15 in
the sweep) - this looks like a genuine limitation of this checkpoint's
generalisation to the order-agnostic regime, not an under-converged-run
artifact. `reorder_amino_acids`'s hard iteration cap + `RuntimeError` (a
spec-mandated deviation from upstream's unbounded hang) correctly catches
this every time; the CLI-level fix here is to not let one bad design abort a
whole `--num_designs` batch.

Spec deviation, already commented at the call site (not something I
introduced): `--sample_final` doesn't build a literal second model with
`config.sample_aa=True` as the spec's §5.3 "Wire the CLI" bullet suggests -
`config.sample_aa` isn't threaded through this port's `Diffusion.forward` at
all (dead key), so doing that literally would have zero effect. The cut-off
session's actual implementation (ordinary forward pass, then explicitly call
`AADecoder.sample` + `AngleHead`) reproduces the intended behaviour instead of
the literal wiring suggestion.

## Outputs

- Code: `scripts/generate.py` (edited this session), `salad_torch/inference/
  {__init__,symmetry,template,postprocess,options}.py`, `salad_torch/modules/
  decoders.py::AADecoder.sample` (both pre-existing from the cut-off session).
- `.agent-research/results/N002-generation-flags/verify_flags.py` - the
  verification harness (subprocess-drives the real CLI, checks finiteness +
  "flag visibly changes the result", does the C4 residual computation).
- `.agent-research/results/N002-generation-flags/check_order_agnostic.py` -
  the 15-combination step/seed sweep that ruled out "just needs more steps".
- `.agent-research/results/N002-generation-flags/confirm_fallback.sbatch` -
  targeted re-confirmation that the fallback fix produces a well-formed PDB
  when it actually fires.
- `.agent-research/results/N002-generation-flags/logs/verify_901360.log` -
  the final clean run (25/25 CLI runs ok, 69/69 checks pass, C4 residuals).
- `.agent-research/results/N002-generation-flags/verify_flags_results.json` -
  structured results from the final run.

## Next

- `--order_agnostic`'s backbone-fragmentation behaviour is worth a closer look
  if this flag becomes load-bearing for a real project - is it specific to
  `default_vp-200k` (an intermediate checkpoint), or does it persist in later/
  fuller-trained checkpoints? Not investigated further here (out of scope for
  "port the flag faithfully").
- Not independently re-verified: `default_vp_scaled`/`default_ve_scaled`/
  `multimotif_vp` configs with these flags (only `default_vp` was checked,
  matching the checkpoint the task specified).

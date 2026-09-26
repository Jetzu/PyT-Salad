# N021 · Packed-batch inference: real batching + (batch x group x L) trade-off surface

type: task · date: 2026-09-26 · status: done

## Goal
Add REAL batching to inference - B designs concatenated along salad's flat
residue axis so each kernel does B designs' work in ONE launch - composable
with N019's `--group_size`, then characterise (batch x group x L) so a user can
max out a given GPU. N019's `--group_size` reclaims idle SMs but cannot make a
kernel bigger; N016 measured arithmetic at ~5% and DRAM at 9% of a step.

## Plan
- `--batch_size B` on `scripts/generate.py`, composing with `--group_size G`
  (one replay advances B*G designs). `--batch_size 1` = today, unchanged.
- Surface over B x G x L, L in {50,100,200,400,1000}.
- QUALITY at n=160/arm (N015 methodology) + aatype divergence, because packed
  designs are NOT bit-identical.
- Standard suite unchanged.

## Changes
- `salad_torch/diffusion/sampling.py`: `pack_designs` (concatenate, never pad;
  pair maps BLOCK-DIAGONAL; `chain_index` RENUMBERED) + `split_packed_snapshot`.
- `salad_torch/modules/fast.py`: `batch_size` on `FastDenoiser` (all buffers
  sized for the flat axis) and `GroupFastDenoiser`; `designs_per_replay`;
  `_autotune` divides by designs not members.
- `scripts/generate.py`: `--batch_size`, `run_grouped` rewritten around phases.
- `docs/USAGE.md`: a `--batch_size` section that leads with "NOT recommended".

Found here and independently by N022: N017's CSE closure is unpicklable, so
inductor's FX-graph cache was silently never written (295 s recompile per
process). N022 landed the fix; N021's duplicate was reverted.

## Evidence
- 904375 smoke; 904424 parity (SUITE_EXIT=0, 0.00002/0.00020 A, geom 38/38).
- 904403-904420: 55-cell (B x G x L) surface, zero failures.
- 904422: `--batch_size 1` byte-identical to `--group_size` (4/4, DETSEG=1).
- 904430 (cpu, 48 cores): packing does NOT leak - a design is bit-identical
  regardless of pack-mates at L=32..400; the only change is GEMM reassociation.
- 904436: full-rollout divergence, floor exactly 0.0, packed = different design.
- 904421/904447: n=160 quality arms, seeds 42 and 1234.
- 904586: baseline control on UNMODIFIED HEAD; 904595: seed-set confound
  control (flat, p >= 0.10); 904428/904581 roofline; 904623 fp32 probe.

## Outcome
**DO NOT SHIP `--batch_size` as a speed knob.** Faster (+42/+16/+4.5/+3.6/0 %
over `--group_size` at L=50/100/200/400/1000) but it changes the designs:
ALA 7.2 % -> 39.8 %, LEU 46.8 % -> 14.1 % (chi2 = 13984, p = 0), helix +0.06,
strand -0.025, Rg +2.4 A, every KS p < 1e-4, replicated on two seeds with the
seed-set confound controlled. Arithmetic utilisation does NOT move: grouping
already reached 10.9 % of fp32 peak, packing adds 0.3 points then regresses -
because matmul FLOPs scale as B while the neighbour sort / Gumbel field /
pair maps scale as B^2. Code kept as an experimental, warned, opt-in flag.

ALSO FOUND (not mine, not fixed): unmodified HEAD's legacy `--fast` path
diverges on 41/160 designs at L=100 against N015's recorded 0/160 at c5ab7a0.

## Outputs
- `.agent-research/results/N021-packed-batching/` — RESULTS.md, SEMANTICS.md,
  surface_table.txt, surface_cells.txt, per_design_arms_seed42.json, scripts,
  sbatch files, logs/.

NOT a TF32 artifact: the fp32 (`exact`) replicate, job 904623, reproduces the
shift and makes it larger (helix 0.904 -> 0.967, Rg 21.9 -> 25.5, diverged
42 -> 9, ALA 0.074 -> 0.415, every Welch p = 0.000).

## Next
- Quality gate at L=200/400, where the speed gain is only 4 %.
- Bisect the pre-existing HEAD divergence regression (c5ab7a0 -> fd9ba32).
- A block-diagonal-aware neighbour selection + Gumbel draw would remove the
  B^2 term; it changes shared model code and would break parity.

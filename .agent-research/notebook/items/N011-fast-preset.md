# N011 · Fast inference preset: maximum speed, numerics allowed to deviate

type: task · date: 2026-09-26 · status: open

## Goal
A `preset="fast"` twin of N009's `preset="exact"`: maximum speed for `--fast`
generation, ALLOWED to deviate numerically from the JAX original, but NOT
allowed to change what the model is. Quality gate on >=5 real designs at L=100:
CA-CA mean 3.70-3.85 A (exact path 3.766 +- 0.013), no NaN, Rg sane (11-16 A
typical), secondary structure not collapsed.

## Plan
Measure each candidate on its own, one length per process (`bench_one` rule),
against the current exact path as baseline; then combine the winners and run
the quality gate end to end. Candidates:
  A. fewer sampling steps (500/400 -> 250/200) - preset default, pure sampler
  B. approximate neighbour selection: topk instead of stable argsort
  C. K reduction beyond the provably-null one (current branch 64 -> 48 / 32)
  D. neighbour reuse across blocks (stride)
  E. need_aux=False (re-measure under the manual graph)
  F. reduced precision: TF32, and whole-graph fp16/bf16 with weights cast once
  G. sync-free conditioning cache key
Rank by speed per unit of deviation (A of drift from the exact path).

## Changes

## Evidence

## Outcome

## Outputs

## Next

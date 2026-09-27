# N026 · Kale feasibility: SALAD-family structure predictor

type: research · date: 2026-09-27 · status: done

## Goal
Scope the feasibility of "Kale": a SALAD-family protein **structure predictor** optimised
for throughput (single-second range), with standardised confidence metrics and multi-chain
co-folding for binder/interface design, ligands, optionally RNPs. User constraints added
during scoping: **~11 M parameters**, **no diffusion module**, **backprop capability**.

## Plan
Four agents, read-only. (1) map `SALAD_torch` first-hand — param counts by instantiation,
timings from the repo's own `.agent-research/`. (2) survey fast/small MSA-free predictors
with live retrieval. (3) survey confidence heads on non-AF2-shaped trunks. (4) survey
standalone EMA / distilled confidence. No code, no jobs.

## Changes
None. Scoping only.

## Evidence
Full report: `results/N026-kale-feasibility-salad-family-structure/RESULTS.md`.

Load-bearing numbers:
- ESMFold fig. S3 (*Science* 379:1123, doi 10.1126/science.ade2574), 8-block trunk, CAMEO
  LDDT: removing **the pLM = -0.16**; removing **triangular updates = -0.01 to -0.04**.
  Triangles are cheap to drop, the pLM is not.
- **No published model reaches TM > 0.7 without an MSA or a >= 0.67 B pLM** (searched for a
  counterexample; none found).
- SALAD measured by instantiation: **11,538,916 params** total (`default_vp`), denoiser
  9,984,957, trunk 6,940,200, encoder 1,553,959 (never called at generation).
- SALAD has **no pair tensor and no triangle op anywhere**; pair is `(N,K,64)`, K<=64.
- Measured H100: **3.56 ms/forward at L=100, 7.38 ms at L=400**; 5.6 % of fp32 peak,
  >12x above its own kernel-launch floor (launch-bound, not compute-bound).
- Data already on disk: `pdb_monomer_train` 1,090,530 rows, `pdb_complex_train` 263,571,
  val 15,777 / 2,279; `pdb_structure.lmdb` ~100 GB, `afdb_structure.lmdb` 56 GB.
- Protenix-Mini ESM ablation (arXiv:2507.11839): dropping MSA costs **-2.6 pts
  intra-protein LDDT but -8.2 pts protein-protein**. The MSA tax is an *interface* tax.
- SimpleFold pLDDT bolt-on (arXiv:2509.18480, verified in `inference.py:245-265`): freeze
  folder, re-run on own output at t=1, detach, 4 transformer blocks -> 50 bins, **r = 0.77**.
- **PAE/pTM/ipTM read a pair representation in every implementation.** SimpleFold,
  OmegaFold, EquiFold emit none. AF3/Boltz-1 keep a PairFormer inside the confidence module.

## Outcome
**Feasible, with one forced design change and one scope cut.**
- Speed is a non-issue: no diffusion means 4 forwards instead of 100-500, so ~0.1 s at
  L=300 — **~10x margin** over the target. Accuracy, not throughput, is the risk.
- The 11 M budget holds for the **trainable trunk**; the deployed footprint includes a
  frozen pLM. EMBER3D (4.7 M trainable + ProtT5 1.5 B, sub-0.3 s) is the existence proof.
- **Forced change: ipTM requires a pair rep** that a SALAD-shaped trunk lacks. Build it
  only inside the confidence head (~4-8 TriMul-only blocks at c_z=64, **~5 M params**),
  trained as a separate frozen-trunk stage. Revised total **~16 M trainable**.
- **Scope cut: ligands out of v1, RNPs out of v1 and v2.** Smallest credible open model
  doing complexes+ligands is Protenix-Mini at ~135 M.
- Architecture: frozen ESM-2 650M -> sparse IPA trunk (K<=64) -> **recycling x3 instead of
  diffusion** (backprop through last cycle only, the ColabDesign pattern). Initial k-NN
  graph seeded from ESM-2's attention contact map — load-bearing here, unlike ESMFold
  where 48 triangle blocks rebuild the pair rep anyway.
- Justification is **differentiability, not speed**: MMseqs2-GPU + OpenFold-TRT already
  does AF2-accuracy in 15.9 s, and CASP16 found no advantage for pLM-using groups. What is
  defensible is that SimpleFold-Turbo's caching speedup is unusable inside a gradient loop,
  so a single-pass differentiable folder is the right tool for design loops.

Top risk: **nobody has shown a pair-tensor-free K<=64 representation can PREDICT rather
than GENERATE** at this size. Searched specifically for sparse/k-NN triangle alternatives —
no working public demonstration exists. Opportunity and risk in one sentence.

## Outputs
- `results/N026-kale-feasibility-salad-family-structure/RESULTS.md` — full report with
  citations, phasing, and an explicit unverified list.

## Next
- **P0 kill gate, ~20 GPU-h (under the 25 GPU-h approval gate, runnable without asking):**
  freeze ESM-2 650M, precompute embeddings for ~50 k chains, train trunk + frames head with
  recycling on monomers <=256 aa. Does the contact-map-seeded sparse graph reach TM > 0.5?
  Must report a **dose-response curve**, not an endpoint (the N-series undertraining trap).
  Kill if a random-init-pLM control is not clearly below it.
- P1 (~300-600 GPU-h) and P2 (~1000+ GPU-h) **need user approval** before submit.
- Decide the `-1` sentinel fix up front if padded batching is needed — it regenerates every
  fixture and breaks parity.

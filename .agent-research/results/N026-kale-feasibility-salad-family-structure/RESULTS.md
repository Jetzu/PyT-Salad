# N026 — Kale: feasibility scope for a SALAD-family structure predictor

Date: 2026-09-26/27. Scoping only — no code written, no jobs submitted.

Question: build "Kale", a SALAD-family protein **structure predictor** optimised for
throughput (single-second range), with standardised confidence metrics, multi-chain
co-folding (binder/interface design), ligands, optionally RNPs. Constraints added by
the user during scoping: **~11 M parameters**, **no diffusion module**, **backprop
capability**.

Evidence base: 4 agents. One mapped `SALAD_torch` first-hand (param counts measured by
instantiation, timings from the repo's own `.agent-research/`). Three surveyed the
literature with live retrieval. Flags from those agents are carried through below.

---

## VERDICT

**Feasible, with one forced design change and one scope cut.**

- Speed is a non-issue: ~10x margin over the target.
- The 11 M budget holds for the *trainable trunk*, not for the deployed footprint.
- Forced change: **ipTM requires a pair representation** that a SALAD-shaped trunk
  does not have. It must be built inside the confidence head.
- Scope cut: **ligands out of v1, RNPs out of v1 and v2.**

---

## 1. The two numbers that set the shape

ESMFold's own ablation (fig. S3, Lin et al., *Science* 379:1123, doi 10.1126/science.ade2574),
8-block trunk, CAMEO LDDT:

| removed | dLDDT |
|---|---|
| **the language model** | **-0.16** |
| triangular updates | -0.01 to -0.04 |
| recycling | -0.01 to -0.04 |
| AF2 distillation targets | -0.01 to -0.04 |

Triangle attention is nearly free to drop. Confirmed independently by three groups:
Pairmixer (arXiv:2510.18870, ICLR 2026) matches Pairformer 0.78 = 0.78 lDDT at -36 %
GPU-days, -51 % inference FLOPs; Pearl (arXiv:2510.24670) ships TriMul-only at SOTA
PoseBusters 84.7 %; SeedFold-Linear (arXiv:2512.24354) is flat-or-better on monomers and
PPI. OpenFold component-skipping (arXiv:2511.14781) reaches the same ordering by a
different method.

**No published model reaches TM > 0.7 without either an MSA or a >= 0.67 B pLM.** The
literature agent searched specifically for a counterexample and found none.

So Kale = **~11 M trainable trunk + a frozen pLM**. That is what the small-model field
already is:

| model | trainable | frozen pLM | CASP14 / CAMEO TM | L~300 |
|---|---|---|---|---|
| **EMBER3D** (bioRxiv 2022.11.14.516473) | **4.7 M** | ProtT5 1.5 B | ~0.68 / - | **0.15-0.3 s**, 8 GB consumer GPU |
| SimpleFold-100M (arXiv:2509.18480) | 94 M | ESM2-3B | 0.611 / 0.803 | 3.9 s (500 steps) -> ~0.43 s w/ Turbo caching |
| HelixFold-Single | ~89 M | own 1.09 B | ~AF2+MSA on CAMEO | - |
| ESMFold | 690 M | ESM2-3B | 0.68 / 0.83 | ~2.5 s |

EMBER3D is the existence proof: 4.7 M trainable, sub-0.3 s. The 11 M budget is generous.

Caveat worth carrying: CASP16 found **no significant advantage for pLM-using groups**
(Wang et al., *Quantitative Biology* doi 10.1002/qub2.70035), and MMseqs2-GPU +
OpenFold-TRT now does AF2-accuracy end-to-end in **15.9 s** (bioRxiv 2026.03.11.711233).
MSA-free is justified here by **orphan/designed sequences and differentiability**, not by
speed any more. State that honestly in any writeup.

---

## 2. Architecture

```
seq -> frozen ESM-2 650M (int8) -+- per-residue embeddings -> input MLP
                                 +- attention contact map --> seeds initial k-NN graph
                                          |
                        +-----------------+--------------+
                        | SALAD trunk: sparse IPA blocks | x12
                        | (N,K,64), K<=64, no pair tensor|
                        +-----------------+--------------+
                              recycle x3 --+
                                           |
        +---------------+-----------+------+------+------------------+
   frames/atom14    pLDDT head          PAE/pTM/ipTM head        distogram
                    (4 tf blocks)       (small TriMul-only pair module)
```

**Diffusion -> recycling.** Iterative refinement is still needed; take AF2's mechanism
instead of a sampler. Deterministic, and backprop through the last cycle only — the
ColabDesign/AFDesign pattern. Local reference: `Protein_Design_Models/af2_mpnn_backprop`.
This single substitution makes all three user constraints (no diffusion, backprop-able,
small) compatible at once.

**Bootstrap fix.** SALAD builds its neighbour graph from coordinates; folding from
sequence there are none at cycle 0. Seed it from ESM-2's attention contact map — free,
since the pLM forward is already paid for. ESMFold does exactly this
(`use_esm_attn_map` -> `esm_z_mlp`, `esmfold.py:203-208`) and notes the boost
"disappears during training" — for them, because 48 triangle blocks rebuild the pair rep
anyway. Kale has no such fallback, so for Kale this is **load-bearing, not optional, and
it is the main unproven piece of the design.**

**Deterministic k-NN, not Gumbel.** SALAD redraws random long-range neighbours every
block; identical seeds give different proteins (48 % of residues differ). Prediction
wants determinism and design gradients need it. Drop the Gumbel sampling, keep the
16-sequence + 16-spatial + top-k-contact structure.

---

## 3. Throughput — not the risk

Measured on disk, H100, `default_vp-200k` (`SALAD_torch/notes/04_PARITY_AND_PERF.md` §7.2):
one 6-block denoiser forward is **3.56 ms at L=100, 7.38 ms at L=400**. SALAD needs
100-500 of those because it is a diffusion model. **Kale needs 4** (recycles).

Budget at L=300: pLM ~10-20 ms + trunk (12 blocks x 4 recycles) ~60 ms + heads ~5 ms
= **~0.1 s**. Ten times inside the single-second target; still ~5x inside it with ESM-2 3B.

Headroom beyond that: the port runs at **5.6 % of fp32 peak**, 946 kernels per step,
**>12x above its own measured kernel-launch floor** (N016/N021). It is launch-bound, not
compute-bound. Fusion is the lever; reduced precision is a measured dead end (bf16/fp16
autocast are ~3x *slower* — more casts = more kernels).

**Conclusion: stop optimising speed, spend the budget on accuracy.**

---

## 4. Confidence — where the design has to bend

### pLDDT: architecture-agnostic, confirmed
AF2, OpenFold, OmegaFold, ABodyBuilder3 and SimpleFold all implement it as
`MLP(per-token single rep) -> 50 bins -> softmax . bin centres`, CE loss at weight ~0.01.
**None touches a pair rep.** ESMFold ships `esmfold_structure_module_only_*` checkpoints
with **0 folding blocks** — pair rep is literally the zero tensor — and pLDDT still works.

SimpleFold is the bolt-on recipe for a pair-rep-free trunk, verified in code
(`src/simplefold/inference.py:245-265`): freeze the folder, re-run it on its own finished
coordinates at t=1, `.detach()` both hops, 4 transformer blocks + MLP -> 50 bins, CE
against the sample's true LDDT. Result: **Pearson r = 0.77** vs LDDT-Ca.

### PAE / pTM / ipTM: NOT free — this is the forced change
Every implementation is `Linear(c_z -> 64 bins, max_bin 31 A)` **on a pair
representation**, because PAE is inherently pairwise. **Nobody in the pair-rep-free camp
emits them**: SimpleFold none, OmegaFold none, EquiFold none. AF3 and Boltz-1 keep a full
PairFormer *inside the confidence module* even though the main path is a diffusion trunk
(Boltz-1: 48 layers, initialised from trunk weights, trained as a separate stage).

**ipTM is the binder-design metric** — it is what BindCraft filters on — so it cannot be
skipped. Fix: build a pair rep **only in the confidence head** — outer product/difference
of token embeddings plus pairwise coordinate features, then ~4-8 **TriMul-only**
(Pairmixer-style) blocks at c_z=64 -> 64-bin PAE; pTM and ipTM are closed-form from there.
Budget **~3-5 M params**, O(L^2) memory confined to one module, trained as a separate
frozen-trunk stage as everyone else does.

**Revised total: ~11 M trunk + ~5 M confidence ~= 16 M trainable.** Still ~40x smaller
than SimpleFold-100M's trunk.

### Why build it in rather than score post-hoc
CASP16 (Fadini, Studer, Read, *Proteins* 94(1):302-313, doi 10.1002/prot.70037): on
**hetero-oligomers the AF2 ipTM/pTM baselines OUTPERFORMED all 23 dedicated EMA
predictors**. Every top local-accuracy group was passing through AF3's own per-atom
pLDDT; the assessors recommend CASP17 exclude pass-through entries. No published
single-model EMA method beats pLDDT from structure alone at the per-target level —
EnQA and GCPNet-EMA both *consume* AF2 pLDDT as an input, i.e. they are refiners.
In-model confidence is not replaceable by an external scorer.

Two calibration notes: calibration is a function of **training maturity, not
architecture** (OpenFold: overconfident -> underconfident -> converged). And **ranking
ability and correlation decouple** — EnQA on CASP16 H1204 scored ranking loss 0.000 with
Pearson -0.01. Pick the metric before picking the method.

Sequence-only confidence (no folding) tops out at **r ~ 0.76-0.77** — AFDistill
(arXiv:2210.03488, ProtBert-420M, 0.028 s @1024 aa, differentiable). Note the often-cited
pLDDT-Predictor (arXiv:2410.21283, "250,000x speedup") was **WITHDRAWN by its authors for
overfitting** — do not cite it.

---

## 5. Cheap vs expensive

**Cheap (already paid for):**
- **The data.** `pdb_monomer_train` 1,090,530 rows; `pdb_complex_train` 263,571 (with
  `n_prot`/`n_nuc`); cluster labels; held-out val (15,777 / 2,279). `pdb_structure.lmdb`
  ~100 GB, `afdb_structure.lmdb` 56 GB, both preprocessed. Normally 60 % of such a project.
- **Multi-chain plumbing.** SALAD threads `chain_index` through 85 sites: cross-chain
  relpos masking, per-chain `index_mean` gating, a cross-chain contact flag, cyclic
  chains, symmetry. Inherited free.
- **~7.9 M of trained trunk weight** reusable as initialisation.
- **pLDDT** (above). **Triangle ops** — never built, so never removed.
- Teachers on disk for distillation/eval: Boltz-2 ckpt, Chai-1 v2, Protenix/AF3,
  `alphafast.sif`, CCD mirror.

**Expensive:**
1. **Interfaces without an MSA.** Protenix-Mini's ESM ablation (arXiv:2507.11839):
   dropping MSA for ESM2-3B costs **-2.6 pts intra-protein LDDT but -8.2 pts
   protein-protein and -5.4 protein-ligand**. The MSA tax is an *interface* tax — and
   interfaces are the stated target. Biggest threat to usefulness.
2. **Ligands.** SALAD has zero HETATM/nucleic handling; non-standard residues -> `UNK`
   (`data/dataset.py:56`). From-scratch AF3-style atom encoder/decoder. Smallest credible
   open model doing complexes+ligands is **Protenix-Mini at ~135 M**. An 11 M
   ligand-capable folder is not a thing.

---

## 6. Risks, ranked

1. **Can a pair-tensor-free K<=64 representation PREDICT rather than GENERATE?**
   Generation accepts any plausible fold; prediction must hit the right one. SimpleFold
   proves the pair rep is droppable — but pays with 2.86 B params and 500 sampling steps.
   Nobody has shown it at 11 M with 4 recycles. The literature agent searched specifically
   for sparse/k-NN or low-rank triangle alternatives and found **no working public
   demonstration** — genuinely unexplored. Opportunity and risk in one sentence.
2. **Interface accuracy without MSAs** (-8.2 pts, above).
3. **Undertraining will lie to you.** Pairmixer states its triangle-ablation gap
   (0.74 -> 0.70) "is largely recovered through longer training". This is exactly the
   `triad_v2_e2_undertraining` trap. Measure the dose-response, never a single step count.
4. **SALAD's `-1` sentinel bug.** The no-neighbour sentinel un-masks residue N-1, which is
   then selected twice and double-counted; the same protein at a different flat-axis
   offset gets a different answer (up to ~5 A). Faithful to upstream JAX, still wrong.
   If Kale needs padded batching this must be fixed **up front** — it regenerates every
   fixture and breaks parity.
5. **CUDA non-reproducibility.** `index_mean` -> `scatter_add` atomics at ~1.6e-7, amplified
   discretely by the stable argsort over near-ties. Deterministic k-NN (§2) removes most of
   this; verify rather than assume.

---

## 7. Phasing and compute

| phase | scope | estimate | gate |
|---|---|---|---|
| **P0** | Freeze ESM-2 650M, precompute embeddings for ~50 k chains, train trunk + frames head with recycling on monomers <=256 aa. Does the contact-map-seeded sparse graph reach TM > 0.5 at all? | **~20 GPU-h** | **Under the 25 GPU-h approval gate — runnable now.** KILL GATE: if a random-init-pLM control is not clearly below it, stop. |
| **P1** | Full PDB + AFDB, 12 blocks, 4 recycles, pLDDT head. Target EMBER3D-class (TM ~0.68) at ~0.1 s | ~300-600 GPU-h | **needs approval** |
| **P2** | `pdb_complex_train`, interface-biased cropping, TriMul-only confidence module -> PAE/pTM/ipTM. Gate on **DockQ**, not LDDT | ~1000+ GPU-h | **needs approval** |
| **P3** | Ligands (AF3-style atom encoder/decoder). Expect the param budget to leave the 11 M class | not scoped | — |

P0 must report a **dose-response curve**, not an endpoint (risk 3).

---

## 8. Recommendation

Build it, but as a **differentiable design oracle first and a general folder second.**

The speed argument alone does not justify the project — MMseqs2-GPU + OpenFold-TRT does
AF2-accuracy in 15.9 s, and SimpleFold-Turbo is already ~0.4 s. What *is* defensible:
BindCraft-style design does thousands of folding calls inside a gradient loop, and
SimpleFold-Turbo's 9-14x caching trick is **inherently unusable there** (cached =
skipped forward passes = no gradient). A genuinely single-pass, deterministic,
differentiable folder is the right tool for design loops, and nothing published fills
that slot at this size. DCFold (arXiv:2605.17899) explicitly names gradient-based binder
hallucination as the payoff of collapsing AF3 to one step.

Start with P0. It is under the GPU gate, it tests the one genuinely unknown thing
(sparse-graph prediction bootstrapped from pLM contacts), and it is cheap to abandon.

---

## Unverified / carried flags

- The literature agent's threads on multi-chain/ligands and distillation **did not
  complete**; those sections are partial coverage only.
- "Kaede" and an "APEX" folder: **not found anywhere** — treat as nonexistent.
- AF3 Supplementary Information not fetchable (Nature auth redirect); AF3 head bin counts
  and layer shapes **not verified at source**.
- Boltz-1/Boltz-2/Chai-1 parameter counts: **not published**.
- Protenix-Mini's 135 M: sourced from the DCFold paper, not from Protenix-Mini itself.
- EMBER3D and OmegaFold CASP14 TM: read off figures, +-0.03.
- ESM-2 intermediate scaling rungs (35M/150M/650M/3B): **not directly retrieved**; one
  fetch returned a table contradicting the *Science* main text and was discarded. Do not
  quote intermediate-rung numbers without checking the SI.
- The ~0.1 s L=300 budget in §3 is **arithmetic from measured SALAD block timings plus an
  estimated pLM cost**, not an end-to-end measurement. Measure it in P0.

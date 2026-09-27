# N025 part II — the MECHANISM of the `--fast` rollout regression

Branch `n006-manual-cudagraph`. Worktrees `/hnvme/workspace/v181bb10-salad/n025`
(fd9ba32) and `n025_good` (c5ab7a0). Scripts in `scripts2/`. All generated PDBs
were written to the node's `$TMPDIR`, metered in-job and deleted (`/hnvme` is in
inode grace).

## VERDICT

**`603051e`'s P7 batched condition-branch prologue (`SALAD_OPT_BATCHCOND`) is the
entire regression.** `SALAD_OPT_BATCHCOND=0` at HEAD restores c5ab7a0's numbers
exactly, at n=160. The CSE pass (`fd9ba32`), `aggressive_fusion`, the
`--group_size` refactor (`b9dd416`) and the CUDA-graph capture are all
exonerated by direct veto.

**P7 is not WRONG — it is numerically exact to fp32 rounding.** Its pair
features match the six sequential `pair_2` calls to <=8.6e-6 absolute
(<=6.2e-7 relative) and its pair MASKS are bit-identical, at a real
mid-rollout state with the real distinct per-block neighbour index sets, on GPU
and on CPU, eager and compiled. So this is a legitimate reassociation
(`bmm(batch=6)` instead of six `mm`s) that the 500-step feedback rollout cannot
absorb, not a coding error in the prologue.

## The localisation, arm by arm

`--fast --preset fast`, L=100, seed 42, `default_vp-200k`.

| # | arm | n | diverged | conv CA-CA | conv Rg | job |
|---|---|---|---|---|---|---|
| 1 | c5ab7a0 | 64 | **0** | 3.7618 | 13.06 | 906583 |
| 2 | **603051e (P7 lands)** | 64 | **11 (17.2 %)** | **3.8738** | **20.75** | 906583 |
| 3 | b9dd416 (`--group_size`) | 64 | 11 (17.2 %) | 3.8791 | 20.80 | 906583 |
| 4 | HEAD, `SALAD_OPT_AGGRFUSE=0` | 160 | 33 (20.6 %) | 3.8702 | 20.75 | 906549 |
| 5 | HEAD, **`SALAD_OPT_BATCHCOND=0`** | 160 | **0** | **3.7632** | **12.94** | 906549 |
| 6 | HEAD, both off | 160 | **0** | 3.7610 | 12.93 | 906549 |
| 7 | HEAD, CSE vetoed | 32 | 5 (15.6 %) | 3.8811 | 20.14 | 904716 |
| 8 | HEAD, compiled, no CUDA-graph capture | 16 | 1 | 3.8633 | 20.17 | 904722 |
| 9 | HEAD, `--preset exact` (fp32) | 32 | 7 (21.9 %) | 3.8802 | 20.66 | 904716 |
| 10 | HEAD, **EAGER** | 32 | **0** | **3.7608** | **12.95** | 904716 |
| 11 | c5ab7a0, EAGER | 32 | 0 | 3.7601 | 12.77 | 904716 |

Arms 2 and 5 together are the whole finding: the regression appears at the P7
commit and is switched off by P7's own env veto, back to the reference numbers
(3.7605 / 13.07) at n=160.

## Why P7 and not the compile

Arm 10 says HEAD EAGER is clean, which at first reads as "it is the compile".
It is not, and the reason matters: **`opt.BATCH_COND` defaults to True from the
environment and is only ever WRITTEN by `FastDenoiser.__init__`**, so the eager
path runs P7 as well — but through `diffusion.py`'s *eager* branch, which feeds
the prologue `torch.stack([cond_neighbours_of(prev_pos) for _ in range(depth)])`,
i.e. six IDENTICAL index sets (its own comment says so). The fast path's
`forward_core` feeds six DISTINCT ones. So "eager clean / compiled broken" is
confounded and must not be read as exonerating the model maths on its own; what
exonerates P7's *maths* is the unit test below, and what localises the defect is
arm 5.

## The unit test: P7's maths is exact

`x_p7.py --unit` / `x_p7u.py`, at a dumped step-30-of-250 state
(`prev_pos |max| = 24.48`, nonzero, `K=48`, per-block condition-neighbour sets
verified NOT identical), `DET_SEGMENT=1`, fp32. Weight cache `in_sync=True`.

| comparison | GPU max abs | CPU max abs | relative |
|---|---|---|---|
| prologue vs six sequential `pair_2`, eager | 8.6e-6 | 0.0 – 5.7e-6 | <=6.2e-7 |
| **sequential compiled vs sequential eager (CONTROL)** | **8.6e-6** | 6.7e-6 | <=3.9e-7 |
| prologue compiled vs prologue eager | 8.6e-6 | 6.7e-6 | <=3.9e-7 |
| prologue compiled vs sequential compiled | 6.7e-6 | 0.0 | <=3.0e-7 |
| pair MASK equality, all 6 blocks | bit-equal | bit-equal | — |

The control line is the point: P7's deviation is the SAME SIZE as the ordinary
compile drift on the shape it replaced, which c5ab7a0 has too and which does not
break anything. So the magnitude is unremarkable; what is different is that P7
perturbs the condition branch of every one of the six blocks on every step, and
the perturbation then has 500 steps and a discrete neighbour argsort to grow in.
(The two scripts disagree by ~6e-6 on the CPU eager comparison — itself evidence
the quantity sits on the rounding floor.)

## Where the rollout actually breaks (per-step, job 904722)

Per-step means over CONVERGED designs only, 16 designs, 250 steps:

| step | raw_t | c5ab7a0 CA-CA / Rg | HEAD CA-CA / Rg |
|---|---|---|---|
| 0 | 1.00 | 1.615 / 8.49 | 1.928 / 7.22 |
| 16 | 0.94 | 1.994 / 11.06 | 1.874 / 10.25 |
| **32** | 0.87 | 2.911 / **12.19** | 2.299 / **15.15** |
| 48 | 0.81 | 3.333 / 12.24 | 2.658 / 15.80 |
| **64** | 0.74 | 2.938 / **12.48** | 2.285 / **22.24** |
| 80 | 0.68 | 3.190 / 12.91 | 2.542 / **24.36** |
| 128 | 0.49 | 3.589 / 12.88 | 3.237 / 21.63 |
| 200 | 0.20 | 3.762 / 12.69 | 3.863 / 20.17 |

GRADUAL, not a threshold crossing: the cloud inflates between steps ~16 and ~80
(raw_t 0.94 -> 0.68), peaks at Rg 24, then contracts only to 20 and never
recovers. `max|pos|` 36-44 A vs 24-38 A at c5ab7a0. That window is the high-t
topology-commitment phase and straddles the self-conditioning cutoff
(`raw_t < 0.8` = step 51 here).

**The 620-688 A monsters are a SECOND, later, SUDDEN event** on top of the
inflated baseline: the diverged design in each broken arm first crosses CA-CA
5 A at step 175/177 (125/127 in the AGGRFUSE=0 arm) with a single-step jump of
x2.85 / x3.37. Gradual inflation = the distribution shift in every design;
late escape = the diverged count.

## Two hypotheses REFUTED

**Randomness reuse / CSE collapsing independent draws — REFUTED three ways.**
(a) The 3216-step records hash the actual `uniform` (Gumbel) and diffusion-noise
tensors used at every step: **0 duplicate uniform hashes, 0 duplicate noise
hashes**, at HEAD and at c5ab7a0. (b) Structurally there is no RNG op inside the
compiled region at all — CUDA graphs cannot capture one, so
`_prepare_inputs` pre-draws the whole `(n_draws, L, L)` field outside and
`_copy_in` writes it into a static buffer, and `apply_diffusion` is outside too;
`fx_graph_cse` has nothing random to merge. (c) Vetoing the pass end to end
leaves the regression fully intact (arm 7; and job 904722's L6, Rg 16.9-28.7).

**Self-conditioning feedback broken by `b9dd416` — REFUTED by instrumentation.**
`prev` hashed in and out at every step: fed-back 800, zero-reset 2416,
**MISMATCH 0**, first zero-reset at step 0 — *identical in all three arms*
(c5ab7a0, HEAD, HEAD+AGGRFUSE=0). 16 designs x 50 live steps = 800 and
16 x 151 zeroed = 2416, and 50+151 = 201 = the whole run, so the
`prev_threshold` gate fires on exactly the same step set at both commits. No
write-after-capture hazard is visible. The `b9dd416` diff on the single-design
path is also verbatim-preserving on inspection (`make_design_data` lifted
unchanged, `forward` split into `_prepare_inputs`/`_run`/`_postprocess` with the
bodies in the same order; `pin_condition` is only reached from `run_grouped`).

## THE MECHANISM: inductor MIS-COMPILES the P7 prologue inside `forward_core`

Job **906623**, arms A1-A4 (`scripts2/x_p7.py`). ONE call of `forward_core` on the
dumped step-30 state, bit-identical inputs, `DET_SEGMENT=1`, fp32, `pair_cond=False`,
`need_aux=need_aa=True`:

| comparison | `pos` max abs | `local` | `raw_angles` | `aatype` | `distogram_neighbours` |
|---|---|---|---|---|---|
| EAGER, P7 on vs off | **9.60e-06** | 6.1e-04 | 5.8e-04 | 0 | 0 |
| **COMPILED, P7 on vs off** | **9.586 A** | **646** | **439** | **12** | 0 |
| P7 **off**, compiled vs eager | **2.00e-05** | 6.1e-04 | 1.8e-03 | 0 | 0 |
| **P7 on**, compiled vs eager | **9.586 A** | **646** | **439** | **12** | 0 |

Read rows 3 and 4 together. With P7 OFF the compiled forward agrees with eager to
**2.0e-05 A** — that is the compile floor on this very forward. With P7 ON the
compiled forward is **9.586 A** away from eager, i.e. **5 orders of magnitude
above that floor**, on a SINGLE step, and `local` (the 128-d feature the sampler
feeds back as `prev_local`) is off by 646. Twelve residues get a different
`aatype` label. This is not reassociation and not chaos: **the compiled P7 path
computes a materially different function.**

Put together with the unit test above:

* P7's maths, eager, is right (<=8.6e-6, masks bit-equal).
* `StackedCondPair.forward` compiled **in isolation** is also right
  (`pro_comp` vs `pro_eager` = 8.6e-6, job 906627).
* P7 compiled **inside the whole `forward_core` graph** is wrong by 9.586 A.

So the defect is in how inductor compiles the prologue *in the context of the
full graph*, not in `cond_prologue.py`'s arithmetic. And it is invisible to
every existing test because **all four parity tests drive the EAGER path**, where
P7 is correct - and where `diffusion.py`'s eager branch additionally feeds the
prologue six IDENTICAL neighbour index sets.

### Why the whole picture is consistent

A 9.6 A error on a single mid-trajectory forward, fed back 500 times, is a
complete explanation of the observed end-to-end signature: the Rg 12.9 -> 20-24
inflation starting at step ~32, the CA-CA 3.760 -> 3.87 shift in the survivors,
and the 16-21 % that escape entirely. No amplification argument is needed.

### Per-block localisation (job 906640, `scripts2/x_p7b.py`)

`trajectory` is `torch.stack([pos after each block])`, so per-block max abs vs the
same state, `DET_SEGMENT=1`, fp32, plain `torch.compile(mode="default")`:

| comparison | blk0 | blk1 | blk2 | blk3 | blk4 | blk5 |
|---|---|---|---|---|---|---|
| EAGER P7on vs P7off | 7.6e-06 | 2.2e-05 | 2.1e-05 | 9.5e-06 | 9.1e-06 | 9.6e-06 |
| **P7off compiled vs eager (FLOOR)** | 2.8e-05 | 3.1e-05 | 4.4e-05 | 4.1e-05 | 1.5e-05 | 2.0e-05 |
| **COMPILED P7on vs P7off** | **0.368** | 4.16 | 5.95 | 5.51 | 8.58 | **9.59** |
| **P7on compiled vs eager** | **0.368** | 4.16 | 5.95 | 5.51 | 8.58 | **9.59** |

The error is already **0.368 A at the output of block 0** and compounds monotonically
through the stack. So the wrong value enters at block 0's `attn_2` - the attention
that consumes the prologue's `pair`/`pair_mask` - on the first block, not by
accumulation over blocks.

### The MINIMAL reproducer

Neither CUDA-graph capture, nor the CSE post-pass, nor `aggressive_fusion`, nor a
TF32 preset is required. Job 906640's arms ran with
`aggressive_fusion=False, pattern_matcher=True, inline_inbuilt_nn_modules=True`
(the torch 2.5.1 defaults) and `joint_custom_post_pass` unset:

    opt.BATCH_COND = True                 # P7
    opt.DET_SEGMENT = True                # remove atomics jitter
    torch.set_float32_matmul_precision("highest")
    fn = torch.compile(model.predict.diffusion.forward_core,
                       mode="default", dynamic=False)
    fn(*args)      # 9.586 A from the eager call on the same args

### Determinism does not rescue it (job 906623, arms Q)

`--fast --preset fast`, HEAD, `SALAD_OPT_DETSEG=1`, n=32:

| arm | diverged | conv CA-CA | conv Rg |
|---|---|---|---|
| `SALAD_OPT_BATCHCOND=1` | **6/32 (18.75 %)** | 3.8788 | 20.16 +- 2.59 |
| `SALAD_OPT_BATCHCOND=0` | **0/32** | 3.7643 | 13.12 +- 2.27 |

So the failure survives full determinism: it is not the `index_sum` atomics
jitter being amplified, it is a deterministic wrong answer.

### Paired rollout: wrong at STEP 0, no amplification phase (job 906633)

`scripts2/x_pair.py`, 3 designs, 250 compiled steps, `DET_SEGMENT=1`, fp32, the
same seed-derived generator per design in both arms.

| arm pair | first \|dCA\| > 1.0 A | final \|dCA\| |
|---|---|---|
| P7 on vs P7 off | **step 0** (9.74 A) | 40.5 / 41.4 / 62.3 A |
| **P7 off vs P7 off replicate (FLOOR)** | **never** | **0.000 A, all 250 steps, 3/3** |

The floor row is the control that makes this airtight: under `DET_SEGMENT=1` the
compiled path is BIT-IDENTICAL to itself across a whole 250-step rollout, so all
40-62 A is P7 and none of it is nondeterminism.

Design 0's first steps, with the fraction of DISCRETE `distogram_neighbours`
indices that disagree between the two legs:

| step | max\|dCA\| | discrete neighbour disagreement |
|---|---|---|
| 0 | 9.74 A | 0.0 % (inputs identical) |
| 1 | 5.20 A | **92.7 %** |
| 5 | 6.35 A | 89.0 % |
| 20 | 15.3 A | 91.9 % |
| 24 | 24.7 A | 93.5 % |
| 104 | 39.5 A | 95.7 % |
| 249 | 40.5 A | 95.8 % |

One wrong forward at step 0 decorrelates 90+ % of the neighbour graph by step 1,
after which the two runs are simply different proteins - and the P7 one is
systematically expanded (Rg 20-28 vs 13) with 16-21 % destroyed outright.
Note step 0 has `prev_pos = 0` (the sampler's `init_prev`), so the miscompile
does not even need a realistic self-conditioning state to fire.

### Which inductor knob owns it: NONE of the ones tried

All arms COMPILED, P7 ON, compared against EAGER P7 ON. Every one returns the
**bit-identical** wrong answer (`pos = 9.586 A`, per-block
`[0.368 4.16 5.95 5.51 8.58 9.59]`):

| probe | result |
|---|---|
| baseline (`mode="default"`) | 9.586 A |
| `aggressive_fusion = False` | 9.586 A |
| `pattern_matcher = False` | 9.586 A |
| `epilogue_fusion = False` | 9.586 A |
| `inline_inbuilt_nn_modules = True` | 9.586 A |

That the numbers are bit-identical across four independent scheduler/rewrite
knobs argues the defect is NOT in inductor's codegen or fusion but in the
**program Dynamo/AOTAutograd traces** for `forward_core` when
`batched_cond_pair_available()` is True - i.e. something about the prologue is
captured wrongly, identically, regardless of how it is then scheduled. Note
`StackedCondPair.forward` compiled ON ITS OWN is correct to 8.6e-6 (job 906627),
so it is the prologue IN THE CONTEXT of the full `forward_core` graph.

Root cause below that level is NOT established. What is established is the
switch: `SALAD_OPT_BATCHCOND=0` (or dropping P7 from the presets) restores
c5ab7a0's numbers exactly at n=160.

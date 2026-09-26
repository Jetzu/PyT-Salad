# Using `salad-torch`

Every flag of `scripts/generate.py`, what it does, and what it costs.

This is a PyTorch port of [salad](https://github.com/mjendrusch/salad). It loads
the released JAX checkpoints unchanged and reproduces the original model's
output to **0.00002 Å**. Where upstream has a quirk, the quirk is reproduced —
those are called out below, because several of them reach the output PDB.

---

## 1. Fastest path to a protein

```bash
python scripts/generate.py \
    --config default_vp --params params/default_vp-200k.jax \
    --num_aa 100 --num_designs 8 \
    --fast --preset fast --group_size auto \
    --out_path outputs/my_designs
```

That writes `outputs/my_designs/design_0.pdb` … `design_7.pdb`.

Drop `--fast` and you get the plain eager path: same answer, ~20× slower.
Use `--preset exact` instead of `fast` when you need parity with JAX salad.

---

## 2. The three flags that decide your runtime

These are the only performance flags. Everything else is a modelling choice.

### `--fast`

Switches to the compiled + CUDA-graph inference path. **GPU only, one fixed
length per process, incompatible with `--sym`.** Costs a one-time compile of
roughly a minute; after that every step is ~20× cheaper than eager. If you are
generating more than two or three designs, always use it.

### `--preset {exact,fast,deterministic}`

Only meaningful with `--fast`.

| preset | numerics | L=100 step | when to use |
|---|---|---|---|
| `exact` *(default)* | parity with JAX salad, 0.00002 Å | 3.18 ms | you are comparing against upstream, or you need the reference answer |
| `fast` | TF32 + half the sampling schedule | 3.01 ms | normal production use |
| `deterministic` | `exact` + reproducible reductions | 3.63 ms | you need the same command to give the same protein twice |

`fast` is `exact` plus two changes: TF32 matmuls, and `steps_scale=0.5` (250
sampling steps instead of 500). Both were validated at **160 designs per arm**
against `exact` — bond geometry, radius of gyration, secondary-structure
composition, clash count and divergence rate are all statistically
indistinguishable (Welch *p* ≥ 0.24, two-sample KS *p* ≥ 0.13). The per-*step*
saving is only 1.23×; most of `fast`'s wall-clock win is the halved schedule.

`deterministic` replaces `scatter_add` (whose atomic accumulation order varies
run to run) with a one-hot matmul. That makes `--seed` actually pin a CUDA run.
**It is not parity-neutral**: it drifts ~3.2× further from JAX than `exact`
(0.199 Å vs 0.062 Å) and flips 7 of 64 residues in the sampling-parity fixture.
Use it to debug or to reproduce a run, not as your production preset.

### `--group_size {auto,N}`

Advances **N independent designs through one CUDA graph** instead of one at a
time. Needs `--fast`.

This is **not a batch axis.** Each design keeps its own buffers, its own noise
and its own seed-derived RNG stream, and computes exactly what it would have
computed alone. The only thing that changes is that the GPU stops idling
between one design's serialised kernel launches. Verified: under
`--preset deterministic`, grouped and ungrouped runs are **bit-identical**,
8/8 designs, max |Δpos| exactly 0.0.

| | L=100 | L=400 |
|---|---|---|
| `fast`, one at a time | 0.603 s/design | 1.36 s |
| `fast`, `--group_size auto` | **0.334 s** (N=8) | 1.18 s (N=4) |
| speedup | **1.81×** | 1.16× |

`auto` **measures** the right N on your actual device: it captures at 1, 2, 3,
4, … and stops when a rung stops paying or free VRAM says stop. A capture is
seconds, not a recompile, so this is cheap. Prefer it over pinning an integer —
a value that won a sweep on one card is wrong on another.

The win comes from **idle SMs**, so it is largest on a big GPU with a short
protein and shrinks toward 1× on a small GPU or a long protein. Memory is
~3.8 MiB per residue per design, so on a 24 GB card VRAM never binds at a group
size worth using — speed picks N first.

`--group_size 1` is the grouped machinery with a group of one. It reproduces the
legacy per-design loop to within 3% and exists as the reference a grouped run
must match.

### `--batch_size N` — measured, and NOT recommended (N021)

**Real batching**: `N` designs concatenated along salad's flat residue axis, so
each kernel does `N` designs' work in one launch. Composes with `--group_size`,
so one replay advances `batch_size × group_size` designs. `--batch_size 1` is
today's path, byte for byte.

It is **faster**, on top of `--group_size`'s own ceiling:

| L | best group-only | best packed | packing's gain |
|---|---|---|---|
| 50 | `G=16` 975 designs/s | `B=8 G=4` 1385/s | **+42 %** |
| 100 | `G=16` 613/s | `B=8 G=2` 713/s | **+16 %** |
| 200 | `G=8` 341/s | `B=4 G=2` 356/s | +4.5 % |
| 400 | `G=4` 171/s | `B=2 G=2` 177/s | +3.6 % |
| 1000 | `G=2` 70/s | packing **loses** (B=4 is 0.84×) | 0 % |

The knee is a flat-axis length, not a batch size: every winning cell has
`batch_size × L` between 400 and 800 residues. That is the recipe if you want
one — `B ≈ 600 / L`, rounded to a power of two, then let `--group_size auto`
pick the rest. There is no `--batch_size auto`, because each `B` is a new
compiled shape (minutes), not a re-capture (seconds).

**But it changes the designs**, and that is why it is not recommended. Unlike
`--group_size`, packing is not semantically free. At n=160 per arm, replicated
on two seeds, with the seed-set confound controlled, `--batch_size 8` against
`--batch_size 1` at L=100:

* alanine 7.2 % → **39.8 %** of residues, leucine 46.8 % → 14.1 %
  (χ² = 13 984, 19 dof, p = 0) — the sequences drift toward poly-alanine;
* helix +0.06, strand −0.025, coil −0.037, Rg +2.4 Å, CA-CA +0.045 Å, all
  surviving Bonferroni, every two-sample KS p < 1e-4;
* diverged designs drop (26/160 → 8/160), which is the one change in the
  user's favour.

It also does **not** raise arithmetic utilisation: grouping already takes the
step from 5.6 % to 10.9 % of fp32 peak, and packing adds 0.3 points before
going backwards past `B=2`. The win is launch slots, the same currency
`--group_size` spends, because the matmul FLOPs scale exactly linearly with `B`
while the neighbour sort, the Gumbel field and the pair maps scale as `B²`.

Use it only if you have measured that the distribution shift does not matter
for what you are doing. Full numbers:
`.agent-research/results/N021-packed-batching/RESULTS.md`.

---

## 3. What to generate

### `--num_aa` — length, chains, and cyclic peptides

An integer, or colon-separated chain sizes. A `c` prefix makes that chain
cyclic.

```
--num_aa 100          one 100-residue chain
--num_aa 100:200      two chains, 100 and 200 residues
--num_aa 100:200:c8   plus an 8-residue cyclic chain
```

> **Upstream quirk, reproduced.** Residue numbering starts at **0**, and each
> new chain starts 50 past the *previous residue's* index — so `100:100` gives
> `0..99` then `149..248`, not `150..249`. Both the 0-based start and the
> off-by-one reach the output PDB.

`--num_aa` is **overridden** by a literal `--dssp` string and by `--template`.

### `--num_designs`

How many designs to produce. With `--group_size N` the last group is short
rather than padded, so `--num_designs 10 --group_size 4` runs 4 + 4 + 2.

### `--config` and `--params`

`--config` selects the architecture; `--params` points at the matching
checkpoint. They must agree.

| config | checkpoint |
|---|---|
| `default_vp` *(default)* | `default_vp-200k.jax` |
| `default_vp_scaled` | `default_vp_scaled-200k.jax` |
| `default_ve_scaled` | `default_ve_scaled-200k.jax` |
| `multimotif_vp` | `multimotif_vp-200k.jax` |

Checkpoints come from Zenodo record
[14711580](https://doi.org/10.5281/zenodo.14711580). `params/` holds twelve
released variants; the four above are the ones with verified parity. They are
pickles of JAX arrays and load without JAX installed.

### `--out_path`

Output directory (created if absent). Default `outputs/designs`. Files are
`design_{i}.pdb`, except under `--renoise`, which switches to
`result_{design}_{idr}_{step}.pdb`.

### `--sequence_only`

Write FASTA (appended to `{out_path}/sequences.fa`) instead of PDB.

---

## 4. Conditioning: steering the output

### `--dssp` — secondary structure

`none` *(default)*, `random`, or a literal per-residue string over `L`/`H`/`E`/`_`
(loop / helix / strand / unconstrained):

```
--dssp HHHHHHHHHH____EEEEEEEE____HHHHHHHHHH
```

> A literal string's **length overrides `--num_aa`** and discards any
> `--merge_chains` or cyclic spec.

### `--dssp_mean` — target composition instead of per-residue

`none`, `random`, or `loop,helix,sheet`. **Not renormalised — it must already
sum to 1.**

```
--dssp_mean 0.3,0.5,0.2
```

### `--template` — build on an existing structure

```
--template 5TPN.pdb,FFFFFF____FFFFFF
```

`<pdb_path>,<constring>`: one character per residue plus `:` chain separators.
`F` = fixed/conditioned on the template, anything else = free. **Overrides
`--num_aa` and `--dssp` entirely and disables cyclic conditioning.**

`--template_aa` additionally conditions on the template's *sequence* identity,
not just its geometry.

### `--merge_chains`

Zeroes `chain_index` before the model sees it. No-op if `--dssp <string>` or
`--template` is given, since both reassign chain afterwards.

### `--order_agnostic`

The network never sees residue order (`resi=-1`, `chain=0`). The output PDB is
re-threaded by following C–N backbone contacts instead of network indexing.

### `--cloud_std`

Overrides the initial point-cloud standard deviation. Empty uses the config's
own value.

---

## 5. Symmetry

Cyclic and screw (helical) symmetry. **Incompatible with `--fast`**, so
symmetric runs use the eager path and are slow.

```bash
--sym c4 --num_aa 60:60:60:60
```

| flag | meaning |
|---|---|
| `--sym` | `''` disables. `cN` (e.g. `c4`) is a convenience that sets `--replicate_count=N`, `--screw_angle=360/N`, `--screw_translation=0` as overridable defaults. Requires `--num_aa` as N colon-separated, ideally equal-length chains. |
| `--replicate_count` | number of subunits; defaults from `cN` |
| `--screw_angle` | degrees about the Y axis; defaults to `360/N` |
| `--screw_radius` | offset of the representative subunit from the axis before replication |
| `--screw_translation` | rise per subunit along Y. `0` = plain cyclic (Cn) point group; nonzero = true helical screw |
| `--sym_mean` | literal `'True'`/`'False'`. Controls the `init_pos` and per-step post-hoc symmetrisers. **The noised-input symmetriser always uses `False` regardless** — an upstream inconsistency, reproduced. |
| `--sym_threshold` | post-hoc symmetrisation only runs while `raw_t >` this. `0.0` (default) symmetrises after every step; raising it (e.g. `0.5`) relaxes the constraint for late low-noise refinement. |

---

## 6. The sampling schedule

Defaults match upstream. Change them only deliberately.

| flag | default | meaning |
|---|---|---|
| `--num_steps` | 500 | denoising steps |
| `--out_steps` | 400 | comma-separated steps at which to write output. **Must be ascending.** |
| `--prev_threshold` | 0.8 | self-conditioning threshold |
| `--timescale_pos` | `cosine(t)` | position noise schedule. Also `linear`, `none`, `framediff`, `ve(...)`, `edm(...)`. |

Note that `--out_steps 400` with `--num_steps 500` means the default run stops
writing at step 400, not 500 — upstream's choice, reproduced.

`--preset fast` multiplies `--num_steps` and `--out_steps` by 0.5. If you set
them explicitly and also use `--preset fast`, the scaling still applies.

### `--renoise`

At every out-step, additionally runs **9 independent partial re-denoise passes**
re-entering at `raw_t=0.4`, with self-conditioning forced off. **Roughly a 3×
wall-clock multiplier per design.** Switches PDB naming to
`result_{design}_{idr}_{step}.pdb`.

### `--sample_final` and `--sample_final_temperature`

On out-steps only, replaces the single-shot sequence decode with an `L`-step
categorical random-order autoregressive re-decode, then rebuilds sidechains from
it. Costs `L` extra decoder-only forward passes per out-step.
`--sample_final_temperature` defaults to `0.1`, matching upstream's hardcoded
value; the flag itself is a port-side addition.

### `--depth_adjust`

`none`, or an integer `N` ≤ the checkpoint's native depth (6 for `default_vp`):
an early-exit ablation that builds and loads only the first `N` trunk blocks.
`N` greater than the native depth is unsupported — there are no extra trained
weights to load.

---

## 7. Reproducibility

### `--seed`

Seeds the noise draw. **On CUDA this does not by itself pin the output.**
Segment reductions use `scatter_add`, whose atomic accumulation order varies
between runs; the neighbour `argsort` downstream turns that into a discrete tie
flip, and 500 sampling steps amplify it. Measured self-divergence for an
identical command run twice: **48% of residues differ.**

To actually pin a CUDA run, add `--preset deterministic` (or set
`SALAD_OPT_DETSEG=1`). On CPU everything is already deterministic.

### `--jax_seed`

An alias for `--seed`, for command lines copied from the JAX CLI. **It does not
give bit-identical output across backends** — the two frameworks' PRNGs are
unrelated. It only gives reproducibility within one backend.

### `--device`, `--compile`, `--timing_json`

`--device` defaults to `cuda` when available. `--compile` passes a `torch.compile`
mode for the eager path (independent of `--fast`, which manages its own
compilation). `--timing_json` writes a timing breakdown to the given path.

---

## 8. Compile time, and making it happen only once

`--fast` compiles and captures the denoiser before the first step. Measured cold
at L=100: **~310 s**. That is a one-time cost, not a per-design one — but it is
paid again for every new *(torch version, GPU architecture, length)* combination,
and by default it is paid again on **every machine and every job**.

That is because inductor caches to `TORCHINDUCTOR_CACHE_DIR`, which defaults to
`/tmp/torchinductor_$USER`. On most clusters — and in most containers — `/tmp` is
node-local and discarded between jobs, so the cache never survives to be reused.

**Point it at storage that persists and is visible from wherever you run:**

```bash
export TORCHINDUCTOR_CACHE_DIR="$YOUR_PERSISTENT_SCRATCH/torchinductor"
```

Use whatever your site calls that — `$WORK`, `$SCRATCH`, `$XDG_CACHE_HOME`, a
project directory, a mounted volume. Do not use a path that is node-local (often
`/tmp` and sometimes `/scratch`), or you get the default behaviour back.

Budget **~3,800 files and ~280 MB per compiled length**. On an inode-quota'd
filesystem that adds up quickly across lengths, so pick a filesystem with inode
headroom rather than the fastest one.

With a persistent cache dir, the first run on a given length pays ~310 s and
every later run — in any process, on any node with the same GPU architecture —
starts warm.

Two things that make the cold compile longer, worth knowing:

* Matmul templates (`SALAD_OPT_GEMMTPL=1`) roughly double it, ~310 s → ~740 s.
* Compilation already uses 32 worker threads by default; there is nothing to
  gain by raising it.

> **Note for torch < 2.6.** The `torch.compiler.save_cache_artifacts()` /
> `load_cache_artifacts()` bundle API ("Mega-Cache") is not available, so a
> prebuilt cache cannot be shipped alongside the code. On a newer torch it can.

## 9. Environment flags

Individual optimisations, for debugging. All default **on** except `DETSEG`, and
`--preset` sets them for you — reach for these only to bisect a problem.

| variable | default | effect |
|---|---|---|
| `SALAD_OPT_BATCHCOND` | on | batch the condition-branch GEMMs across the 6 trunk blocks |
| `SALAD_OPT_AGGRFUSE` | on | aggressive inductor fusion |
| `SALAD_OPT_LEAN` | on | skip unused auxiliary outputs during sampling |
| `SALAD_OPT_NOPC` | on | skip redundant pair conditioning |
| `SALAD_OPT_AUX` | on | keep auxiliary outputs available |
| `SALAD_OPT_GATHER` | on | gather instead of one-hot matmul for embeddings |
| `SALAD_OPT_HOIST` | on | hoist loop-invariant neighbour computation |
| `SALAD_OPT_DETSEG` | **off** | deterministic segment reduction (see `--preset deterministic`) |

Setting any to `0`/`false`/`no`/`off` disables it.

---

## 10. Hardware notes

Developed and measured on an H100. Nothing in the code is architecture-specific
— no Hopper intrinsics, no inline PTX, no hand-written CUDA, no hardcoded block
sizes. TF32 needs Ampere or newer; CUDA graphs and `torch.compile` are generic,
and the generated Triton kernels are autotuned per device at compile time.

Two honest caveats:

- **All performance numbers above are H100 numbers.** The model is
  launch-bound by roughly 10× on any device, so the *shape* of the problem
  carries over, but the `--group_size` win in particular scales with spare SMs:
  extrapolating by SM count gives ~1.85× on a 4090 and ~1.0× below ~46 SMs.
  That extrapolation has **no measurement behind it** — but it does not need
  trusting, because `--group_size auto` measures the truth on your card.
- **Blackwell (sm_120) is untested.** It is not a code problem: it needs
  CUDA 12.8+ and PyTorch ≥ 2.7, and the development environment here is
  torch 2.5.1 + cu121, which cannot target it at all.

Both presets still run **>12× above the measured theoretical floor** for this
model (0.258 ms/step for `exact`, 0.239 ms for `fast`, against 3.18 / 3.01 ms
actual). The binding constraint is kernel launch serialisation, not arithmetic
or bandwidth — L2 runs at 21% of its rate and DRAM reads are 9% of the step.
There is real headroom left; see `notes/04_PARITY_AND_PERF.md`.

---

## 11. Things that will bite you

**Padding is not safe, and batch position changes the answer.** A `-1` "no
neighbour" sentinel un-masks the last residue, which is then selected twice and
double-counted in attention. The same protein at a different offset in the flat
residue axis comes out up to ~5 Å different. This is deterministic, not sampling
noise, and it is bit-identical in JAX — the port is faithful and upstream is
wrong. **Concatenate examples along the residue axis; never pad.**

**Sampling trajectories are chaotic.** A single forward matches JAX to 2e-7, but
feeding the output back in for 500 steps amplifies fp32 reduction-order
differences. Compare trajectories as a *ratio* to an identical-code rerun, never
against an absolute tolerance.

**`--fast` fixes the length.** One compiled graph per shape. Generating several
lengths in one process degrades the CUDA graphs badly — run one length per
process.

**`--fast` is incompatible with `--sym`**, and `--group_size` additionally
refuses `--renoise` and `--sequence_only`.

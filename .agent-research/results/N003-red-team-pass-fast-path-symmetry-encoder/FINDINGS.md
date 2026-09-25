# N003 red team — SALAD_torch, 2026-09-25

Adversarial pass over `modules/fast.py` (condition-cache + static buffers —
uncommitted when this pass started, **landed as `75f4fb2` while it ran**), `inference/*` + `scripts/generate.py`, `modules/encoder.py`, and
`data/` + `training/`. Source was never edited; every item below has a
reproducer in this directory.

**Baselines re-run first:** geometry 38/38 PASS. model parity 0.00002 A on
`default_vp` AND `default_ve_scaled` — PASS. `tests/test_stack_parity.py`
does **not** pass; it crashes (see D2).

**Note:** `salad_torch/modules/fast.py` and `salad_torch/training/loop.py` were
edited by another agent
(N001/N004) mid-session; `fast.py` landed as `75f4fb2`, so target 4 is only
partly covered. All `fast.py` results below were measured against the code that
is now `75f4fb2` (verified identical: `_cond_cache` at lines 71/149/165/168).

`r05_encoder_ref.npz` is gitignored (`*.npz`); regenerate it with
`r05_encoder_parity_jax.py`.

---

## SILENTLY WRONG (worst first)

### S1. `FastDenoiser` caches `prepare_condition` with no key — all conditioning after the first call is ignored
`modules/fast.py:160-168`. `self._cond_cache` is set on the first `forward`
and never invalidated by anything the caller does. Reproducer
`r03_fast_attack.py` §B, log `r03_901501.log`, one `FastDenoiser`, identical
pinned `uniform`, reference = uncompiled `forward_core`:

| conditioning added on call 2 | \|fast − correct\| | \|fast − stale\| | correct vs stale |
|---|---|---|---|
| `dssp_condition` | **5.524 A** | 4.3e-4 A | 5.524 A |
| `chain_index`    | **2.529 A** | 6.95 A   | 7.163 A |
| `dmap`+`dmap_mask` (template) | **15.343 A** | 5.3e-4 A | 15.343 A |
| `dssp_mean`      | **4.934 A** | 6.4e-4 A | 4.934 A |

And two *designs* in one process (§B5): different `dssp_condition` per design
gives outputs that differ by **1.7e-4 A** when the correct answer is
**11.4 A** — i.e. every design after the first silently reuses design 0's
conditioning. This is exactly the `--dssp random` / `--dssp_mean random` /
`--num_designs N` case. No error, no warning.

`chain_index` is only *partly* stale because it reaches the trunk twice: as
the cached `flags`/`condition` and as the live `chain` tensor argument. That
makes the wrongness harder to spot, not smaller.

Now committed as `75f4fb2` ("N001: cache loop-invariant conditioning").
Mitigating: `scripts/generate.py` does not use `FastDenoiser` at all (it
`torch.compile`s the blocks instead), so nothing user-facing is affected
today. Only `scripts/bench_fast.py` / `diag_fast.py` construct one, and they
never change conditioning. The trap is armed for whoever wires it into the CLI.

Suggested fix: key the cache on the identity/version of the conditioning
tensors actually read by `prepare_condition` (`aa_gt`, `chain_index`,
`batch_index`, `mask`, `aa_condition`, `dssp_condition`, `dssp_mean`,
`dmap_mask`, `dmap`, `omap`, `chain_contacts`), or drop the cache.

### S2. `FastDenoiser` mutates the caller's `data` dict in place
`modules/fast.py:130-140`. When a `_CORE_ARGS` key has no matching static
buffer, the code does `self._static_in[k] = src`, adopting the *caller's*
tensor as the persistent buffer; the next call then does `dst.copy_(src)`
into it. `_blank_inputs()` sets `cyclic_mask=None`, so this fires on
`cyclic_mask` for every cyclic config, at the construction length.

Reproducer `r03_fast_attack.py` §C2: caller A passes an all-False
`cyclic_mask`, caller B an all-True one; after B's call, **A's own tensor
reads 64/64 True**. Generalises to every key whenever `L != self.length`
(the first such call donates all its tensors).

### S3. Clash-repulsion gradient clamps the wrong quantity — ~100% error vs JAX
`inference/symmetry.py:115-118`.

    JAX  : dist = sqrt( sum_c max(dx_c,      1e-3)**2 )   # clamps the DIFFERENCE
    port : dist = sqrt( sum_c max(dx_c**2,   1e-3)    )   # clamps the SQUARE

These are not the same function. JAX's (buggy but reproducible) form zeroes
every *negative* component to 1e-3, so a pair at (−5,0,0) reads as distance
1.7e-3, i.e. maximally clashing; the port reads 5.0. Measured
(`r01_clash_grad_{torch,jax}.py`):

| case | rel err vs JAX | corr | per-step push (time=1) |
|---|---|---|---|
| spread blob (24 CA, σ=6 A) | 0.736 | 0.769 | — |
| clashing blob (40 CA, σ=3 A) | **1.006** | 0.946 | JAX 0.0078 A vs port 0.0142 A |

End-to-end in `screw_replicate` (`r02_screw_*.py`, case `c4 L=100 cc=T
t=.9`): **5.97e-3 A per call**, applied on every step with `raw_t > 0.5`,
inside the sampling feedback loop. Every other `screw_replicate` path matches
JAX to 1.5e-5 A.

### S4. `--cloud_std` is a silent no-op on the default config
`modules/model.py:46`:

    cloud_std = c.sigma_data if c.diffusion_kind == "vpfixed" else data.get("cloud_std")

Upstream `StructureDiffusionInference` / `StructureDiffusionNoise`
(`noise_schedule_benchmark.py:573-577`, `:615-619`):

    cloud_std = None
    if c.diffusion_kind in ("vpfixed", "flow"): cloud_std = c.sigma_data
    if "cloud_std" in data:                     cloud_std = data["cloud_std"]   # data WINS

Two divergences: (a) under `vpfixed` (= `default_vp`, the CLI default) the
port ignores `data["cloud_std"]` entirely, so `--cloud_std` does nothing —
verified: `--cloud_std` unset / `0.0` / `99.0` all produce a byte-identical
PDB (`md5 480b387d8f27`). (b) `diffusion_kind == "flow"` should get
`sigma_data` and does not (latent — no released config uses it). Also latent:
the port passes `flow=kind in ("flow","flow_scaled")` where the upstream
*inference* path uses `flow = kind == "flow"`.

### S5. `FastDenoiser` ignores `predict_aa=False` when building `atom_pos`
`modules/fast.py:183-184` overrides `result["aatype"]` *after*
`forward_core` has already run `angle_head` on the **predicted** aatype. The
returned pair is inconsistent: `aatype == aa_gt` but `atom_pos` is the
sidechain build for a different sequence. Measured (§E2):
`|fast − angle_head(aa_gt)| = 7.727 A`, `|fast − angle_head(predicted)| =
8.3e-5 A`. The eager path takes the other branch.

### S6. `FastDenoiser` ignores `aa_condition`
`forward_core` has no `aa_condition` override, unlike `Diffusion.forward`
(`diffusion.py:275`). §F1: 12/64 residues return an aatype that disagrees
with the eager path under `--template_aa`-style conditioning.

### S7. `--dssp random` / `--dssp_mean random` ignore `--seed`
`inference/options.py` uses the unseeded stdlib `random`; `generate.py:372`
seeds only a `torch.Generator`. Two runs of
`--num_aa 48 --dssp random --seed 42` gave **max|dCA| 15.5 A** (mean 4.2 A);
`--dssp_mean random --seed 7` gave 25.0 A. The plain path with the same seed
agrees to 1e-3 A (CPU reduction noise). `--jax_seed`'s help text explicitly
promises reproducibility within one backend.

### S8. The Gumbel neighbour draw never sees `--seed`
`Diffusion.forward` calls `extract_neighbours(...)` with `uniform=None` and no
`generator`, so the draw uses the process-global torch RNG (seeded
nondeterministically — `torch.initial_seed()` differs per process). `--seed`
only reaches `apply_diffusion`. Measured at L=48: 60 steps, same seed →
CA RMSD 0.004–0.019 A (different seeds → 7.8 A). Small, but it is the one
knob the repo advertises as making runs reproducible, and the repo's own
chaos analysis (notes §2) says perturbations at this scale grow over a
trajectory.

### S9. `--sym` with L not divisible by `replicate_count` leaves an orphan tail
`inference/symmetry.py:148-188` passes the trailing `L − size*count`
residues through unchanged while re-centering the symmetrised part on its own
CA mean, so the tail sits in a different frame. Upstream instead returns
`count*size` residues (a downstream shape error). `r07_remainder_drift.py`:
the tail centroid sits 10–17 A off the symmetric core and stays there.
End-to-end, `--num_aa 100 --sym c12` (size 8, remainder 4) writes a PDB with
an **11.66 A CA–CA break** at the seam and the 4 tail residues 9.99 A from
the core; `--num_aa 101 --sym c4` (remainder 1) survives at 3.43 A. Silent
either way — the CLI does not warn that `count` must divide L.

### S10. `--sym c1` / `--replicate_count 1` is a no-op in the port, not in JAX
`symmetry.py:149`: `if size == 0 or count <= 1: return x`. JAX still centres
on the CA mean and still applies the clash-repulsion push. `r02` grid:
1.36 A (plain), 2.42 A (`chain_center=True`), 2.27 A (`radius=15`).

### S11. Compiled path is neither eager-equal nor bit-reproducible
At `torch.set_float32_matmul_precision("highest")`, identical pinned
`uniform`, reference = uncompiled `forward_core`:
- §A1 compiled vs eager: **5.9e-4 A** (brief expected ~1e-4 A);
- §H1/H2: the *same* `FastDenoiser` called twice on identical inputs, with a
  second instance interleaved, differs by 5.3e-5 / **2.5e-4 A** —
  the compiled path is not deterministic across calls.
For scale, notes §2 measures a 1e-6 input perturbation growing to 0.58 A on
the worst trajectory step. This is 500x that perturbation, per step.

---

## CRASHES

### D1. `modules/encoder.py` cannot run at all — `Rigid.__getitem__` was removed
`encoder.py:302` and `:372` still use `frames[:, None]`. Commit `dda09aa`
("all call sites updated") removed `Rigid.__getitem__` in favour of
`unsqueeze_batch`, but `encoder.py` (added earlier, in `00bbcfd`) was missed.
**Every** call raises `TypeError: 'Rigid' object is not subscriptable`
(`r04_encoder_stress.py`, 13/13 inputs). It was invisible because the encoder
has no test and `load_encoder(strict=True)` only checks the state dict.
Fix: `frames.unsqueeze_batch(1)` in both places.

This is now load-bearing: the concurrent N004 edit to `training/loop.py:205`
makes `prepare_batch` call `encoder(enc_data)` whenever `augment_size > 0`,
i.e. the training loop hits it on the first batch.

**Good news** — with the one-line shim applied, the encoder is numerically
exact. I built the parity test that did not exist
(`r05_encoder_parity_{jax,torch}.py`, real `default_vp-200k` weights,
upstream `NSB.Encoder` under `structure_diffusion/~encode/encoder`):

    L16c1 / L64c1 / L96c3:  local rel <= 1.9e-06, pos rel <= 1.8e-07,
                            coords max 0.00001 A, backbone[0:5] exact

Degenerate inputs are all finite: L=1/2/3/5/47/48/49, collinear N-CA-C,
all-residues-identical, all-zero, 1e6-scale, all-masked, duplicate chains.
Gradients finite (104 params, 23 legitimately `None`, 0 NaN, 0 all-zero);
collinear backward finite. Only `L=0` crashes (reshape of 0 elements).
Recommend committing `r05_*` as `tests/test_encoder_parity.py`.

### D2. `tests/test_stack_parity.py` is broken — the claimed baseline does not run
`blocks.py:309` calls `neighbour_fn(pos, prev_pos, block_idx)` (3 args) since
`dda09aa`; the test's closure at `test_stack_parity.py:93` takes 2.
`TypeError`. `notes/04_PARITY_AND_PERF.md` still lists it as passing at
0.0003 A. Fix: `def neighbour_fn(_pos, _prev_pos, _block_idx=0)`.

### D3. `--depth_adjust N > native depth` dies with an opaque `IndexError`
`haiku_to_torch.py:191 arr = arr[layer]` → `IndexError: index 6 is out of
bounds for axis 0 with size 6`. The flag's own help text says N > native depth
is unsupported, so this only needs an argparse-level check.

---

## LOWER SEVERITY / NOTES

- **N1.** `FastDenoiser.forward(compute_distogram=True)` is silently accepted
  and dropped — `forward_core` never computes one, so `result` has no
  `"distogram"` key (§I1). Should raise.
- **N2.** CUDA graphs only *partially* engage: inductor logs `skipping
  cudagraphs due to cpu device (arg5_1)` for the subgraph containing
  `sidechain.py:351 _tables(...)` (module-level CPU tensors lifted into the
  traced region). Two `cache_size_limit` recompile storms on
  `diffusion.py:29 _slice` as well. The static-buffer change in `fast.py`
  is paying complexity for a partly disabled optimisation.
- **N3.** `FastDenoiser` with `L != self.length` raises a `TorchRuntimeError`
  from the default `uniform` draw (good), but runs *successfully* if the
  caller supplies a correctly sized `uniform` — after donating all its
  tensors to `_static_in` (see S2).

---

## ATTACKS THAT FOUND NOTHING

- `inference/template.py`: matches upstream `parse_template` line for line;
  end-to-end motif recovery 0.099 A CA RMSD, dmap MAE 0.058 A, corr 1.000.
  `--template` + `--template_aa` + `--sym` + multi-chain constrings all run;
  a constring/PDB length mismatch raises a clear `ValueError`.
- `screw_replicate` for every case where `count` divides `L` and
  `chain_center=False`: matches JAX to 1.5e-5 A, including `c12` with
  `radius=15, translation=15` and `translation=1e6`.
- C4 symmetry on the written PDB is genuinely C4: subunit-to-subunit Kabsch
  RMSD 0.024–0.031 A, rotation 89.99 / 179.99 / 90.02 deg about ±Y, residual
  translation 0.02 A (`r06_sym_residual.py` + `r06b_sym_axis.py`). Better than
  the claimed 0.04–0.07 A. (My first pass reported a 16 A "exact-Cn residual";
  that was a sign convention error in my own test, not a defect.)
- `--sym_threshold` at 10.0 / −5.0, `--replicate_count 0`, `count > L`,
  `screw_radius/translation = 1e6`, `--sym_mean True`, `--order_agnostic`
  (+`--sym`), `--sequence_only`, `--renoise`, `--sample_final`,
  `--merge_chains`, unsorted `--out_steps`, literal `--dssp` + `--sym`: all
  run without crashing.
- Held `FastDenoiser` outputs are NOT overwritten by the next call (§D1) —
  the `clone()` + `cudagraph_mark_step_begin()` handling is correct.
- At the construction length, `FastDenoiser` does not corrupt the caller's
  non-`None` inputs (§C1).
- `data/collate.py`: `concat_collate`/`split_chunks` do what the docstring
  says (concatenate, never stack; per-chunk local `batch_index` preserved).

---

## How to reproduce

CPU (login node), `PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python`:

    $PY r01_clash_grad_torch.py && apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD --bind /tmp:/tmp ../../../salad.sif r01_clash_grad_jax.py
    $PY r02_screw_torch.py      && apptainer run ... r02_screw_jax.py
    $PY r04_encoder_stress.py                      # D1; contains a test-local Rigid shim
    apptainer run ... r05_encoder_parity_jax.py && $PY r05_encoder_parity_torch.py
    $PY r07_remainder_drift.py

GPU (1 job, ~25 min, 0.4 GPU-h):

    sbatch r03.sbatch            # -> r03_<jobid>.log ; S1, S2, S5, S6, S11, N1-N3

Recorded runs: 901501 (full, 00:24 elapsed), 901487 (first attempt, harness bug).
Total GPU cost of this pass: ~0.9 GPU-h.

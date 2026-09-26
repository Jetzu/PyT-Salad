# N010 red-team: PyTorch SALAD port

**Verdict:** The committed `n006-manual-cudagraph` fast path has two high-severity silent failures: a sum-collision conditioning cache (2.56 Å wrong on a pinned H100 input) and a captured graph that accepts a new length but returns the old length. The verification suite also has serious execution gaps: `pytest tests` collects zero tests, missing fixtures make four parity scripts exit successfully, and both loss checkers exit successfully after printing `FAIL`.

Scope: no source files were edited by this audit. All scripts/logs/results are here. The committed base was `31937680e2a6b477d2d4bc94dee5959d79b2f0ec`. The first GPU attempt (`repro_fast_slurm.log`) passed a 2D fixture uniform to a 3D fast API, so its parity/determinism values are **invalid and excluded**; `repro_fast_recheck.log` uses a correctly shaped, repeated 3D field. **Concurrent edits appeared in the live worktree during this audit**, including `fast.py`, `diffusion.py`, and `generate.py`; the pinned H100 graph comparison ran 08:04:01–08:06:37 before the observed `fast.py` edit at 08:07:48. The end-to-end CLI smoke ran 08:06:43–08:08:19 and may have imported code during the concurrent edits. Findings about the fast graph are grounded in the committed version; the revised uncommitted version needs its own verification.

## Part A — public entry points and effective coverage

| Public entry point | Classification and reproduced evidence | Upstream correspondence checked |
|---|---|---|
| `StructureDiffusionPredict.forward`, checkpoint converter, four shipped configs | **Real numerical coverage when manually invoked.** `test_model_parity.py` compares 11 continuous outputs and exact `aatype` to independent fixture arrays with tight tolerances; all four configs exited 0 here. **Vacuous if a fixture/checkpoint is missing:** it prints `SKIP` and returns 0. Its provenance fields are self-asserted (Part B). | `upstream/salad/salad/modules/noise_schedule_benchmark.py:643-645,1077-1118` |
| `StructureDiffusionInference.forward`, `scripts/generate.py` | **Missing regression coverage.** The model parity script calls `StructureDiffusionPredict`, not this wrapper. N010 did run a two-step GPU CLI smoke to PDB, but no test checks its output against JAX. | `upstream/salad/salad/modules/noise_schedule_benchmark.py:442-448`; `upstream/salad/salad/training/eval_noise_schedule_benchmark.py:536-544,605-625` |
| `FastDenoiser` compiled/captured inference | **Missing parity test.** None of the five `test_*.py` files references it. N010's pinned H100 reproducer is the first targeted parity run here; it found failures below. | `upstream/salad/salad/modules/noise_schedule_benchmark.py:643-645,1077-1118` |
| `diffusion.sampling.make_initial_data`, `sample`, `SamplerConfig` | **Missing direct coverage.** `test_sampling_parity.py` hand-codes its own loop (`:93-116`) and never calls `sample`. That manual trajectory comparison is meaningful but coarse: current run had 4.370 Å worst step and 0.05374 Å mean final deviation, and passed the documented chaos-scaled bounds. Missing fixture/checkpoint returns 0. | `upstream/salad/salad/training/eval_noise_schedule_benchmark.py:536-546,590-625` |
| `Encoder.forward` | **Real numerical coverage when manually invoked.** `test_encoder_parity.py:40-77` loads real weights and checks local/coordinate outputs at two shapes against JAX arrays; it exited 0. Missing fixture/checkpoint returns 0. | `upstream/salad/salad/modules/noise_schedule_benchmark.py:694,745-785` |
| `DiffusionStack` and geometry primitives | **Real numerical coverage when manually invoked.** `test_stack_parity.py:98-120` compared local/pos/trajectory; `test_geometry_parity.py:45-144` passed 38 independent array checks. Stack silently returns 0 on missing fixture; geometry fails at import if its fixture is missing. No self-comparison was found in these checks. | `upstream/salad/salad/modules/noise_schedule_benchmark.py:1587,2105`; `upstream/salad/salad/modules/utils/geometry.py:51,293,516` |
| `total_loss`, sequence/structure/violation terms | **Vacuous regression check.** `tests/harness/check_loss.py` and `check_loss2.py` print `FAIL` for a corrupted reference and return `None` (process exit 0); neither is collected by pytest. The 12 terms are computed, but no assertion gates them. Only the `default_vp` path is checked by these scripts; the shipped alternate configs' training loss is not checked. | `upstream/salad/salad/modules/noise_schedule_benchmark.py:1182,1199-1235,1416-1424` |
| `PackedProteinStream`, `NpzAssemblySource`, synthetic source, `concat_collate`, `split_chunks`, `example_to_tensors`, DSSP | **Missing regression coverage.** No public test calls these. N010's synthetic `next_item()` at size 32 only establishes a smoke path; the real on-disk source was not exercised. Size 16 crashes (Part B). | `upstream/salad/salad/data/allpdb.py:812`; `upstream/salad/salad/training/train_noise_schedule_benchmark.py:160-180` |
| `prepare_batch`, `reveal_conditioning`, `training_step`, `Trainer`, checkpoint save/load, `training.train.main` | **Missing regression coverage.** No public test invokes these. The prior synthetic-step account is not an automated gate; N010 did not rerun optimizer steps. | `upstream/salad/salad/modules/noise_schedule_benchmark.py:106-160,223-258`; `upstream/salad/salad/training/train_noise_schedule_benchmark.py:160-183` |
| `inference.symmetry`, template/options/postprocess, generation flags | **Missing regression coverage.** No public test references these paths. Direct N010 symmetry probes found discrepancies below. N010 ran a two-step symmetry+template+sequence CLI smoke; non-template motif and full 500-step quality were not checked. | `upstream/salad/salad/training/eval_noise_schedule_benchmark.py:180-219,364,501-508,537-544,611-619` |
| `snapshot_to_protein`, `to_pdb`, `write_pdb` | **Missing regression coverage.** No public test calls them. N010 manually compared four small `to_pdb` layouts with the importable upstream writer; strings matched exactly. A two-step `generate.py --fast` smoke wrote a 303-line PDB, but neither check is part of the suite. | `upstream/salad/salad/aflib/common/protein.py:238-315`; `upstream/salad/salad/training/eval_noise_schedule_benchmark.py:659-669` |

Literal test inventory / run output:

```text
$ python -m pytest tests -q
no tests ran in 1.11s
pytest_exit=5
$ rg '^def test_|^class Test' tests --glob '*.py'
[no matches]
$ rg 'FastDenoiser|training_step|PackedProteinStream|write_pdb|to_pdb|total_loss|screw_replicate' tests --glob 'test_*.py'
[no matches]
$ python tests/test_geometry_parity.py; python tests/test_stack_parity.py; python tests/test_encoder_parity.py
38/38 passed
[PASS] coordinates  mean=0.0000 A  max=0.0003 A
[PASS] L96c3 coords mean=0.00000 A max=0.00001 A
$ SALAD_CONFIG=default_vp_scaled python tests/test_model_parity.py
coordinate deviation: mean 0.00000 A, max 0.00002 A
$ SALAD_CONFIG=default_ve_scaled python tests/test_model_parity.py
coordinate deviation: mean 0.00000 A, max 0.00002 A
$ SALAD_CONFIG=multimotif_vp python tests/test_model_parity.py
coordinate deviation: mean 0.00001 A, max 0.00020 A
$ python tests/test_sampling_parity.py
worst per-step deviation over the whole trajectory: 4.370e+00 A
final structure deviation: mean 0.05374 A, max 0.35083 A
OK
```

The forced-missing reproducer patches `os.path.exists` only in process memory; source/fixtures were not moved:

```text
SKIP: missing /hnvme/workspace/v181bb10-salad/default_vp-200k.npz
model missing return 0
sampling missing return 0
SKIP: missing /home/hpc/v181bb/v181bb10/SALAD_torch/fixtures/block_ref.npz
stack missing return 0
encoder missing return 0
```

## Part B — ranked findings

### Silently wrong

**S1. Critical — same-sum conditioning collision returns the previous design's denoised coordinates.** The committed cache key sums each conditioning tensor (`HEAD:salad_torch/modules/fast.py:210-223`). Swapping the DSSP labels preserves the sum but changes the condition passed to the model. With *identical* `uniform` pinned to a `(19,48,48)` stack on H100, the fast second result remains the first result within 8e-6 Å and differs from eager on the second condition by 2.562 Å. Explicit invalidation returns to 1.1e-5 Å agreement. Upstream builds conditioning from the actual per-residue values (`upstream/salad/salad/modules/noise_schedule_benchmark.py:482-513,2527-2586`), and predicts with it (`:643-645`). [Reproducer](repro_fast_gpu.py); [literal Slurm output](repro_fast_recheck.log):

```text
uniform_shapes (48, 48) (19, 48, 48)
base_max 7.62939453125e-06
repeat_max 7.62939453125e-06 held_change 0.0
collision_sums 24 24 fast_cross 7.62939453125e-06 fast_vs_eager 2.5619845390319824
refresh_vs_eager 1.0967254638671875e-05
```

The separate CPU dummy-model reproducer isolates the key logic without compilation:

```text
CACHE first [0.0, 1.0, 2.0, 3.0] second [0.0, 1.0, 2.0, 3.0] eager [3.0, 2.0, 1.0, 0.0] refreshed [3.0, 2.0, 1.0, 0.0]
```

**S2. High — a captured `FastDenoiser` accepts a different length and silently returns the old-length output.** `_run` replaces a static-input dictionary entry on shape mismatch, but the captured graph still reads the addresses from capture (`HEAD:salad_torch/modules/fast.py:167-185`). A length-48 instance fed length-49 tensors and a correctly shaped new uniform stack returned 48 coordinates. Upstream's forward uses the supplied residue-axis length (`upstream/salad/salad/modules/noise_schedule_benchmark.py:997-1043,1077-1079`). [Pinned H100 output](repro_fast_recheck.log):

```text
length_mismatch_returned 48
```

**S3. High — the symmetry hook constrains `snapshot["pos"]`, but the PDB writer uses the unconstrained `snapshot["atom_pos"]`.** `sample()` applies Hook C to `data["pos"]` and then stores the model's unchanged `update["atom_pos"]`; `snapshot_to_protein()` converts that unchanged atom14 tensor to the PDB (`salad_torch/diffusion/sampling.py:170-188`; `salad_torch/io/pdb.py:172-180`). The upstream eval script has the **same behavior**, setting symmetrized `data["pos"]` but copying the pre-hook `update["atom_pos"]` and writing it (`upstream/salad/salad/training/eval_noise_schedule_benchmark.py:610-623,659-669`). This is a user-visible symmetry output error reproduced from upstream, not a new port divergence. A one-step toy isolates it, with the exact C4 hook and writer; a two-step H100 CLI smoke with `--sym c4 --template ... --template_aa` also completed and wrote an unsymmetric PDB. The two-step structure is too short a trajectory to evaluate model quality, but the toy shows the output plumbing error independent of model quality. [Toy reproducer/output](repro_sym_snapshot.log), [CLI job](repro_sym_template_slurm.log), [PDB residual](repro_pdb_sym_residual.log):

```text
SNAPSHOT pos_C4_maxcoord 4.76837158203125e-07
SNAPSHOT atom_pos_C4_maxcoord 2.2078309059143066
WRITTEN atom37_CA_C4_maxcoord 2.2078309059143066
config default_vp  L=48  steps=2 out_steps=(1,)  sym=c4(count=4)
  design 0:    0.5s  CA-CA 2.097 A (min 0.952 max 2.728)  Rg 11.1 A  -> .agent-research/results/N010-codex-redteam/generated_sym_template/design_0.pdb
PDB_C4 ... ca_count 48 rms 6.627384065795158 max 16.244416579243467
```

**S4. High verification failure — loss checkers can print `FAIL` and exit successfully.** Both `tests/harness/check_loss.py:97-113` and `check_loss2.py:97-113` label a term `FAIL` but never assert/return failure. N010 copied each fixture inside this folder, increased `loss_pos` by 1000, and called `main()`. Upstream computes the corresponding term as part of the objective (`upstream/salad/salad/modules/noise_schedule_benchmark.py:1232-1274`). [Reproducers](repro_verification.py), [second checker](repro_loss2_vacuous.py):

```text
pos  1018.5926151276  18.5926189423  1.000e+03  9.817e-01 FAIL
LOSS CHECK main returns None
pos  1007.4770302773  7.4770317078  1.000e+03  9.926e-01 FAIL
check_loss2 returned None
```

**S5. Medium verification failure — model-fixture provenance can be forged.** Only the four `model_*.npz` fixtures carry backend/precision fields; `block_ref`, `stack_ref`, `loss_ref*`, `viol_ref*`, `encoder_default_vp`, `geometry`, `sampling_default_vp`, and `kabsch_ref` record neither field (see `repro_cpu.log`). The model parity guard trusts those strings (`tests/test_model_parity.py:70-80`). N010 copied an actually CPU-produced fixture, set its claimed platform to `gpu` and precision to `highest`, and the entire model parity test accepted and passed it. A second copy also perturbed `out_raw_angles` by +0.01; it still passed under that output's declared 1e-4 relative tolerance. This demonstrates unverifiable provenance and an accepted *small* numeric error, **not** that the currently stored unlabelled fixtures are numerically wrong or that a TF32-scale structural error passes. The fixture inventory literally printed `FIXTURE block_ref.npz {}`, `FIXTURE stack_ref.npz {}`, `FIXTURE loss_ref.npz {}`, `FIXTURE loss_ref2.npz {}`, `FIXTURE viol_ref.npz {}`, `FIXTURE viol_ref2.npz {}`, `FIXTURE encoder_default_vp.npz {}`, `FIXTURE geometry.npz {}`, `FIXTURE sampling_default_vp.npz {}`, and `FIXTURE kabsch_ref.npz {}`. The oracle being checked is upstream `StructureDiffusionPredict` (`upstream/salad/salad/modules/noise_schedule_benchmark.py:643-645,997-1118`). [Reproducer/output](repro_verification.log):

```text
MODEL TEST spoofed GPU backend on original CPU fixture
fixture provenance: platform=gpu matmul_precision=highest
coordinate deviation: mean 0.00000 A, max 0.00002 A
OK
MODEL TEST main returns 0
$ python .agent-research/results/N010-codex-redteam/repro_wrong_fixture.py
INJECT out_raw_angles +0.01; provenance falsely gpu/highest
[PASS] raw_angles  rel=3.295e-05  abs=1.10e-02  (tol 1e-04)
OK
test_return 0
```

Generator audit (literal command: `rg -n 'jax_default_matmul_precision|np.savez' tests/harness/gen_* scripts/gen_fixtures_*`):

```text
tests/harness/gen_block_ref.py:9:jax.config.update("jax_default_matmul_precision", "highest")
tests/harness/gen_encoder_ref.py:28:jax.config.update("jax_default_matmul_precision", "highest")
tests/harness/gen_loss_ref.py:17:jax.config.update("jax_default_matmul_precision", "highest")
tests/harness/gen_loss_ref2.py:17:jax.config.update("jax_default_matmul_precision", "highest")
scripts/gen_fixtures_geometry.py:19:jax.config.update("jax_default_matmul_precision", "highest")
scripts/gen_fixtures_sampling.py:27:jax.config.update("jax_default_matmul_precision", "highest")
tests/harness/gen_stack_ref.py:57:np.savez(sys.argv[3], out_local=np.asarray(f), out_pos=np.asarray(p),
```

`gen_stack_ref.py:1-58` has no precision pin or metadata write, so a future GPU regeneration can repeat the TF32 problem; the already-stored stack fixture passed current numerical parity, which does not prove its backend. No checked-in producer for `viol_ref*` or `kabsch_ref` was found by `rg -n 'viol_ref|kabsch_ref' tests/harness scripts` (no matches). The other listed generators pin precision in source but still omit recorded backend/precision from their NPZ output.

**S6. Medium — symmetry leaves unconstrained tails and large-order inputs unchanged; `c1` skips upstream centering.** For L=9, C4 the ninth residue is passed through unchanged; for order greater than L all positions pass through. For C1, the port returns the input whereas upstream runs `cyclic_centering` then subtracts the final CA mean (`upstream/salad/salad/training/eval_noise_schedule_benchmark.py:176-179,189-219`). The tail-preservation is documented in the port as a shape-stability choice (`salad_torch/inference/symmetry.py:136-150,186-188`), but it silently fails the requested symmetry on those residues. Direct source-formula probe and outputs (`repro_cpu.log`, `repro_clash.log`, `repro_symmetry_extra.log`):

```text
SYM 9 4 out_length 9 identity False remainder_equal True
SYM 3 4 out_length 3 identity True remainder_equal True
C1 input_CA_center [25.5, 26.5, 27.5] torch_CA_center [25.5, 26.5, 27.5] upstream_formula_CA_center [0.0, 0.0, 0.0] max_delta 27.5
C4 residual 4.76837158203125e-07
ORDER 64 L 8 unchanged True
```

**S7. Low — the symmetry clash nudge uses a different distance formula.** Upstream clamps each signed coordinate difference before squaring (`upstream/salad/salad/training/eval_noise_schedule_benchmark.py:204-211`); the port squares first then clamps (`salad_torch/inference/symmetry.py:118-125`). Applying the literal upstream expression with Torch autograd to the same four CA points changes the clash gradient by up to 6.76e-5. This only affects the `chain_center` hook. [Reproducer/output](repro_clash.log):

```text
CLASH max_delta 6.764131831005216e-05
```

**S8. Low — `FastDenoiser(compute_distogram=True)` omits the requested output.** The argument is accepted but never used by the committed fast forward (`HEAD:salad_torch/modules/fast.py:192-255`); upstream unconditionally emits `distogram` (`upstream/salad/salad/modules/noise_schedule_benchmark.py:1090-1095`). A CPU dummy-model probe of the current fast API returned:

```text
FAST compute_distogram=True key_present False
```

### Crashes

**C1. Low — `--sym c0` divides by zero in parsing.** The parser computes 360/N with no positive-order check (`salad_torch/inference/symmetry.py:71-77`); upstream exposes a numeric `replicate_count` and divides the length by it (`upstream/salad/salad/training/eval_noise_schedule_benchmark.py:189`). [Reproducer/output](repro_cpu.log):

```text
SYM c0 ZeroDivisionError float division by zero
```

**C2. Low — the synthetic training source crashes at the allowed minimum size of 16.** `build_source` asks NumPy for integer lengths in `(16, min(300,num_aa))`, an empty interval at `--num-aa 16` (`salad_torch/training/train.py:81-85`). The upstream real-data training stream allows `size=num_aa, min_size=16, max_size=num_aa` (`upstream/salad/salad/training/train_noise_schedule_benchmark.py:160-167`). [Reproducer/output](repro_data.log):

```text
DATA 16 ValueError low >= high
DATA 32 valid 19 shape (32, 24, 3)
```

### Attacks with no failure observed

- Correctly shaped, pinned fast inputs repeated to 7.63e-6 Å; held output stayed unchanged (`held_change 0.0`), and two graph instances interleaved to 1.03e-5 Å (`repro_fast_recheck.log`; literal output: `interleave_f2_change 1.0251998901367188e-05`). Upstream forward reference: `upstream/salad/salad/modules/noise_schedule_benchmark.py:643-645`.
- Length-1 and length-2 losses remained finite; fully masked inputs produced zero for all 12 terms; duplicating an 8-residue item into a two-item batch reproduced the single-item total exactly (`repro_loss_edge.log`). Literal output: `EDGE 1 normal total 94.03773498535156 finite True zero_terms []`; `EDGE 1 masked total 0.0 finite True`; `BATCH_DUP one 130.22137451171875 two 130.22137451171875 delta 0.0`. These reductions agree with the upstream masks and batch weighting at `upstream/salad/salad/modules/noise_schedule_benchmark.py:1199-1235` for the probed examples. No optimizer step was run.
- Four small PDB cases matched upstream text byte for byte (`repro_pdb.log`; literal output: `PDB [0] identical True lines 8`, `PDB [0, 0] identical True lines 12`, `PDB [0, 1] identical True lines 13`, `PDB [0, 2] identical True lines 13`); upstream writer: `upstream/salad/salad/aflib/common/protein.py:238-315`.
- A full 500-step C4 generation and non-template motif combination were **not verified**; the direct C4 replicate had a 4.77e-7 coordinate residual (`repro_symmetry_extra.log`). JAX container execution failed because `squashfuse_ll` could not find `/dev/fuse` (`repro_sym_jax.log`; literal output: `FATAL: container creation failed: image driver mount failure: image driver squashfuse_ll instance exited with error: squashfuse_ll exited: fuse: device not found`); the exact upstream source lines cited above were inspected instead.

N010's short `generate.py --fast` run did complete and write `generated/design_0.pdb` (303 lines). It used two denoising steps solely as an API/path smoke, so it is no evidence of generated-structure quality:

```text
fast path compiled + captured in 89s (one-time)
  design 0:    0.1s  CA-CA 2.646 A (min 1.475 max 3.370)  Rg 8.1 A  -> .agent-research/results/N010-codex-redteam/generated/design_0.pdb
```

## Fix / verification plan

1. Replace the sum cache key with a collision-resistant per-tensor identity/version or content digest; include shape/dtype/device and reject in-place same-sum edits. Add a pinned two-conditioning regression to the fast graph path.
2. Validate every static graph input shape, dtype, and device before copy/replay; rebuild or raise on mismatch. Assert returned residue length. Also reject an incorrectly shaped `uniform` stack.
3. Make loss checkers raise/return nonzero on any failed term and give pytest actual callable test functions. Change fixture-missing paths from success to explicit skip status or failure in required CI. Attach a signed/hash-checked fixture manifest with backend/precision metadata for all references.
4. Rebuild atom14/atom37 coordinates from the post-hook symmetric backbone before writing, or make the PDB output contract explicitly distinguish pre-hook atoms from the symmetric backbone; verify the written C4 coordinates. Validate symmetry order/divisibility and C1 centering. Match the upstream clash distance expression if exact parity is required.
5. Add tests of the *public* `sample`, training/data/checkpoint, PDB writer, and CLI fast path; run relevant default and conditioned configs. Recheck the concurrent uncommitted fast-path changes before integration.

No `.agent-research/notebook` entry or commit was made: the task's explicit write boundary permits only this N010 results directory. No source file was changed by this audit.

# N024 red-team findings (no code changed)

## Verdict

**The two targeted commits do not explain the severe legacy `--fast --preset fast` quality regression.** The regression survives disabling `fd9ba32`'s CSE pass, and disappears when `603051e`'s P7 batched condition branch is disabled while CSE remains active. The P7 change predates both targeted commits. The archived `603051e` code, before either target commit, already produces the same failure (5/32 invalid designs); P7-off on that same snapshot restores 0/32 invalid designs.

## Controlled full-rollout experiment

Command: `scripts/generate.py --fast --preset fast --num_aa 100 --num_designs 32 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz`, with `SALAD_OPT_DETSEG=1` to eliminate CUDA segment-reduction self-divergence. The four HEAD arms used the same current source (`274dfc4`, whose only difference from `df4f470` is the new quality test), independent compiler caches, one H100 per run, and the unchanged fast preset 250/200 schedule. Geometry was scored by `tests/test_design_quality.py:metrics`. GPU placement is required to test the production CUDA graph; no training was run. The toggles were applied before `FastDenoiser` construction. Scripts and full logs are in this folder.

| Arm | CSE | P7 `BATCH_COND` | TF32 | Diverged / 32 | Mean Rg, survivors (Å) | Mean CA–CA, survivors (Å) |
|---|---|---|---|---:|---:|---:|
| Production controls | on | on | on | 7 | 20.02 | 3.8832 |
| CSE disabled | off | on | on | 5 | 20.62 | 3.8793 |
| P7 disabled | on | off | on | **0** | **13.41** | **3.7568** |
| TF32 disabled | on | on | off | 7 | 20.46 | 3.8772 |
| Historical `603051e` (before both targets) | absent | on | on | 5 | 19.74 | 3.8594 |
| Historical `603051e`, P7 disabled | absent | off | on | **0** | **13.44** | **3.7594** |

Sources: `metrics_all.txt`, `metrics_at603.txt`, `cse_904711.log`, `p7_904712.log`, `tf32_904743.log`, `at603_904751.log`, `at603p7off_904763.log`, `metrics_at603_p7off.txt`. The historical arms use a `git archive 603051e salad_torch scripts` snapshot in `at603/` and the same seeded command, checkpoint, H100 class, and DETSEG setting. The CSE-on arm contains designs with mean CA–CA 299–471 Å and Rg 363–530 Å (`cse_904711.log:45,49,56,58,61`). The CSE-off arm still contains five such invalid designs (`cse_904711.log:83,94,96,98,99`). The P7-off arm has normal CA–CA near 3.76 Å across all 32 (`p7_904712.log:30-63`). The full-output difference between CSE-on and CSE-off at equal seed under DETSEG can be large (`paired_cse.txt`), so the CPU forward bit-identity claim does **not** establish full GPU rollout identity; the severe quality phenotype remains with CSE removed.

## Hypothesis findings

### A1. CSE merges stateful or impure graph nodes

**Not supported as the severe regression's cause.** The current `fx_graph_cse` preserves RNG ops, keeps distinct output storages, and computes mutation regions before deduplicating (`torch/_functorch/compile_utils.py:42-157`). A source audit of the compiled `forward_core` path found no in-place mutation, `.data` read, or global RNG operation in the default inference route. The `uniform_` fallback in `geometry/neighbours.py:140-148` is skipped because every selector receives a pre-drawn field (`modules/diffusion.py:267,283-285,327-355`). Most decisively, CSE-off still gives 5/32 invalid designs and Rg 20.62 Å. This does not prove the private pass safe for every possible future graph or configuration.

### A2. CSE collapses independent random draws

**Refuted for this path.** `FastDenoiser._prepare_inputs` draws a fresh `(n_draws,L,L)` field every step outside the captured graph (`modules/fast.py:684-703`), and `_copy_in` writes it to the graph's static `uniform` buffer (`modules/fast.py:559-606`). `forward_core` unbinds that input and passes distinct slices to the neighbour selectors (`modules/diffusion.py:267,323-355`). In this installed PyTorch 2.5.1 build, `fx_graph_cse` explicitly excludes `rand_ops` (`compile_utils.py:20-39,87-98`); a direct `make_fx` probe with two non-output-alias `aten.rand` calls retained both (`cse_rng_unit.log`: 2 before, 2 after). P7-on failures persist with CSE disabled.

### A3. Process-global post-pass leakage

**Confirmed scope leak, but not this CLI failure.** `FastDenoiser.__init__` assigns `_ind.joint_custom_post_pass = _cse_graph_pass` (`modules/fast.py:417-418`) and never restores the prior value. A direct CPU-process probe found `None` before construction, `_cse_graph_pass` afterward, and the same pass after deleting the denoiser (`global_pass_probe.log`). A later unrelated Inductor compile in the same Python process inherits CSE. The legacy generator builds its denoiser after loading the model (`scripts/generate.py:730-747`) and then samples; this leak does not explain the observed production trajectory. The `7eb9a10` follow-up moves the pass to module scope for pickling; it does not change this global lifetime (`git show 7eb9a10 -- salad_torch/modules/fast.py`).

### B1. Split forward changes static buffer write/read timing

**Refuted as the severe regression's cause.** In the legacy path, `forward` calls `_prepare_inputs`, `_run`, `_postprocess` in order (`modules/fast.py:657-677`). `_run` calls `_copy_in` before graph replay (`modules/fast.py:604-618`); `_copy_in` copies every `_CORE_ARGS` tensor into its persistent static buffer (`modules/fast.py:559-594`). The pre-`b9dd416` `_run` had the same copy loop immediately before replay (`git show b9dd416^:salad_torch/modules/fast.py`, `_run`). No grouped driver is entered when `--group_size` is unset (`scripts/generate.py:725-728`). P7-off restores quality while retaining this refactor.

### B2. `pin_condition` serves stale conditioning

**Refuted for the requested legacy command.** `pin_condition` is called only by `GroupFastDenoiser.pin_conditions` (`modules/fast.py:1171-1184`), itself called by the grouped CLI (`scripts/generate.py:583`); the unset `--group_size` path instead uses the content-hash cache in `_prepare_inputs` (`modules/fast.py:697-749`). The grouped driver invalidates before each design group (`scripts/generate.py:570-583`). During normal sampling only dynamic state, not hash-key conditioning, is updated (`diffusion/sampling.py:340-377,449-473`). A caller changing conditioning after pinning without invalidation could serve stale data, but that is outside this command and was not stress-tested here.

### B3. Self-conditioning feedback broken by the split

**Refuted as the severe regression's cause.** `_pack_inputs` still maps `prev["pos"]` and `prev["local"]` to `prev_pos` and `prev_local` (`modules/fast.py:751-764`); `_postprocess` still returns freshly cloned output `pos` and `local` as `new_prev` (`modules/fast.py:767-798`). `sample()` assigns `update["pos"]` to `data["pos"]` for the next step and takes the returned `prev` (`diffusion/sampling.py:359-367`); this existing function was not edited by `b9dd416` (that commit only added `sample_group`). `prev` resets to the initial value when `raw_t < 0.8`, as before (`sampling.py:353-357`). P7-off restores quality with all these paths unchanged.

### B4. Grouping shifts the distribution

**The cited legacy-vs-group helix comparison is not an identity test.** The legacy path uses one generator seeded with `seed` (`scripts/generate.py:749`); the grouped path seeds each design with `seed * 1_000_003 + design` (`scripts/generate.py:373-390`). Thus identical design indices do not get identical noise across these two entry points. N019's valid grouped `group_size=1` vs `8` DETSEG comparison reported 8/8 bit-identical positions in both presets (`.agent-research/results/N019-multidesign-occupancy/logs/final_904175.log:59-75`). I did not repeat that comparison on this HEAD, so a new grouping-only defect is not ruled out; it cannot cause the unset-`--group_size` regression.

## Ranked cause assessment

1. **`603051e` P7 batched condition branch (high confidence).** It landed between `c5ab7a0` and `b9dd416` (`git log c5ab7a0..b9dd416`). Its default-on `BATCH_COND` setting is the only tested switch that restores normal geometry when turned off, while leaving both targeted commits present. The archived commit already shows 5/32 invalid designs before the targeted commits landed, and 0/32 with P7 off on that same snapshot. An eager GPU `forward_core` comparison at L=100, unconditioned, `SALAD_OPT_DETSEG=1`, `torch` matmul precision `high` was exactly equal on `pos` and `local` with P7 on/off (`p7fwd_904764.log:25-37`). This narrows the problem to P7 in the compiled/captured production path, or to state-dependent inputs absent from that single eager probe; it does not identify the exact faulty kernel or state transition. The earlier P7 checks were forward parity and same-input numerical controls (`.agent-research/results/N014-batched-gemm/README.md:67-122`), not full production rollouts. The exact internal fault inside P7 or its compiled lowering is not yet isolated.
2. **CSE changes per-design trajectories but is not the severe quality driver (low as primary cause).** On/off outputs are not identical under DETSEG (`paired_cse.txt`), but both arms have invalid proteins and almost identical survivor Rg. A small interaction cannot be ruled out from 32 designs.
3. **`b9dd416` legacy refactor (very low).** Source order is preserved, pinning and grouping are not reached, and P7-off rescues the identical refactored path.

## Investigation plan (no fix made)

Capture the same real sampling state and injected uniform field immediately before the first P7-on/P7-off difference. Compare eager, compiled without capture, and compiled with capture outputs of `StackedCondPair` and `forward_core` under DETSEG, then inspect the first mismatching Inductor kernel and its static inputs. This will distinguish a state-dependent P7 computation error from compiled/captured lowering or stale-buffer use. Keep the rollout quality gate as the acceptance check.

## Limits and unchecked work

- The exact internal P7 fault is not yet isolated. Eager one-forward equality rules out a simple eager arithmetic discrepancy on the synthetic unconditioned L=100 input, but a state-dependent P7 discrepancy or compiled/captured lowering issue remains.
- I did not inspect the full optimized Inductor FX graph node by node or test every supported conditioning configuration. The CSE purity conclusion is limited to the default L=100 inference route plus the direct RNG probe.
- I did not rerun current-HEAD grouped `group_size=1` vs `8`, or measure secondary structure in these arms. Geometry and invalid-design counts are the decision metrics here.
- All repository writes made by this task are confined to this `codex/` folder; Inductor used ephemeral compiler caches under `/tmp`. No live source edits, commits, or PR actions were made by this red-team task.

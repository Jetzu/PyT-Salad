# N026 Codex root cause and fix round

**Verdict:** PyTorch 2.5.1 Inductor miscompiles one fusion: `op195` (six-block condition-neighbour `stack`) with `op207` (indirect gather for tiled relative-position bins). A graph break after the stack restores correctness and retains 95.1% of the measured P7 launch-saving benefit. This is a patch artifact only; no production source was changed by this work.

## First bad fusion

The N025 real step-30 L=100 state was rebuilt with `DET_SEGMENT=1`, fp32, `torch.compile(mode="default", dynamic=False)`, no CUDA graph. P7 was set directly with `opt.BATCH_COND=True` for the forward probes. The source is `bisect_forward.py`; all inputs to eager and compiled arms are the same saved state. The trace archives `trace_on.tar.gz` and `trace_off.tar.gz` contain `output_code.py`, scheduler IR before/after fusion, and FX graph dumps. `on.fusions.tsv`/`off.fusions.tsv` and `fusion_decisions.diff` contain the scheduler fusion decisions. `on.kernels.tsv`/`off.kernels.tsv` list each generated Triton kernel; `p7_on_only_groups.tsv` lists all 176 P7-on groups whose exact FX source-node signature is absent from the P7-off trace. Later FX node numbers shift when P7 is enabled, so this exact-name diff is an inventory, not proof that all 176 are semantically unique.

The accepted-fusion prefix bisect made the first bad decision exact: prefix 14 gives `pos` error 1.86e-5 A; prefix 15 gives 9.586393 A (`bisect_907076.log:39-40`). The 15th decision is `op195 + op207` (`lim15.log:20`), and vetoing that pair alone in the otherwise fully fused graph gives 1.81e-5 A (`pair_907104.log:26-28`). The source IR has `op195` writing an int64 `[600,48]` neighbour stack and `op207` reading it for indirect residue/chain/batch indices (`trace_lim14.tar.gz`, `ir_pre_fusion.txt`).

At prefix 14, Inductor emits two generated kernels (`limit14_unfused_kernels.py`); at prefix 15, it emits `triton_poi_fused_add_clamp_eq_full_like_index_repeat_stack_sub_where_72` (`limit15_fused_kernel.py:48`). In the fused source, the three indirect `tl.load`s at lines 217, 222, and 225 are guarded by `tmp102 & tmp36 & tmp4 & tmp58 & tmp80 & xmask`. Here `tmp4` is `row < 100` (line 63) and `tmp36` is `100 <= row < 200` (lines 92-96), so the mask is false on every lane. In the unfused kernel the corresponding loads use only `xmask` (`limit14_unfused_kernels.py:252-260`). This is the first numerical error; the batched pair projections consume wrong integer bins.

## Exact kernel-level comparison

`capture_kernel.py` saved every pointer input and both outputs of that exact fused Triton launch during a model forward (`real_kernel_io.pt`). `standalone_kernel.py` then launched the generated fused kernel and the two generated unfused kernels **standalone on the same captured real inputs**. Both stack outputs match bit for bit. The fused relative-position bins differ from the unfused ones in **23,070/28,800** positions, maximum absolute error **64 bins** (`standalone_907151.log:27-31`). The fused standalone output also matches the captured model-forward output bit for bit. Example: row 0, neighbour 1 has bin 32 fused versus 31 unfused. A separate pure-PyTorch recomputation on the captured stack agrees with the unfused result and gives the same 23,070 errors (`kernel_numerics.txt`).

The no-salad issue reproducer is `minimal_conditional.py`. It constructs six conditional neighbour arrays from seeded synthetic tensors and computes the same tiled relative-position bins. Default Inductor gets **22,222/28,800** int64 bins wrong, maximum error 64; `max_fusion_size=1` gets **0** wrong (`minimal_907142.log:25-28`). `UPSTREAM_ISSUE.md` gives a concise issue text and the generated kernel source.

## Tested fix

`fix.patch` is a patch against the branch index (`git apply --check --cached` passed). It inserts `torch._dynamo.graph_break()` immediately after `cond_nbr_stack = torch.stack(...)` in `Diffusion.forward_core`, before `StackedCondPair` computes relative positions; it restores `batch_cond_pair=True` in the three presets and the default P7 flag. `patch_runtime.py` applied that exact graph break to a runtime copy of the method for validation, without editing production files. The source graph break is local to the P7 branch and leaves the batched bmm projections compiled.

One real-state forward with P7 on and the graph break gives `pos` max abs 1.53e-5 A, `local` 7.32e-4, and trajectory 3.58e-5 against eager P7 on (`barrier_907155.log:26-29`). The compiled P7-on baseline is 9.586 A away; the P7-off compile floor is about 2e-5 A (N025, `RESULTS_PART2.md`).

The required GPU quality run passed with **0/160 diverged** at L=100, `--preset fast`, CA-CA 3.7631 A and Rg 13.05 A (`quality_907167.log:190-194`). The job set `SALAD_OPT_BATCHCOND=1`, and `quality_p7_marker.txt` records the runtime assertion that `FastDenoiser` actually enabled `opt.BATCH_COND`. PDBs and Inductor caches were written under the node-local `$TMPDIR` and removed by the test.

Whole-forward lean time at L=100, separate process per arm, three 30-iteration reps each, CUDA graph fast path (`bench_n026bench_{off,broken,barrier}_*.log`): P7 off 2.86060 ms, broken P7 on 2.66596 ms, fixed P7 on 2.67550 ms. The fix is **6.47% faster than P7 off** and retains **95.1%** of P7's measured gain; its cost against broken P7 is 0.36%. The log lines show `P7 False/True/True` for the three arms.

## CPU checkpoint parity

[Pending final DET_SEGMENT=0 control. The first four-config run set DET_SEGMENT=1; three configs passed and default_ve_scaled missed only atom_pos tolerance. This setting differs from the prior checkpoint-parity gate and changes reduction order.]

## Integration plan and limits

Apply `fix.patch` to the intended working branch, reconcile it with any concurrent working-tree edits, then run the same GPU quality/benchmark and CPU parity gates on the **integrated** tree. The patch and test harness in this folder are the implementation to integrate; no production file was edited here. The current live tree gained separate, uncommitted P7 edits while these jobs ran, so the patch is explicitly based on the branch index and should not be applied blindly over those edits.

Not checked: newer PyTorch releases, GPUs other than H100/sm_90, training/grad-enabled P7 (which the model deliberately disables), and the integrated production patch after another agent lands it. No commit or PR was made because this fix-round scope permits writes only under `codex/`.

# N029 — the 4.6 GiB L2 traffic gap (E7): attribution, and what is real

Read `RESULTS.md`. Headline: E7 closes as a negative result — the 5.2x is
1.75x definitional (N016's minimum counts each tensor once; any implementation
writes *and* reads it), 2.08x hardware access amplification, and 1.36x
DRAM-side. The pointwise-fusion prize is **82.4 MiB**, not 3.9 GiB. The one
large addressable class is **776 MiB / 67 launches of `clone`-for-`bmm`**
(N028's O2). A by-product, O1, is **−7.3 % of a lean step** — and it is a
*launch* win, not a traffic win, which is the whole point.

## Files

| file | what |
|---|---|
| `RESULTS.md` | the item |
| `O1_pairvec_share.patch` | the O1 code change (NOT landed — see RESULTS §5) |
| `attribution_tables.txt` | per-kernel logical traffic, measured-vs-logical join, producer/consumer flow |
| `ncu_per_kernel.txt` | ncu per-group table with the read/write split |
| `o1_eager_arm_diff.txt` | the arm-vs-arm eager deviation that refuted O1's bit-exactness |
| `kernel_times_*.json` | per-kernel GPU time censuses |
| `scripts/x_graph.py` | traced-ATEN-graph analysis (unfused / minimum / matmul census / duplicates) |
| `scripts/x_ncu.py`, `x_ncu_preset.py` | the Nsight Compute harness (N016's, at this HEAD) |
| `scripts/parse_ncu2.py` | weighted ncu summary parser — handles sector units and thousands separators |
| `scripts/x_code.py` | dumps inductor `output_code` via `TORCH_LOGS=output_code` |
| `scripts/parse_code.py` | per-buffer / per-kernel LOGICAL traffic from `output_code` |
| `scripts/flow.py` | producer→consumer traffic classification (the fusion-prize number) |
| `scripts/join.py` | joins measured ncu bytes to logical bytes per Triton kernel |
| `scripts/x_prof.py` | per-kernel GPU time census |
| `scripts/x_cpu_bits.py` | CPU bitwise control (N017's, retargeted at `SALAD_OPT_PAIRVEC`) |
| `scripts/x_eager_arms.py` | eager arm-vs-arm diff on the parity fixtures |
| `run_*.sbatch` | one job per measurement; placement stated in each |
| `logs/` | every job's stdout, plus the raw ncu tables |

## Jobs

| job | what | node |
|---|---|---|
| 907096 | traced graph + baseline timing | h100 x1 |
| 907097 | ncu, L2 total + read/write split | h100 x1 |
| 907131 | ncu, + atomic/reduce sectors | h100 x1 |
| 907101 | inductor `output_code` dump | h100 x1 |
| 907140 | per-kernel time census | h100 x1 |
| 907141 | ncu under preset `fast` (TF32) | h100 x1 |
| 907152 | O1 A/B bench + census, L=100/200 | h100 x1 |
| 907154 | CPU bitwise control, L=64/100 | cpu, 48 cores |
| 907177/907188 | parity suite, both arms (907177 had no `fixtures/` symlink) | cpu, 48 cores |
| 907178 / 907217 | `test_design_quality --n 160 --preset fast`, O1 arm / baseline arm | h100 x1 |

Total well under 3 GPU-hours.

"""N016/R2 - aggregate `ncu --print-summary per-gpu` output over a whole forward.

ncu groups identical launches and prints min/max/AVERAGE per group plus an
`Invocations N` count. Summing the per-group averages (as a naive grep would)
undercounts by ~8x; the total is sum(Invocations * average).

    python parse_ncu.py logs/ncu_raw.txt
"""
import re
import sys

U = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
inv_re = re.compile(r"Invocations (\d+)\s*$")
met_re = re.compile(r"^\s+(\S+)\s+(byte|Kbyte|Mbyte|Gbyte)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
path = sys.argv[1] if len(sys.argv) > 1 else "logs/ncu_raw.txt"
cur, tot, groups, launches = None, {}, 0, 0
for line in open(path):
    m = inv_re.search(line)
    if m:
        cur = int(m.group(1)); groups += 1; launches += cur; continue
    m = met_re.match(line.rstrip("\n"))
    if m and cur:
        d = tot.setdefault(m.group(1), [0.0, 0.0, 0.0])
        for i in range(3):
            d[i] += cur * float(m.group(3 + i)) * U[m.group(2)]
print(f"groups={groups}  launches={launches}")
for k, (lo, hi, av) in sorted(tot.items()):
    print(f"{k:<26} TOTAL {av/2**20:9.2f} MiB  [min-sum {lo/2**20:8.2f}, "
          f"max-sum {hi/2**20:8.2f}]")
dr, l2 = tot["dram__bytes_read.sum"][2], tot["lts__t_bytes.sum"][2]
print(f"\nDRAM read {dr/2**20:.1f} MiB -> {dr/2.94e12*1e3:.4f} ms at 2.94 TB/s")
print(f"L2        {l2/2**20:.1f} MiB -> {l2/6.53e12*1e3:.4f} ms at 6.53 TB/s")
print(f"L2 hit rate = 1 - DRAM/L2 = {(1-dr/l2)*100:.1f}%   L2/DRAM = {l2/dr:.1f}x")

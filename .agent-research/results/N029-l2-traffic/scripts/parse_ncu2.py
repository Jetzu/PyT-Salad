"""N029 - weighted per-group parse of ncu --print-summary output, incl. sectors."""
import re, sys, collections
U = {"byte":1.0,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9,"sector":32.0,"Ksector":32e3,"Msector":32e6,"Gsector":32e9}
inv = re.compile(r"^\s*(\S.*?)\s+Invocations (\d+)\s*$")
met = re.compile(r"^\s+(\S+)\s+(byte|Kbyte|Mbyte|Gbyte|sector|Ksector|Msector|Gsector)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s*$")
rows, cur = [], None
for line in open(sys.argv[1]):
    m = inv.match(line.rstrip())
    if m:
        cur = {"name": m.group(1), "n": int(m.group(2))}; rows.append(cur); continue
    m = met.match(line.rstrip("\n"))
    if m and cur:
        cur[m.group(1)] = float(m.group(5).replace(",", "")) * U[m.group(2)]
def cls(n):
    l = n.lower()
    if "triton" in l: return "triton"
    if any(k in l for k in ("xmma","gemv","cutlass","splitk","cublas","gemm","internal::kernel")): return "cublas"
    if "memcpy" in l or "memset" in l: return "memcpy"
    return "other"
K = ("lts__t_bytes.sum","lts__t_sectors_op_read.sum","lts__t_sectors_op_write.sum","dram__bytes_read.sum")
tot = collections.Counter(); per = collections.defaultdict(collections.Counter); ln = collections.Counter()
for r in rows:
    c = cls(r["name"]); ln[c] += r["n"]
    for k in K:
        tot[k] += r["n"]*r.get(k,0); per[c][k] += r["n"]*r.get(k,0)
M = 2**20
print(f"groups={len(rows)} launches={sum(r['n'] for r in rows)}")
hdr = f"{'class':<9}{'launch':>7}{'L2 MiB':>10}{'L2rd':>9}{'L2wr':>9}{'DRAMrd':>9}"
print(hdr)
for c in list(per)+["TOTAL"]:
    d = tot if c == "TOTAL" else per[c]
    n = sum(ln.values()) if c == "TOTAL" else ln[c]
    print(f"{c:<9}{n:7d}{d['lts__t_bytes.sum']/M:10.1f}"
          f"{d['lts__t_sectors_op_read.sum']/M:9.1f}{d['lts__t_sectors_op_write.sum']/M:9.1f}"
          f"{d['dram__bytes_read.sum']/M:9.1f}")
print("\n== TOP 40 kernel groups by L2 bytes ==")
rows.sort(key=lambda r: -r["n"]*r.get("lts__t_bytes.sum",0))
T = tot["lts__t_bytes.sum"]
print(f"{'MiB':>8}{'%':>5}{'inv':>5}{'MiB/ea':>8}{'rd':>8}{'wr':>8}  name")
for r in rows[:40]:
    b = r["n"]*r.get("lts__t_bytes.sum",0)
    print(f"{b/M:8.1f}{100*b/T:5.1f}{r['n']:5d}{r.get('lts__t_bytes.sum',0)/M:8.3f}"
          f"{r['n']*r.get('lts__t_sectors_op_read.sum',0)/M:8.1f}"
          f"{r['n']*r.get('lts__t_sectors_op_write.sum',0)/M:8.1f}  {r['name'][:92]}")

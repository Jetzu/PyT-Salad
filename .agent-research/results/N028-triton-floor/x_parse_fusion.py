"""N028: why do N Triton kernels carry exactly ONE scheduler node?

Inputs: the `x_fusion.py` dump for one arm (ir_pre/post_fusion, fusion.log,
output_code.py, fx_graph_readable.py, census.json).

`ir_post_fusion.txt` prints a fused group's header (`opA_opB_...:
FusedSchedulerNode(...)`) AND each of its members again as a bare
`opN: SchedulerNode(...)` line, so a member is only distinguishable from a real
single by whether its name appears in some group header. Getting that wrong
inflates the single count from 215 to 729 - the reconciliation to check is
`fused groups + singles == the profiler's triton launch count`.
"""
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

D = sys.argv[1]
ARM = os.path.basename(D.rstrip("/"))


def rd(n):
    p = os.path.join(D, n + ".gz")
    return (gzip.open(p, "rt", errors="replace") if os.path.exists(p)
            else open(os.path.join(D, n), errors="replace")).read()


post = rd("ir_post_fusion.txt")
hdr = re.findall(r"^([A-Za-z0-9_]+): FusedSchedulerNode\((.*?)\)$", post, re.M)
members, sizes = set(), []
for name, _ in hdr:
    ms = name.split("_")
    members.update(ms)
    sizes.append(len(ms))
allsn = set(re.findall(r"^(op[0-9]+): SchedulerNode\(", post, re.M))
extern = set(re.findall(r"^(op[0-9]+): ExternKernelSchedulerNode", post, re.M))
single = allsn - members
pre = set(re.findall(r"^(op[0-9]+): SchedulerNode\(",
                     rd("ir_pre_fusion.txt"), re.M))
print(f"[{ARM}] {len(pre)} pointwise/reduction scheduler nodes -> "
      f"{len(hdr)} fused groups (mean {sum(sizes)/max(len(hdr),1):.2f} nodes) + "
      f"{len(single)} SINGLE nodes = {len(hdr)+len(single)} triton kernels, "
      f"plus {len(extern)} extern (cuBLAS) nodes")
print(f"[{ARM}] {len(single)/(len(hdr)+len(single))*100:.0f} % of triton "
      f"kernels carry exactly one node")

# ------------------------------------------------------------- fusion log
WHY = re.compile(r"cannot fuse (op[0-9_]+) with (op[0-9_]+): (.*?)\s*$")
reasons = Counter()
per_node = defaultdict(set)
n = 0
for line in rd("fusion.log").splitlines():
    m = WHY.search(line)
    if not m:
        continue
    n += 1
    a, b, why = m.groups()
    key = re.sub(r"\(.*", "(...)", re.sub(r"node[12]", "nodeX", why)).strip()
    reasons[key] += 1
    for nd in (a, b):
        for part in nd.split("_"):
            per_node[part].add(key)
print(f"\n[{ARM}] {n} `cannot fuse` decisions logged, "
      f"{len(reasons)} distinct reasons")
for k, v in reasons.most_common():
    print(f"{v:>8}  {k}")

print(f"\n[{ARM}] the {len(single)} SINGLE nodes, by the reason set that "
      f"blocked them (a node hits several)")
cnt = Counter()
only_extern = set()
never = set()
for nd in single:
    rs = per_node.get(nd)
    if not rs:
        never.add(nd)
        continue
    for k in rs:
        cnt[k] += 1
    if rs <= {"nodeX is extern or nop"}:
        only_extern.add(nd)
for k, v in cnt.most_common():
    print(f"{v:>8}  {k}   ({v/len(single)*100:.0f} % of single nodes)")
print(f"{len(never):>8}  (no candidate partner was ever scored at all)")
print(f"{len(only_extern):>8}  blocked ONLY by `is extern or nop` - i.e. their "
      f"every neighbour is a cuBLAS call")

# ------------------------------------------------------------- kernels
KDEF = re.compile(r"^(triton_\w+?_(\d+))\s*=\s*async_compile\.triton")
kern, block = {}, []
for line in rd("output_code.py").splitlines():
    m = KDEF.match(line)
    if m:
        info = {"aten": [], "src": []}
        for bl in block:
            mm = re.match(r"#\s*Topologically Sorted Source Nodes:\s*\[(.*?)\],"
                          r"\s*Original ATen:\s*\[(.*?)\]", bl)
            if mm:
                info["src"] = [s.strip() for s in mm.group(1).split(",") if s.strip()]
                info["aten"] = [s.strip() for s in mm.group(2).split(",") if s.strip()]
        kern[m.group(1)] = info
        block = []
    elif line.startswith("#"):
        block.append(line)
    elif line.strip():
        block = []

cen = json.loads(rd("census.json"))
prof = {r["name"]: r for r in cen["rows"]}
tri = {k: v for k, v in prof.items() if k.startswith("triton")}
print(f"\n[{ARM}] profiled captured replay: "
      f"{sum(r['launches'] for r in prof.values()):.0f} kernels, "
      f"{sum(r['launches'] for r in tri.values()):.0f} triton launches "
      f"({sum(r['us'] for r in tri.values()):.0f} us), "
      f"{len(tri)} distinct triton kernels")

one = {k: v for k, v in kern.items() if len(v["aten"]) == 1}
by_op = defaultdict(lambda: [0.0, 0.0, 0])
tot = [0.0, 0.0]
for k, v in one.items():
    r = tri.get(k)
    if not r:
        continue
    tot[0] += r["launches"]; tot[1] += r["us"]
    e = by_op[v["aten"][0]]
    e[0] += r["launches"]; e[1] += r["us"]; e[2] += 1
print(f"\n[{ARM}] triton kernels whose whole body is ONE aten op: "
      f"{len(one)} distinct, {tot[0]:.0f} launches, {tot[1]:.0f} us "
      f"({tot[1]/sum(r['us'] for r in tri.values())*100:.0f} % of triton time)")
print(f"{'launches':>9} {'us':>8} {'us/l':>6} {'distinct':>9}  aten op")
for op, (nl, us, d) in sorted(by_op.items(), key=lambda kv: -kv[1][1]):
    print(f"{nl:>9.0f} {us:>8.1f} {us/max(nl,1):>6.2f} {d:>9}  {op}")

# the 20 most expensive triton kernels overall, with their aten content
print(f"\n[{ARM}] 20 most expensive triton kernels")
print(f"{'launches':>9} {'us':>8} {'us/l':>6}  kernel / aten ops")
for k, r in sorted(tri.items(), key=lambda kv: -kv[1]["us"])[:20]:
    a = ",".join(kern.get(k, {}).get("aten", []))[:90]
    print(f"{r['launches']:>9.0f} {r['us']:>8.1f} {r['us']/r['launches']:>6.2f}  "
          f"{k[:52]}\n{'':>26}{a}")

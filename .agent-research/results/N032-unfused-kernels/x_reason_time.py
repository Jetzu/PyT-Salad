"""N032: extend N028's x_parse_fusion.py with a reason -> TIME table.

N028 already gets reason -> COUNT of single scheduler nodes hit by that
reason. This adds reason -> US by resolving each single node opN to the
kernel that actually executes it (via its output buffer bufN, walking
inductor's buffer-reuse aliasing in `call()`), then reading that kernel's
per-launch time from the captured-replay census.

Usage: python x_reason_time.py <dump_dir>
  dump_dir must contain (gzipped or not): ir_pre_fusion.txt, ir_post_fusion.txt,
  fusion.log, output_code.py, census.json
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
pre = rd("ir_pre_fusion.txt")
hdr = re.findall(r"^([A-Za-z0-9_]+): FusedSchedulerNode\((.*?)\)$", post, re.M)
members = set()
for name, _ in hdr:
    members.update(name.split("_"))
allsn = set(re.findall(r"^(op[0-9]+): SchedulerNode\(", post, re.M))
extern = set(re.findall(r"^(op[0-9]+): ExternKernelSchedulerNode", post, re.M))
single = allsn - members
pre_nodes = set(re.findall(r"^(op[0-9]+): SchedulerNode\(", pre, re.M))

print(f"[{ARM}] {len(pre_nodes)} pointwise/reduction nodes -> {len(hdr)} fused "
      f"groups + {len(single)} singles = {len(hdr)+len(single)} triton kernels, "
      f"{len(extern)} extern")

# ---- op N -> its FIRST written buffer name (bufN in the overwhelming case,
# verified: op10 writes buf10, op17->buf17, op23->buf23, op30->buf30 in this
# codebase's numbering - inductor names ComputedBuffers after the op that
# creates them). Parse `opN.writes = [MemoryDep('bufX', ...` to get the REAL
# name rather than assume opN<->bufN, since fused headers write >1 buffer.
op_writes = {}
for m in re.finditer(r"^(op[0-9]+)\.writes = \[(.*?)\]\s*$", post, re.M):
    op, body = m.groups()
    bufs = re.findall(r"MemoryDep\('([a-z0-9_]+)'", body)
    if bufs:
        op_writes[op] = bufs[0]  # single nodes write exactly one buffer

# ---- reason log ---------------------------------------------------------
WHY = re.compile(r"cannot fuse (op[0-9_]+) with (op[0-9_]+): (.*?)\s*$")
reasons = Counter()
per_node = defaultdict(set)
for line in rd("fusion.log").splitlines():
    m = WHY.search(line)
    if not m:
        continue
    a, b, why = m.groups()
    key = re.sub(r"\(.*", "(...)", re.sub(r"node[12]", "nodeX", why)).strip()
    reasons[key] += 1
    for nd in (a, b):
        for part in nd.split("_"):
            per_node[part].add(key)

# ---- output_code.py: kernel defs (aten ops) + call() buffer aliasing ----
code = rd("output_code.py")
KDEF = re.compile(r"^(triton_\w+?_(\d+))\s*=\s*async_compile\.triton")
kern_aten = {}
block = []
for line in code.splitlines():
    m = KDEF.match(line)
    if m:
        aten = []
        for bl in block:
            mm = re.match(r"#\s*Topologically Sorted Source Nodes:\s*\[(.*?)\],"
                          r"\s*Original ATen:\s*\[(.*?)\]", bl)
            if mm:
                aten = [s.strip() for s in mm.group(2).split(",") if s.strip()]
        kern_aten[m.group(1)] = aten
        block = []
    elif line.startswith("#"):
        block.append(line)
    elif line.strip():
        block = []

call_body = re.search(r"^def call\(args\):(.*?)^def ", code, re.M | re.S)
call_body = call_body.group(1) if call_body else code
call_lines = call_body.splitlines()

# union-find over buffer aliases: "bufX = bufY[; del bufY]  # reuse"
parent = {}


def find(x):
    while parent.get(x, x) != x:
        parent[x] = parent.get(parent[x], parent[x])
        x = parent[x]
    return x


ALIAS = re.compile(r"^\s*(buf\w+)\s*=\s*(buf\w+)\s*;.*#\s*reuse")
for ln in call_lines:
    m = ALIAS.match(ln)
    if m:
        a, b = m.groups()
        parent[a] = find(b)

# first kernel.run(...) occurrence (by line index) mentioning a given buffer
# name (after alias resolution to the earliest-created storage name is not
# needed - we search for the buffer's OWN name, since inductor reuses the
# python variable name at each rebind, and the write always happens under the
# CURRENT name).
RUN = re.compile(r"^\s*(triton_\w+)\.run\((.*)\)")
buf_first_run = {}
for i, ln in enumerate(call_lines):
    m = RUN.match(ln)
    if not m:
        continue
    kname, argstr = m.groups()
    args = re.findall(r"\bbuf\w+\b", argstr)
    for a in args:
        if a not in buf_first_run:
            buf_first_run[a] = kname

# ---- per-kernel census (launches, us) -----------------------------------
cen = json.loads(rd("census.json"))
prof = {r["name"]: r for r in cen["rows"]}
tri = {k: v for k, v in prof.items() if k.startswith("triton")}
tri_us_per_launch = {k: v["us"] / v["launches"] for k, v in tri.items() if v["launches"] > 0}
total_triton_us = sum(v["us"] for v in tri.values())
total_triton_launches = sum(v["launches"] for v in tri.values())

# ---- resolve each single opN -> kernel -> per-launch us ------------------
resolved, unresolved = 0, 0
node_us = {}
node_kernel = {}
for nd in single:
    buf = op_writes.get(nd)
    if buf is None:
        unresolved += 1
        continue
    kname = buf_first_run.get(buf)
    if kname is None:
        unresolved += 1
        continue
    us = tri_us_per_launch.get(kname)
    if us is None:
        unresolved += 1
        continue
    node_us[nd] = us
    node_kernel[nd] = kname
    resolved += 1

print(f"[{ARM}] resolved {resolved}/{len(single)} single nodes to a kernel + "
      f"per-launch us ({unresolved} unresolved)")

# ---- Table 1: overlapping reason -> count / time (matches N028 style) ---
print(f"\n[{ARM}] reason -> (node count, node count with resolved time, "
      f"total us [overlapping - a node can hit several reasons])")
cnt = Counter()
time_by_reason = defaultdict(float)
ncount_by_reason = Counter()
never = set()
for nd in single:
    rs = per_node.get(nd)
    if not rs:
        never.add(nd)
        continue
    for k in rs:
        cnt[k] += 1
        if nd in node_us:
            time_by_reason[k] += node_us[nd]
            ncount_by_reason[k] += 1
rows = []
for k, v in cnt.items():
    rows.append((time_by_reason.get(k, 0.0), v, ncount_by_reason.get(k, 0), k))
rows.sort(reverse=True)
for us, v, nres, k in rows:
    print(f"{us:>9.1f} us  {v:>4} nodes ({nres:>4} w/ resolved time)  {k}")
print(f"{sum(node_us.values()) if node_us else 0:>9.1f} us  total single-node "
      f"time resolved (nodes are double counted above; this line is not)")
print(f"{len(never):>4}  singles with NO logged reason at all")

# ---- Table 2: partitioned - ONE primary reason per node, ranked by time -
PRIORITY = [
    "tiling mismatch (...)",
    "nodes numel incompatibility",
    "numel/rnumel mismatch (...)",
    "memory deps did not match",
    "no shared data",
    "will increase peak memory",
    "intermediate nodes between nodeX & nodeX",
    "nodeX is extern or nop",
]
primary_time = defaultdict(float)
primary_count = Counter()
for nd in single:
    rs = per_node.get(nd, set())
    us = node_us.get(nd, 0.0)
    chosen = next((p for p in PRIORITY if p in rs), None)
    if chosen is None:
        chosen = "(no reason logged)" if not rs else next(iter(rs))
    primary_time[chosen] += us
    primary_count[chosen] += 1

print(f"\n[{ARM}] PARTITIONED (one primary reason/node, priority-ordered so "
      f"totals don't overlap), ranked by TIME:")
tot_us = sum(primary_time.values())
tot_n = sum(primary_count.values())
for k, us in sorted(primary_time.items(), key=lambda kv: -kv[1]):
    n = primary_count[k]
    print(f"{us:>9.1f} us ({us/max(tot_us,1e-9)*100:>5.1f}%)  {n:>4} nodes  {k}")
print(f"{tot_us:>9.1f} us total (resolved), {tot_n} nodes, "
      f"{total_triton_us:.1f} us = all triton time, "
      f"{total_triton_launches:.0f} triton launches")

# ---- Table 3: by innermost aten-op signature of the RESOLVED kernel -----
print(f"\n[{ARM}] single nodes by their kernel's Original-ATen signature "
      f"(what kind of op is stuck unfused):")
by_sig = defaultdict(lambda: [0.0, 0])
for nd in single:
    kname = node_kernel.get(nd)
    us = node_us.get(nd, 0.0)
    sig = ",".join(kern_aten.get(kname, ["?"])) if kname else "?(unresolved)"
    by_sig[sig][0] += us
    by_sig[sig][1] += 1
for sig, (us, n) in sorted(by_sig.items(), key=lambda kv: -kv[1][0])[:25]:
    print(f"{us:>9.1f} us  {n:>4} nodes  {sig[:100]}")

# ---- Table 4: cross-tab - for the top signatures, which PRIMARY reason? ---
print(f"\n[{ARM}] cross-tab: top signatures x their primary (partitioned) reason")
sig_reason = defaultdict(Counter)
sig_us = defaultdict(float)
for nd in single:
    kname = node_kernel.get(nd)
    sig = ",".join(kern_aten.get(kname, ["?"])) if kname else "?(unresolved)"
    rs = per_node.get(nd, set())
    chosen = next((p for p in PRIORITY if p in rs), None)
    if chosen is None:
        chosen = "(no reason logged)" if not rs else next(iter(rs))
    sig_reason[sig][chosen] += 1
    sig_us[sig] += node_us.get(nd, 0.0)
for sig, _ in sorted(sig_us.items(), key=lambda kv: -kv[1])[:12]:
    reasons_str = ", ".join(f"{k}x{v}" for k, v in sig_reason[sig].most_common())
    print(f"{sig_us[sig]:>9.1f} us  {sig[:60]:<60}  {reasons_str}")

json.dump(dict(single_count=len(single), fused_groups=len(hdr),
                resolved=resolved, unresolved=unresolved,
                total_triton_us=total_triton_us,
                total_triton_launches=total_triton_launches,
                overlapping=[{"reason": k, "nodes": v,
                              "resolved_nodes": ncount_by_reason.get(k, 0),
                              "us": time_by_reason.get(k, 0.0)}
                             for us, v, nres, k in rows],
                partitioned=[{"reason": k, "nodes": primary_count[k], "us": us}
                             for k, us in sorted(primary_time.items(),
                                                  key=lambda kv: -kv[1])],
                by_signature=[{"sig": sig, "us": us, "nodes": n}
                              for sig, (us, n) in sorted(
                                  by_sig.items(), key=lambda kv: -kv[1][0])]),
          open(os.path.join(D, f"reason_time_{ARM}.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(D, f'reason_time_{ARM}.json')}")

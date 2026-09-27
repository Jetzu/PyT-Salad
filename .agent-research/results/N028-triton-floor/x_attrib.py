"""N028: attribute the expensive single-op triton kernels to SOURCE.

kernel -> the post-grad node names in its `# Graph fragment:` block ->
`fx_graph_readable.py`'s `# File:` frames (innermost, and the nearest preceding
CALL-SITE frame, because the innermost is usually a leaf helper).
"""
import gzip
import json
import os
import re
import sys
from collections import defaultdict

D = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 25


def rd(n):
    p = os.path.join(D, n + ".gz")
    return (gzip.open(p, "rt", errors="replace") if os.path.exists(p)
            else open(os.path.join(D, n), errors="replace")).read()


ANCHOR = {"diffusion.py", "blocks.py", "attention.py", "cond_prologue.py",
          "decoders.py", "model.py", "sidechain.py"}
FILE_RE = re.compile(r"\s*#\s*File:\s*(\S+?):(\d+)\s+in\s+(\S+),\s*code:\s*(.*)")
NODE_RE = re.compile(r"\s*(\w+)\s*:\s*\"[^\"]*\"\s*=\s*")
nodes = {}
inner = anchor = None
for line in rd("fx_graph_readable.py").splitlines():
    m = FILE_RE.match(line)
    if m:
        f = os.path.basename(m.group(1))
        inner = (f, int(m.group(2)), m.group(3), m.group(4).strip())
        if f in ANCHOR:
            anchor = inner
        continue
    m = NODE_RE.match(line)
    if m:
        nodes[m.group(1)] = (inner, anchor)

KDEF = re.compile(r"^(triton_\w+?_(\d+))\s*=\s*async_compile\.triton")
kern, block = {}, []
for line in rd("output_code.py").splitlines():
    m = KDEF.match(line)
    if m:
        info = {"aten": [], "nodes": []}
        for bl in block:
            mm = re.match(r"#\s*Topologically Sorted Source Nodes:\s*\[(.*?)\],"
                          r"\s*Original ATen:\s*\[(.*?)\]", bl)
            if mm:
                info["aten"] = [s.strip() for s in mm.group(2).split(",") if s.strip()]
            mm = re.match(r"#\s*%(\w+)\s*:", bl)
            if mm:
                info["nodes"].append(mm.group(1))
        kern[m.group(1)] = info
        block = []
    elif line.startswith("#"):
        block.append(line)
    elif line.strip():
        block = []

cen = json.loads(rd("census.json"))
tri = {r["name"]: r for r in cen["rows"] if r["name"].startswith("triton")}


def src(k):
    for nd in kern.get(k, {}).get("nodes", []):
        if nd in nodes and nodes[nd][0]:
            i, a = nodes[nd]
            return (f"{i[0]}:{i[1]} in {i[2]}  |  {i[3][:70]}",
                    (f"{a[0]}:{a[1]}" if a else "?"))
    return ("?", "?")


rows = sorted(tri.items(), key=lambda kv: -kv[1]["us"])
print(f"{'launches':>8} {'us':>7} {'us/l':>6}  aten | source")
agg = defaultdict(lambda: [0.0, 0.0])
for k, r in rows[:TOPN]:
    a = ",".join(kern.get(k, {}).get("aten", []))
    s, anc = src(k)
    single = " [1-OP]" if len(kern.get(k, {}).get("aten", [])) == 1 else ""
    print(f"{r['launches']:>8.0f} {r['us']:>7.1f} {r['us']/r['launches']:>6.2f}  "
          f"{a[:60]}{single}\n{'':>25}{s}   <- {anc}")
for k, r in rows:
    s, _ = src(k)
    e = agg[s]
    e[0] += r["launches"]; e[1] += r["us"]
print("\nby innermost source line (top 20):")
print(f"{'launches':>8} {'us':>7}  source")
for s, (n, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"{n:>8.0f} {us:>7.1f}  {s}")

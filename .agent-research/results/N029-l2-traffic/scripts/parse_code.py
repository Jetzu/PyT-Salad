"""Parse inductor g2.py: per-buffer / per-kernel materialised traffic."""
import re, sys, collections, math

src = open("g2.py").read()
lines = src.splitlines()

DT = {"torch.float32":4,"torch.float64":8,"torch.int64":8,"torch.int32":4,
      "torch.int16":2,"torch.bool":1,"torch.uint8":1,"torch.float16":2,"torch.bfloat16":2}

# --- kernel signatures -------------------------------------------------
sig = {}
for i,l in enumerate(lines):
    m = re.match(r"def (triton_[a-zA-Z0-9_]+)\((.*)\):", l)
    if m:
        sig[m.group(1)] = [a.strip() for a in m.group(2).split(",")]
# also the @triton.jit wrapper name used at the call site is <name> (same)

# --- buffer sizes ------------------------------------------------------
call_start = next(i for i,l in enumerate(lines) if l.startswith("def call("))
body = lines[call_start:]
bufsize = {}   # name -> bytes
bufshape = {}
for l in body:
    m = re.match(r"\s*(\w+) = empty_strided_cuda\(\((.*?)\), \((.*?)\), (torch\.\w+)\)", l)
    if m:
        shp = [int(x) for x in re.findall(r"\d+", m.group(2))]
        n = 1
        for s in shp: n *= s
        bufsize[m.group(1)] = n*DT[m.group(4)]
        bufshape[m.group(1)] = (tuple(shp), m.group(4))
    m = re.match(r"\s*assert_size_stride\((\w+), \((.*?)\), ", l)
    if m:
        shp = [int(x) for x in re.findall(r"\d+", m.group(2))]
        n = 1
        for s in shp: n *= s
        bufsize.setdefault(m.group(1), n*4)   # params/inputs assumed fp32
        bufshape.setdefault(m.group(1), (tuple(shp), "param"))
    m = re.match(r"\s*(\w+) = (\w+); del \w+  # reuse", l)
    if m and m.group(2) in bufsize:
        bufsize[m.group(1)] = bufsize[m.group(2)]
        bufshape[m.group(1)] = bufshape[m.group(2)]
    m = re.match(r"\s*(\w+) = reinterpret_tensor\((\w+), \((.*?)\),", l)
    if m:
        shp = [int(x) for x in re.findall(r"\d+", m.group(3))]
        n = 1
        for s in shp: n *= s
        bufsize[m.group(1)] = n*4
        bufshape[m.group(1)] = (tuple(shp), "view:"+m.group(2))

def base(nm):
    return nm

read = collections.Counter(); write = collections.Counter()
kern_traffic = collections.Counter(); kern_n = collections.Counter()
kern_kind = {}
unknown = collections.Counter()

def argbytes(a):
    a = a.strip()
    m = re.match(r"reinterpret_tensor\((\w+), \((.*?)\)", a)
    if m:
        shp = [int(x) for x in re.findall(r"\d+", m.group(2))]
        n = 1
        for s in shp: n *= s
        return n*4, m.group(1)
    if a in bufsize:
        return bufsize[a], a
    unknown[a] += 1
    return 0, a

def split_args(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([": depth += 1
        if ch in ")]": depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip(): out.append(cur)
    return [x.strip() for x in out]

for l in body:
    m = re.match(r"\s*(triton_\w+)\.run\((.*), grid=", l)
    if m:
        name, args = m.group(1), split_args(m.group(2))
        s = sig.get(name)
        if s is None:
            continue
        ptrs = [a for a in s if a.endswith(("_ptr0","_ptr1","_ptr2","_ptr3","_ptr4","_ptr5",
                 "_ptr6","_ptr7","_ptr8","_ptr9","_ptr10","_ptr11","_ptr12","_ptr13","_ptr14",
                 "_ptr15","_ptr16","_ptr17","_ptr18","_ptr19","_ptr20"))]
        tot = 0
        for i,p in enumerate(ptrs):
            if i >= len(args): break
            b, nm = argbytes(args[i])
            if p.startswith("in_out"):
                read[nm]+=1; write[nm]+=1; tot += 2*b
            elif p.startswith("in_"):
                read[nm]+=1; tot += b
            else:
                write[nm]+=1; tot += b
        kern_traffic[name] += tot; kern_n[name] += 1; kern_kind[name]="triton"
        continue
    m = re.match(r"\s*extern_kernels\.(bmm|mm|addmm)\((.*)\)\s*$", l)
    if m:
        kind, args = m.group(1), split_args(m.group(2))
        tot = 0
        for a in args:
            if a.startswith("out="): a = a[4:]
            b, nm = argbytes(a)
            tot += b
            if "out=" in l and a == split_args(m.group(2))[-1][4:]:
                write[nm]+=1
            else:
                read[nm]+=1
        kern_traffic["extern_"+kind] += tot; kern_n["extern_"+kind] += 1
        kern_kind["extern_"+kind]="extern"

tot_tri = sum(v for k,v in kern_traffic.items() if kern_kind.get(k)=="triton")
tot_ext = sum(v for k,v in kern_traffic.items() if kern_kind.get(k)=="extern")
print(f"TRITON kernels {sum(v for k,v in kern_n.items() if kern_kind.get(k)=='triton')} launches, "
      f"{len([k for k in kern_n if kern_kind.get(k)=='triton'])} distinct, "
      f"IDEAL traffic {tot_tri/2**20:.1f} MiB")
print(f"EXTERN gemm   {sum(v for k,v in kern_n.items() if kern_kind.get(k)=='extern')} launches, "
      f"IDEAL (operands once) {tot_ext/2**20:.1f} MiB")
print(f"unknown args: {sum(unknown.values())} -> {unknown.most_common(5)}")

print("\n== TOP 30 TRITON KERNELS by ideal traffic (bytes of buffers they touch) ==")
for k,v in sorted(((k,v) for k,v in kern_traffic.items() if kern_kind.get(k)=="triton"),
                  key=lambda x:-x[1])[:30]:
    print(f"{v/2**20:9.2f} MiB  n={kern_n[k]:3d}  {v/kern_n[k]/2**20:7.3f} MiB/ea  {k[:95]}")

print("\n== BUFFERS by total traffic (size x (writes+reads)) ==")
rows=[]
for b in set(list(read)+list(write)):
    if b not in bufsize: continue
    rows.append((bufsize[b]*(read[b]+write[b]), b, read[b], write[b], bufsize[b]))
rows.sort(reverse=True)
tot_buf = sum(r[0] for r in rows)
print(f"total buffer traffic {tot_buf/2**20:.1f} MiB over {len(rows)} buffers")
for t,b,r,w,s in rows[:30]:
    print(f"{t/2**20:9.2f} MiB  {b:>9} size {s/2**20:7.3f} MiB  r={r:3d} w={w:3d}  {bufshape.get(b)}")

# aggregate by buffer SHAPE class
cls = collections.Counter(); clsn = collections.Counter()
for t,b,r,w,s in rows:
    sh = bufshape.get(b, ("?","?"))
    key = str(sh[0])
    cls[key]+=t; clsn[key]+=1
print("\n== traffic by buffer SHAPE ==")
for k,v in cls.most_common(25):
    print(f"{v/2**20:9.2f} MiB  {clsn[k]:4d} buffers  shape {k}")

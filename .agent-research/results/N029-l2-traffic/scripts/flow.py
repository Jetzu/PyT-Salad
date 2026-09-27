import re, collections, sys
sys.argv=[0]
exec(open("parse_code.py").read().split("tot_tri =")[0])
# rebuild with producer/consumer kind per buffer
g2 = open("g2.py").read().splitlines()
prod = {}; cons = collections.defaultdict(list)
for l in g2:
    m = re.match(r"\s*(triton_\w+)\.run\((.*), grid=", l)
    if m:
        name, args = m.group(1), split_args(m.group(2))
        s = sig.get(name)
        if not s: continue
        ptrs=[a for a in s if re.match(r"(in_out_ptr|in_ptr|out_ptr)\d+$", a)]
        for i,p in enumerate(ptrs):
            if i>=len(args): break
            b,nm = argbytes(args[i])
            if p.startswith("in_out"): prod.setdefault(nm,("triton",name)); cons[nm].append(("triton",name))
            elif p.startswith("in_"): cons[nm].append(("triton",name))
            else: prod[nm]=("triton",name)
        continue
    m = re.match(r"\s*extern_kernels\.(bmm|mm|addmm)\((.*)\)\s*$", l)
    if m:
        args = split_args(m.group(2))
        for a in args:
            out = a.startswith("out=")
            if out: a = a[4:]
            b,nm = argbytes(a)
            if out: prod[nm]=("gemm",m.group(1))
            else: cons[nm].append(("gemm",m.group(1)))
def root(nm):
    sh = bufshape.get(nm)
    while sh and isinstance(sh[1],str) and sh[1].startswith("view:"):
        nm = sh[1][5:]; sh = bufshape.get(nm)
    return nm
flow = collections.Counter(); fn = collections.Counter()
detail = collections.defaultdict(list)
for b, cs in cons.items():
    p = prod.get(b)
    if p is None:
        pk = "param/input"
    else:
        pk = p[0]
    for c in cs:
        key = f"{pk} -> {c[0]}"
        flow[key] += bufsize.get(b,0); fn[key]+=1
        if pk=="triton" and c[0]=="triton":
            detail[(prod[b][1], c[1])].append(bufsize.get(b,0))
M=2**20
print("LOGICAL read traffic by producer->consumer:")
for k,v in flow.most_common():
    print(f"  {k:<24} {v/M:9.1f} MiB  over {fn[k]:5d} edges")
print(f"\ntriton->triton edges (the remaining FUSION prize): "
      f"{flow['triton -> triton']/M:.1f} MiB read + the same again as writes")
rows=sorted(((sum(v),len(v),k) for k,v in detail.items()), reverse=True)
print("\ntop triton->triton producer/consumer pairs:")
for t,n,k in rows[:20]:
    print(f"  {t/M:8.2f} MiB n={n:3d}  {k[0][:46]:<46} -> {k[1][:46]}")

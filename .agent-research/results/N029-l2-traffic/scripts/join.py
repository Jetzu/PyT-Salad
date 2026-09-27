import re, collections, sys
sys.argv=[0]
exec(open("parse_code.py").read().split("tot_tri =")[0])
# source comment per kernel launch
g2 = open("g2.py").read().splitlines()
ksrc = {}
last_comment = ""
for l in g2:
    if "Original ATen:" in l:
        last_comment = l.strip()
    m = re.match(r"\s*(triton_\w+)\.run\(", l)
    if m and m.group(1) not in ksrc:
        nodes = re.search(r"Source Nodes: \[(.*?)\], Original", last_comment)
        ksrc[m.group(1)] = (nodes.group(1)[:60] if nodes else "")
# ncu measured
U={"byte":1.0,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9,"sector":32.0}
inv=re.compile(r"^\s*(\S.*?)\s+Invocations (\d+)\s*$")
met=re.compile(r"^\s+(\S+)\s+(byte|Kbyte|Mbyte|Gbyte|sector)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s*$")
meas=collections.Counter(); measn=collections.Counter(); mrd=collections.Counter(); mwr=collections.Counter()
cur=None
P="/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/ncu_raw_907097.txt"
for line in open(P):
    m=inv.match(line.rstrip())
    if m:
        nm=m.group(1).split(" (")[0].strip(); cur=(nm,int(m.group(2)))
        measn[nm]+=cur[1]; continue
    m=met.match(line.rstrip("\n"))
    if m and cur:
        v=float(m.group(5).replace(",",""))*U[m.group(2)]*cur[1]
        if m.group(1)=="lts__t_bytes.sum": meas[cur[0]]+=v
        elif m.group(1)=="lts__t_sectors_op_read.sum": mrd[cur[0]]+=v
        elif m.group(1)=="lts__t_sectors_op_write.sum": mwr[cur[0]]+=v
M=2**20
rows=[]
for k,v in kern_traffic.items():
    if kern_kind.get(k)!="triton": continue
    rows.append((meas.get(k,0), v, kern_n[k], measn.get(k,0), mrd.get(k,0), mwr.get(k,0), k))
rows.sort(reverse=True)
print(f"{'measMiB':>9}{'idealMiB':>9}{'amp':>6}{'n':>4}{'rd':>8}{'wr':>8}  kernel / source nodes")
tm=ti=0
for me,id_,n,nn,rd,wr,k in rows[:32]:
    tm+=me; ti+=id_
    print(f"{me/M:9.1f}{id_/M:9.1f}{(me/id_ if id_ else 0):6.2f}{n:4d}{rd/M:8.1f}{wr/M:8.1f}  {k[:52]:<52} {ksrc.get(k,'')[:45]}")
TM=sum(r[0] for r in rows); TI=sum(r[1] for r in rows)
print(f"\nALL TRITON: measured {TM/M:.1f} MiB, ideal {TI/M:.1f} MiB, amplification {TM/TI:.2f}x")
# unmatched
um=[k for k in meas if "triton" in k and k not in kern_traffic]
print(f"ncu triton groups not in g2: {len(um)} -> {sum(meas[k] for k in um)/M:.1f} MiB")

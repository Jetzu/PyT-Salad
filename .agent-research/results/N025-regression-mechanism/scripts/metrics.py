import os, sys, numpy as np
def ca(p):
    v=[]
    for l in open(p):
        if l.startswith("ATOM") and l[12:16].strip()=="CA":
            v.append((float(l[30:38]),float(l[38:46]),float(l[46:54])))
    return np.asarray(v,dtype=np.float64)
for d in sys.argv[1:]:
    rows=[]
    for f in sorted(os.listdir(d)):
        if not f.endswith(".pdb"): continue
        c=ca(os.path.join(d,f))
        b=np.linalg.norm(np.diff(c,axis=0),axis=1)
        rg=float(np.sqrt(((c-c.mean(0))**2).sum(1).mean()))
        cc=float(b.mean()); nan=bool(np.isnan(c).any())
        rows.append((f,cc,rg,nan or cc>4.5 or cc<3.0 or rg>30.0))
    div=[r for r in rows if r[3]]; ok=[r for r in rows if not r[3]]
    print(f"{os.path.basename(d):28s} n={len(rows):3d} diverged={len(div):3d} "
          f"({len(div)/max(1,len(rows)):.1%})  converged CA-CA="
          f"{np.mean([r[1] for r in ok]) if ok else float('nan'):.4f} "
          f"Rg={np.mean([r[2] for r in ok]) if ok else float('nan'):.2f}")
    for r in div[:6]: print(f"    DIVERGED {r[0]} CA-CA {r[1]:.2f} Rg {r[2]:.2f}")

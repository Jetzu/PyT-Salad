import math, numpy as np
def read_ca(p):
    return np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])]
                     for l in open(p) if l.startswith("ATOM") and l[12:16].strip()=="CA"])
def kabsch(P,Q):
    Pm,Qm=P.mean(0),Q.mean(0); Pc,Qc=P-Pm,Q-Qm
    V,S,W=np.linalg.svd(Pc.T@Qc); d=np.sign(np.linalg.det(V@W))
    R=V@np.diag([1,1,d])@W
    rms=float(np.sqrt((((Pc@R)-Qc)**2).sum(1).mean()))
    return R,Qm-Pm@R,rms
for tag,n in [("c4_div",4),("screw_zero_r",4),("sym_mean",4),("c12_small",12),("c4_nondiv",4)]:
    ca=read_ca(f"/tmp/rt_gen/{tag}/design_0.pdb"); L=len(ca); size=L//n
    print(f"\n{tag}: L={L} n={n} size={size}")
    for i in range(1,n):
        R,t,rms=kabsch(ca[:size],ca[i*size:(i+1)*size])
        ang=math.degrees(math.acos(max(-1,min(1,(np.trace(R)-1)/2))))
        # rotation axis from the skew part
        ax=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
        ax=ax/ (np.linalg.norm(ax)+1e-12)
        print(f"  0->{i}: rmsd {rms:.4f} A  angle {ang:7.2f} deg (ideal {360*i/n:6.1f})  "
              f"axis {np.round(ax,3)}  |t| {np.linalg.norm(t):.2f}")

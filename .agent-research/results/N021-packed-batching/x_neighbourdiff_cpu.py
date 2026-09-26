import sys, torch
import os; sys.path.insert(0, os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n021tree"))
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.diffusion.sampling import make_initial_data, pack_designs
from salad_torch.diffusion.schedules import parse_num_aa
from salad_torch.modules.model import StructureDiffusionPredict, apply_diffusion
import salad_torch.geometry.neighbours as NB
torch.set_float32_matmul_precision("highest")
cfg = get_config("default_vp"); dev = torch.device("cpu")
model = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(model, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
L = int(sys.argv[1]); B = 2
num_aa, resi, chain, _, _ = parse_num_aa(str(L)); A = model.diffusion.atom_count
def mk(seed):
    d, p = make_initial_data(num_aa, resi, chain, None, device=dev)
    g = torch.Generator().manual_seed(seed)
    d["t_pos"] = torch.full((L,), 0.7); d["t_seq"] = torch.ones(L)
    d["pos"] = torch.randn(L, A, 3, generator=g)
    return d, p, torch.rand(L, L, generator=g).clamp_(1e-6,1-1e-6), torch.randn(L, A, 3, generator=g)
D = [mk(200+i) for i in range(B)]
# capture every get_neighbours result
cap = []
orig = NB.get_neighbours
def patched(count):
    inner = orig(count)
    def f(distance, mask, neighbours=None):
        r = inner(distance, mask, neighbours)
        cap.append((count, r))
        return r
    return f
NB.get_neighbours = patched
# neighbours.py functions captured `get_neighbours` at def time inside closures -> they look it up at call time (module global), so patch works
def run(idxs):
    cap.clear()
    ds=[D[i][0] for i in idxs]; ps=[D[i][1] for i in idxs]
    pd, pp, lens = pack_designs(ds, ps)
    N=L*len(idxs); U=torch.full((N,N),0.5)
    for j,i in enumerate(idxs): U[j*L:(j+1)*L, j*L:(j+1)*L]=D[i][2]
    nz=torch.cat([D[i][3] for i in idxs],0)
    pd=dict(pd); pd.update(apply_diffusion(cfg,pd,noise=nz,seq_noise=torch.full((N,),0.5)))
    with torch.no_grad(): r,_=model(pd,pp,uniform=U,compute_distogram=False)
    return r, list(cap)
r0, c0 = run([0])
r01, c01 = run([0,1])
print(f"L={L}: {len(c0)} get_neighbours calls alone, {len(c01)} packed")
bad=0
for k,((cnt0,n0),(cnt1,n1)) in enumerate(zip(c0,c01)):
    sub = n1[:L]
    # packed indices of member 0 are the same absolute indices (offset 0)
    neg0 = (n0==-1).sum().item(); neg1=(sub==-1).sum().item()
    same = torch.equal(n0, sub)
    if not same or neg0!=neg1:
        bad+=1
        if bad<=6:
            diff=(n0!=sub).sum().item()
            print(f"  call {k:3d} count={cnt0:3d} K={n0.shape[1]:3d} MISMATCH {diff} entries; -1 alone={neg0} packed={neg1}")
    elif k<3 or neg0:
        pass
print(f"  total mismatching neighbour calls: {bad}/{len(c0)}")
print(f"  -1 counts per call (alone): {[int((n==-1).sum()) for _,n in c0][:12]}")

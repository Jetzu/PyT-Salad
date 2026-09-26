import sys, torch
import os; sys.path.insert(0, os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n021tree"))
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.diffusion.sampling import make_initial_data, pack_designs
from salad_torch.diffusion.schedules import parse_num_aa
from salad_torch.modules.model import StructureDiffusionPredict, apply_diffusion
torch.set_float32_matmul_precision("highest")
cfg = get_config("default_vp"); dev = torch.device("cpu")
model = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(model, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
L = int(sys.argv[1]); B = 3
num_aa, resi, chain, _, _ = parse_num_aa(str(L)); A = model.diffusion.atom_count
def mk(seed):
    d, p = make_initial_data(num_aa, resi, chain, None, device=dev)
    g = torch.Generator().manual_seed(seed)
    d["t_pos"] = torch.full((L,), 0.7); d["t_seq"] = torch.ones(L)
    d["pos"] = torch.randn(L, A, 3, generator=g)
    return d, p, torch.rand(L, L, generator=g).clamp_(1e-6,1-1e-6), torch.randn(L, A, 3, generator=g)
D = [mk(200+i) for i in range(B)]
rec = []
def hook(name):
    def f(mod, inp, out):
        for j, t in enumerate(out if isinstance(out, (tuple, list)) else [out]):
            if torch.is_tensor(t): rec.append((name, j, t.detach().clone()))
    return f
for nm, m in model.named_modules():
    if len(list(m.children())) == 0: m.register_forward_hook(hook(nm))
def run(idxs):
    rec.clear()
    ds=[D[i][0] for i in idxs]; ps=[D[i][1] for i in idxs]
    pd, pp, lens = pack_designs(ds, ps)
    N=L*len(idxs); U=torch.full((N,N),0.5)
    for j,i in enumerate(idxs): U[j*L:(j+1)*L, j*L:(j+1)*L]=D[i][2]
    nz=torch.cat([D[i][3] for i in idxs],0)
    pd=dict(pd); pd.update(apply_diffusion(cfg,pd,noise=nz,seq_noise=torch.full((N,),0.5)))
    with torch.no_grad(): r,_=model(pd,pp,uniform=U,compute_distogram=False)
    return list(rec)
a = run([0]); b = run([0,1]); c = run([0,2])
print(f"L={L}: {len(a)} records")
def sub(ta, tb):
    if ta.dim()!=tb.dim() or ta.shape[1:]!=tb.shape[1:]: return None
    if tb.shape[0] == ta.shape[0]*2: return tb[:ta.shape[0]]
    if tb.shape == ta.shape: return tb
    return None
def d(ta, tb):
    if tb is None: return None
    return (ta-tb).abs().max().item() if ta.is_floating_point() else float((ta!=tb).sum())
for k,((na,ja,ta),(nb,jb,tb),(nc,jc,tc)) in enumerate(zip(a,b,c)):
    if k>52: break
    sb, sc = sub(ta,tb), sub(ta,tc)
    if sb is None: print(f"  rec {k:3d} {na}[{ja}] SKIP shapes {tuple(ta.shape)} {tuple(tb.shape)}"); continue
    d_ab = d(ta,sb)
    d_bc = (sb-sc).abs().max().item() if sb.is_floating_point() else float((sb!=sc).sum())
    if k>=30: print(f"  rec {k:3d} {na}[{ja}] {tuple(ta.shape)} alone-vs-pack={d_ab:.3e} packmate_delta={d_bc:.3e}")

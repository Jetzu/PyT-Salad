import sys, torch
import os; sys.path.insert(0, os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n021tree"))
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.diffusion.sampling import make_initial_data, pack_designs, split_packed_snapshot
from salad_torch.diffusion.schedules import parse_num_aa
from salad_torch.modules.model import StructureDiffusionPredict, apply_diffusion
torch.set_float32_matmul_precision("highest")
cfg = get_config("default_vp"); dev = torch.device("cpu")
model = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(model, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
L = int(sys.argv[1])
num_aa, resi, chain, _, _ = parse_num_aa(str(L))
A = model.diffusion.atom_count
def mk(seed):
    d, p = make_initial_data(num_aa, resi, chain, None, device=dev)
    g = torch.Generator().manual_seed(seed)
    d["t_pos"] = torch.full((L,), 0.7); d["t_seq"] = torch.ones(L)
    d["pos"] = torch.randn(L, A, 3, generator=g)
    u = torch.rand(L, L, generator=g).clamp_(1e-6, 1-1e-6)
    nz = torch.randn(L, A, 3, generator=g)
    return d, p, u, nz
D = [mk(200+i) for i in range(4)]
def run(idxs):
    ds = [D[i][0] for i in idxs]; ps = [D[i][1] for i in idxs]
    pd, pp, lens = pack_designs(ds, ps)
    N = L*len(idxs)
    U = torch.full((N, N), 0.5)
    for j, i in enumerate(idxs): U[j*L:(j+1)*L, j*L:(j+1)*L] = D[i][2]
    noise = torch.cat([D[i][3] for i in idxs], 0)
    pd = dict(pd); pd.update(apply_diffusion(cfg, pd, noise=noise, seq_noise=torch.full((N,), 0.5)))
    with torch.no_grad():
        r, _ = model(pd, pp, uniform=U, compute_distogram=False)
    snap = dict(r); snap["batch_index"]=pd["batch_index"]
    return split_packed_snapshot(snap, lens)
alone0 = run([0])[0]
p01 = run([0,1]); p02 = run([0,2]); p012 = run([0,1,2]); p10 = run([1,0])
def cmp(tag, a, b):
    print(f"  {tag:34s} pos {(a['pos']-b['pos']).abs().max():.6e}  local {(a['local']-b['local']).abs().max():.3e}  aatype {(a['aatype']!=b['aatype']).sum().item()}/{L}")
print(f"L={L}: is design 0 sensitive to WHO ELSE is in the pack?")
cmp("[0,1] m0 vs [0,2] m0", p01[0], p02[0])
cmp("[0,1] m0 vs [0,1,2] m0", p01[0], p012[0])
cmp("[0] alone vs [0,1] m0", alone0, p01[0])
cmp("[0] alone vs [1,0] m1 (0 is LAST)", alone0, p10[1])

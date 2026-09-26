"""Turn the N020 job logs into the tables in RESULTS.md."""
import glob
import json
import os
import re
import statistics as st
import sys

R = os.path.dirname(os.path.abspath(__file__))


def bench_rows():
    rows = []
    for f in sorted(glob.glob(f"{R}/gpu_bench*_*.log")):
        for line in open(f):
            if line.startswith("RESULT "):
                rows.append(json.loads(line[7:]))
    return rows


def pershape_rows():
    out = []
    for f in sorted(glob.glob(f"{R}/gpu_shapes_*.log")):
        for line in open(f):
            if line.startswith("PERSHAPE "):
                out.append(json.loads(line[9:]))
    return out


def autotune_choices():
    """Harvest inductor's own AUTOTUNE blocks: winner + per-choice timings."""
    res = {}
    for f in sorted(glob.glob(f"{R}/gpu_bench*_*.log")):
        arm = None
        cur = None
        for line in open(f):
            m = re.match(r"^########## ARM=(\S+) L=(\d+)", line)
            if m:
                arm = f"{m.group(1)}_L{m.group(2)}"
            m = re.match(r"^AUTOTUNE (\w+)\(([^)]*)\)", line)
            if m:
                cur = (m.group(1), m.group(2))
                res.setdefault(arm, {}).setdefault(cur, [])
                continue
            if cur and arm:
                m = re.match(r"^\s{2}(\S+)\s+([\d.]+)\s*(?:ms|us)\s+([\d.]+)%", line)
                if m:
                    res[arm][cur].append((m.group(1), float(m.group(2)), float(m.group(3))))
                elif line.strip() and not line.startswith(" "):
                    cur = None
    return res


if __name__ == "__main__":
    print("## whole-forward\n")
    print("| arm | L | mm_mode | tpl | lean ms (3 reps) | full ms | kernels | tri_tem | triton | gemm | memcpy | GPU busy us | compile s | pos_sum |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(bench_rows(), key=lambda r: (r["L"], r["tag"])):
        c = r["census"]
        lean = "/".join(f"{x:.3f}" for x in r["lean_ms"])
        full = "/".join(f"{x:.3f}" for x in r["full_ms"])
        print(f"| {r['tag']} | {r['L']} | {r['mm_mode']} | {int(r['gemm_templates'])} | {lean} | {full} | "
              f"{r['kernels']:.0f} | {c['triton_tem'][0]:.0f} | {c['triton'][0]:.0f} | {c['gemm'][0]:.0f} | "
              f"{c['memcpy'][0]:.0f} | {r['gpu_busy_us']:.0f} | {r['compile_s']:.0f} | {r['pos_sum']:.6f} |")
    print("\n## per-shape\n")
    for blk in pershape_rows():
        print(f"\n### L={blk['L']} precision={blk['prec']}\n")
        print("| site | M | K | N | bias | n | bmm us | mm us | tpl us | tpl/mm | mm/bmm | max\\|d\\| tpl-vs-cublas | max\\|d\\| bmm-vs-mm | verdict |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in blk["rows"]:
            ratio = r["ratio_tpl_vs_extern"]
            v = "TRITON" if (ratio or 0) > 1.0 else "cuBLAS"
            if r["maxd_tpl_vs_cublas"] != 0.0:
                v += " (not bit-exact)"
            print(f"| {r['site'][:30]} | {r['M']} | {r['K']} | {r['N']} | {int(r['bias'])} | {r['count']} | "
                  f"{r['us_bmm']:.3f} | {r['us_extern']:.3f} | {r['us_tpl']:.3f} | "
                  f"{(ratio or float('nan')):.2f}x | {r['ratio_extern_vs_bmm']:.2f}x | "
                  f"{r['maxd_tpl_vs_cublas']:.3e} | {r['maxd_bmm_vs_mm']:.3e} | {v} |")
    print("\n## inductor's own AUTOTUNE choices\n")
    for arm, shapes in sorted(autotune_choices().items()):
        print(f"\n### {arm}\n")
        print("| op(shape) | winner | winner ms | best ATEN ms | best TRITON ms |")
        print("|---|---|---|---|---|")
        for (op, shape), ch in shapes.items():
            if not ch:
                continue
            aten = [c for c in ch if "extern" in c[0] or "aten" in c[0]]
            tri = [c for c in ch if c[0].startswith("triton")]
            print(f"| {op}({shape}) | {ch[0][0]} | {ch[0][1]:.4f} | "
                  f"{min(a[1] for a in aten) if aten else float('nan'):.4f} | "
                  f"{min(t[1] for t in tri) if tri else float('nan'):.4f} |")

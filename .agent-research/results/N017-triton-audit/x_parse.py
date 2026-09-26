"""N017 step 2: attribute every profiled kernel to source.

Chain:
  profiler kernel name
    -> inductor `output_code.py` kernel definition (carries the post-grad node
       names it fused, in the `# Graph fragment:` block)
    -> `fx_graph_readable.py` node, which carries the INNERMOST `# File:` frame
    -> the nearest preceding CALLER-level `# File:` frame ("anchor"), because
       the innermost frame is usually a leaf helper (`rigid.vec_norm2`) that
       says nothing about which stage called it.
"""
import json
import os
import re
import sys
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__))
IND = os.path.join(OUT, "inductor")

# files that are CALL SITES rather than leaf helpers
ANCHOR = {"diffusion.py", "blocks.py", "attention.py", "cond_prologue.py",
          "decoders.py", "model.py", "sidechain.py"}

FILE_RE = re.compile(r"\s*#\s*File:\s*(\S+?):(\d+)\s+in\s+(\S+),\s*code:\s*(.*)")
NODE_RE = re.compile(r"\s*(\w+)\s*:\s*\"[^\"]*\"\s*=\s*")


def load_fx(path):
    nodes = {}      # name -> dict(order, inner, anchor, fn, code)
    order = 0
    inner = anchor = None
    for line in open(path, errors="replace"):
        m = FILE_RE.match(line)
        if m:
            f, ln, fn, code = os.path.basename(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            inner = (f, ln, fn, code.strip())
            if f in ANCHOR:
                anchor = inner
            continue
        m = NODE_RE.match(line)
        if m:
            nodes[m.group(1)] = dict(order=order, inner=inner, anchor=anchor)
            order += 1
    return nodes


KDEF = re.compile(r"^(triton_\w+?_(\d+))\s*=\s*async_compile\.triton")


def load_kernels(path):
    kernels = {}
    block = []
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = KDEF.match(line)
        if m:
            info = {"aten": [], "nodes": [], "src": []}
            for bl in block:
                mm = re.match(r"#\s*Topologically Sorted Source Nodes:\s*\[(.*?)\],\s*Original ATen:\s*\[(.*?)\]", bl)
                if mm:
                    info["src"] = [s.strip() for s in mm.group(1).split(",") if s.strip()]
                    info["aten"] = [s.strip() for s in mm.group(2).split(",") if s.strip()]
                mm = re.match(r"#\s*%(\w+)\s*:", bl)
                if mm:
                    info["nodes"].append(mm.group(1))
            kernels[m.group(1)] = info
            block = []
        elif line.startswith("#"):
            block.append(line)
        elif line.strip():
            block = []
    return kernels


def stage_of(anchor, inner):
    """(stage, detail) from the anchor call site and the innermost frame."""
    if anchor is None:
        return "<prologue/no-anchor>", (inner[0] + ":" + str(inner[1])) if inner else "?"
    f, ln, fn, code = anchor
    det = f"{f}:{ln}"
    if f == "attention.py":
        return "IPA attention", det
    if f == "cond_prologue.py":
        return "P7 batched cond-pair prologue", det
    if f == "decoders.py":
        return "decoders", det
    if f == "sidechain.py":
        return "sidechain", det
    if f == "blocks.py":
        if fn in ("inner",):
            return "neighbour selection (cond branch)", det
        if "pair_1" in code or "pair_2" in code or fn == "forward" and "DiffusionPairFeatures" in code:
            return "block glue", det
        return {"forward"}.__contains__(fn) and ("block: " + _blockpart(code, ln)) or ("blocks:" + fn), det
    if f == "diffusion.py":
        if fn in ("neighbour_fn", "cond_neighbours_of", "inner"):
            return "neighbour selection (current branch)", det
        return "forward_core prologue", det
    return f"other:{f}", det


def _blockpart(code, ln):
    if "self.pair_1" in code or "self.pair_2" in code:
        return "pair features (call)"
    if "self.attn_" in code:
        return "attention (call)"
    if "self.update(" in code:
        return "global update"
    if "update_positions" in code:
        return "position update"
    if "layer_norm" in code:
        return "layer norms"
    return f"line {ln}"


def main():
    prof = json.load(open(os.path.join(OUT, "profile_lean.json")))
    rows = prof["captured"]
    kernels, nodes = {}, {}
    for g in sorted(os.listdir(IND)):
        d = os.path.join(IND, g)
        if os.path.exists(os.path.join(d, "output_code.py")):
            kernels.update(load_kernels(os.path.join(d, "output_code.py")))
        if os.path.exists(os.path.join(d, "fx_graph_readable.py")):
            nodes.update(load_fx(os.path.join(d, "fx_graph_readable.py")))
    print(f"# parsed {len(kernels)} kernel defs, {len(nodes)} fx nodes", file=sys.stderr)

    by_stage = defaultdict(lambda: {"count": 0.0, "us": 0.0})
    by_detail = defaultdict(lambda: {"count": 0.0, "us": 0.0})
    per_kernel = []
    missing = []
    for r in rows:
        name, cnt, us = r["name"], r["count"], r["us_total"]
        if not name.startswith("triton_"):
            low = name.lower()
            kind = ("<cublas>" if ("gemm" in low or "cutlass" in low or "sm80" in low
                                   or "sm90" in low or "ampere" in low or "cublas" in low)
                    else "<memcpy>" if "memcpy" in low or "memset" in low
                    else f"<aten:{name.split('<')[0][:40]}>")
            by_stage[kind]["count"] += cnt
            by_stage[kind]["us"] += us
            by_detail[kind]["count"] += cnt
            by_detail[kind]["us"] += us
            continue
        info = kernels.get(name)
        if info is None:
            missing.append(name)
            by_stage["<triton unmatched>"]["count"] += cnt
            by_stage["<triton unmatched>"]["us"] += us
            continue
        ns = [nodes[n] for n in info["nodes"] if n in nodes]
        ns.sort(key=lambda d: d["order"])
        if not ns:
            st, det = "<no fx node>", "?"
        else:
            st, det = stage_of(ns[0]["anchor"], ns[0]["inner"])
        by_stage[st]["count"] += cnt
        by_stage[st]["us"] += us
        by_detail[f"{st} | {det}"]["count"] += cnt
        by_detail[f"{st} | {det}"]["us"] += us
        per_kernel.append(dict(name=name, count=cnt, us=us, us_each=r["us_each"],
                               stage=st, detail=det, aten=info["aten"],
                               src=info["src"][:12],
                               inner=sorted({f"{n['inner'][0]}:{n['inner'][1]}"
                                             for n in ns if n["inner"]})))
    json.dump({"by_stage": dict(by_stage), "by_detail": dict(by_detail),
               "per_kernel": per_kernel, "unmatched": missing},
              open(os.path.join(OUT, "attribution.json"), "w"), indent=1)

    tot = sum(v["us"] for v in by_stage.values())
    ntot = sum(v["count"] for v in by_stage.values())
    print(f"\n{'STAGE':44s} {'count':>6s} {'us/step':>8s} {'%':>6s}")
    for k, v in sorted(by_stage.items(), key=lambda kv: -kv[1]["us"]):
        print(f"{k:44s} {v['count']:6.0f} {v['us']:8.1f} {100*v['us']/tot:5.1f}%")
    print(f"{'TOTAL':44s} {ntot:6.0f} {tot:8.1f}")
    print(f"\n{'DETAIL':62s} {'count':>6s} {'us/step':>8s}")
    for k, v in sorted(by_detail.items(), key=lambda kv: -kv[1]["us"])[:40]:
        print(f"{k:62s} {v['count']:6.0f} {v['us']:8.1f}")
    print(f"\nunmatched triton kernel names: {len(missing)} {missing[:10]}")
    print(f"\n{'TOP KERNELS BY TIME':40s} {'n':>4s} {'us':>7s} {'us/ea':>7s}  stage")
    for r in sorted(per_kernel, key=lambda r: -r["us"])[:25]:
        print(f"{r['name'][:40]:40s} {r['count']:4.0f} {r['us']:7.1f} {r['us_each']:7.2f}  "
              f"{r['stage']} | {r['detail']}")
    print(f"\n{'TOP KERNELS BY COUNT':40s} {'n':>4s} {'us':>7s} {'us/ea':>7s}  stage")
    for r in sorted(per_kernel, key=lambda r: -r["count"])[:25]:
        print(f"{r['name'][:40]:40s} {r['count']:4.0f} {r['us']:7.1f} {r['us_each']:7.2f}  "
              f"{r['stage']} | {r['detail']}")


main()

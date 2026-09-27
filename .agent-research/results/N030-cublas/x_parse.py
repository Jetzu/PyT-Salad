"""N030: RESULT lines -> a table."""
import json
import sys

rows = []
for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith("RESULT "):
            rows.append(json.loads(line[7:]))
base = None
print(f"{'arm':<18}{'L':>5}{'lean ms':>10}{'vs base':>9}{'full ms':>10}"
      f"{'kern':>6}{'tri':>6}{'tem':>6}{'gemm':>6}{'gemm us':>9}{'busy us':>9}{'cmpl s':>8}")
for r in rows:
    c = r["census"]
    lean = min(r["lean_ms"])
    if r["arm"] == "off":
        base = lean
    d = f"{(lean/base-1)*100:+.1f}%" if base else "-"
    print(f"{r['arm']:<18}{r['L']:>5}{lean:>10.4f}{d:>9}{min(r['full_ms']):>10.4f}"
          f"{r['kernels']:>6.0f}{c['triton'][0]:>6.0f}{c['triton_tem'][0]:>6.0f}"
          f"{c['gemm'][0]:>6.0f}{c['gemm'][1]:>9.1f}{r['gpu_busy_us']:>9.1f}"
          f"{r['compile_s']:>8.0f}")
print()
for r in rows:
    if r.get("filter"):
        f = r["filter"]
        print(f"{r['arm']}: filter {json.dumps({k: v for k, v in f.items()})}")
    if r.get("counters"):
        print(f"{r['arm']}: counters {r['counters']}")

#!/usr/bin/env python3
"""Turn the sweep CSVs into the markdown tables used by README.md.

Nothing here computes a result: every number comes from a CSV that a run wrote,
so a table in the document can always be regenerated from the data beside it.

    ./report.py            # all sections
    ./report.py matrix     # just the correctness tables
"""

import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ORDER, TOPOLOGIES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, half


def read(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def matrix_tables():
    out = []
    for fname, title in (("matrix-scaled.csv", "scaled prompt"),
                         ("matrix-builtin.csv", "built in prompt")):
        rows = read(fname)
        if not rows:
            continue
        agg = defaultdict(lambda: [0, 0, 0.0, 0])
        for r in rows:
            k = r["topology"]
            p = int(r["cases_pass"])
            f = int(r["cases_fail"])
            agg[k][0] += p
            agg[k][1] += p + f
            agg[k][3] += 1
            try:
                v = float(r["max_nmse"])
            except ValueError:
                continue
            if v == v:
                agg[k][2] = max(agg[k][2], v)

        out.append(f"### {title}\n")
        out.append("| topology | runs | cases passed | rate | worst NMSE |")
        out.append("|---|---|---|---|---|")
        for topo in ORDER:
            if topo not in agg:
                continue
            p, n, worst, runs = agg[topo]
            rate, half = wilson(p, n)
            out.append(f"| {TOPOLOGIES[topo]['label']} | {runs} | {p}/{n} | "
                       f"{rate*100:.2f} +/- {half*100:.2f} % | {worst:.2e} |")
        out.append("")

        # per context detail, one row per topology
        ctxs = sorted({int(r["ctx"]) for r in rows})
        cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in rows:
            c = cells[r["topology"]][int(r["ctx"])]
            c[0] += int(r["cases_pass"])
            c[1] += int(r["cases_pass"]) + int(r["cases_fail"])
        out.append("| topology | " + " | ".join(str(c) for c in ctxs) + " |")
        out.append("|---" * (len(ctxs) + 1) + "|")
        for topo in ORDER:
            if topo not in cells:
                continue
            line = [TOPOLOGIES[topo]["label"]]
            for c in ctxs:
                p, n = cells[topo][c]
                if n == 0:
                    line.append("n/a")
                elif p == n:
                    line.append("pass")
                else:
                    line.append(f"**{p}/{n}**")
            out.append("| " + " | ".join(line) + " |")
        out.append("")
    return "\n".join(out)


def bench_tables():
    out = []
    for fname, title in (("bench-qwen06.csv", "Qwen3-0.6B Q4_K_M"),
                         ("bench-qwen8.csv", "Qwen3-8B Q8_0"),
                         ("bench-qwen30.csv", "Qwen3-30B-A3B Q4_K_M")):
        rows = read(fname)
        if not rows:
            continue
        depths = sorted({int(r["depth"]) for r in rows})
        cells = defaultdict(list)
        for r in rows:
            try:
                v = float(r["avg_ts"])
            except ValueError:
                continue
            if v != v:
                continue
            cells[(r["topology"], r["kind"], int(r["depth"]))].append(v)

        for kind, label in (("pp", "prompt processing (pp512)"),
                            ("tg", "token generation (tg128)")):
            out.append(f"### {title}, {kind} tokens/s at depth\n")
            out.append(f"{label}\n")
            out.append("| topology | " + " | ".join(str(d) for d in depths) + " |")
            out.append("|---" * (len(depths) + 1) + "|")
            for topo in ORDER:
                vals = [cells.get((topo, kind, d)) for d in depths]
                if not any(vals):
                    continue
                line = [TOPOLOGIES[topo]["label"]]
                for v in vals:
                    line.append("-" if not v else f"{sum(v)/len(v):.1f}")
                out.append("| " + " | ".join(line) + " |")
            out.append("")

        # how much the second die adds
        for kind in ("pp", "tg"):
            out.append(f"### {title}, {kind}, both dies against one die\n")
            out.append("| depth | ROCm 1 die | ROCm 2 dies | ratio | Vulkan 1 die | Vulkan 2 dies | ratio |")
            out.append("|---|---|---|---|---|---|---|")

            def mean(topo, k, d):
                v = cells.get((topo, k, d))
                return sum(v) / len(v) if v else float("nan")

            for d in depths:
                r1, rb = mean("rocm-die1", kind, d), mean("rocm-both", kind, d)
                v1, vb = mean("vk-die1", kind, d), mean("vk-both", kind, d)
                rr = rb / r1 if r1 == r1 and r1 else float("nan")
                vr = vb / v1 if v1 == v1 and v1 else float("nan")
                out.append(f"| {d} | {r1:.1f} | {rb:.1f} | {rr:.2f}x | "
                           f"{v1:.1f} | {vb:.1f} | {vr:.2f}x |")
            out.append("")
    return "\n".join(out)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "matrix"):
        print(matrix_tables())
    if what in ("all", "bench"):
        print(bench_tables())


if __name__ == "__main__":
    main()

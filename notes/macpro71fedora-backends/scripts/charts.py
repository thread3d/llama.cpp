#!/usr/bin/env python3
"""Draw the charts for the MacPro71Fedora backend study.

Every figure is written to ../charts. The statistics are the same as in the
Vulkan context study so the two documents agree:

  - a rate is shown with a Wilson score interval over the individual case
    verdicts, so the bar width reflects how many verdicts the cell has
  - a throughput is shown as the mean over repetitions with a Student t
    interval for 95 %

    ./charts.py
"""

import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ORDER, TOPOLOGIES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
CHARTS = os.path.join(HERE, "charts")

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179}

GROUP_COLOR = {"cpu": "#444444", "rocm": "#1f5fa8", "vk": "#1b7f4b"}


def t_crit(n):
    return T95.get(n, 1.96) if n > 1 else 0.0


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, half


def load_matrix(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["ctx"] = int(r["ctx"])
                r["cases_pass"] = int(r["cases_pass"])
                r["cases_fail"] = int(r["cases_fail"])
                r["cases_skip"] = int(r["cases_skip"])
                r["max_nmse"] = float(r["max_nmse"])
                r["exit"] = int(r["exit"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def load_bench(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["depth"] = int(r["depth"])
                r["avg_ts"] = float(r["avg_ts"])
                r["stddev_ts"] = float(r["stddev_ts"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def style(ax, ylabel, title, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8, framealpha=0.9)


def fig_pass_rate(rows, fname, title):
    """Case pass rate per topology against context size."""
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["topology"], r["ctx"])
        cells[k][0] += r["cases_pass"]
        cells[k][1] += r["cases_pass"] + r["cases_fail"]

    ctxs = sorted({r["ctx"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xscale("log", base=2)
    worst = 1.0
    for topo in ORDER:
        if not any(t == topo for t, _ in cells):
            continue
        pts = [(c, *wilson(*cells[(topo, c)])) for c in ctxs if (topo, c) in cells]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        es = [p[2] * 100 for p in pts]
        spec = TOPOLOGIES[topo]
        ax.errorbar(xs, ys, yerr=es, label=spec["label"], color=spec["color"],
                    marker=spec["marker"], capsize=3, lw=1.6, ms=5)
        worst = min(worst, min(ys))

    ax.set_ylim(max(0, worst - 3), 101)
    style(ax, "case pass rate (%)", title, "context size (tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, fname), dpi=140)
    plt.close(fig)


def fig_nmse(rows, fname, title):
    """Worst NMSE per run against context size, on a log scale."""
    cells = defaultdict(list)
    for r in rows:
        if r["max_nmse"] == r["max_nmse"] and r["max_nmse"] > 0:
            cells[(r["topology"], r["ctx"])].append(r["max_nmse"])

    ctxs = sorted({r["ctx"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    for topo in ORDER:
        pts = []
        for c in ctxs:
            vals = cells.get((topo, c))
            if not vals:
                continue
            logs = np.log10(np.asarray(vals))
            mean = 10 ** logs.mean()
            half = 0.0
            if logs.size > 1:
                half = t_crit(logs.size) * logs.std(ddof=1) / math.sqrt(logs.size)
            pts.append((c, mean, mean * (10 ** half - 1)))
        if not pts:
            continue
        spec = TOPOLOGIES[topo]
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    label=spec["label"], color=spec["color"], marker=spec["marker"],
                    capsize=3, lw=1.6, ms=5)

    ax.set_yscale("log")
    ax.axhline(1e-5, color="#b03030", ls="--", lw=1, label="1e-5 tolerance")
    style(ax, "worst NMSE of a run", title, "context size (tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, fname), dpi=140)
    plt.close(fig)


def fig_throughput(rows, kind, fname, title, ylabel):
    """Throughput against KV depth, one line per topology."""
    if not rows:
        return
    cells = defaultdict(list)
    for r in rows:
        if r["kind"] == kind and r["avg_ts"] == r["avg_ts"]:
            cells[(r["topology"], r["depth"])].append(r["avg_ts"])

    depths = sorted({r["depth"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    for topo in ORDER:
        pts = []
        for d in depths:
            vals = cells.get((topo, d))
            if not vals:
                continue
            v = np.asarray(vals)
            half = 0.0
            if v.size > 1:
                half = t_crit(v.size) * v.std(ddof=1) / math.sqrt(v.size)
            pts.append((d, v.mean(), half))
        if not pts:
            continue
        spec = TOPOLOGIES[topo]
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    label=spec["label"], color=spec["color"], marker=spec["marker"],
                    capsize=3, lw=1.6, ms=5)

    ax.set_yscale("log")
    style(ax, ylabel, title, "KV depth (tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, fname), dpi=140)
    plt.close(fig)


def fig_speedup(rows, fname, title):
    """Speedup over a single die, per topology, at the deepest measured depth."""
    if not rows:
        return
    cells = defaultdict(list)
    for r in rows:
        if r["avg_ts"] == r["avg_ts"]:
            cells[(r["topology"], r["kind"], r["depth"])].append(r["avg_ts"])
    if not cells:
        return
    depths = sorted({k[2] for k in cells})
    depth = depths[len(depths) // 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, kind, name in ((axes[0], "pp", "prompt processing"), (axes[1], "tg", "token generation")):
        base = None
        labels, vals, colors = [], [], []
        for topo in ORDER:
            v = cells.get((topo, kind, depth))
            if not v:
                continue
            mean = float(np.mean(v))
            if topo == "rocm-die1":
                base = mean
            labels.append(topo)
            vals.append(mean)
            colors.append(TOPOLOGIES[topo]["color"])
        if base is None and vals:
            base = vals[0]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"tokens/s at depth {depth}")
        ax.set_title(name)
        ax.grid(True, axis="x", alpha=0.25)
    axes[0].set_ylabel("topology")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, fname), dpi=140)
    plt.close(fig)


def markdown_summary(rows_short, rows_scaled, bench):
    """A compact table block for the README, from the same data as the charts."""
    out = []
    worst = defaultdict(lambda: (0, 0))
    for r in rows_scaled:
        k = r["topology"]
        p, n = worst[k]
        worst[k] = (p + r["cases_pass"], n + r["cases_pass"] + r["cases_fail"])
    out.append("| topology | cases passed | rate |")
    out.append("|---|---|---|")
    for topo in ORDER:
        if topo not in worst:
            continue
        p, n = worst[topo]
        rate, half = wilson(p, n)
        out.append(f"| {TOPOLOGIES[topo]['label']} | {p}/{n} | {rate*100:.2f} +/- {half*100:.2f} % |")
    return "\n".join(out)


def main():
    os.makedirs(CHARTS, exist_ok=True)

    scaled = load_matrix("matrix-scaled.csv")
    builtin = load_matrix("matrix-builtin.csv")

    fig_pass_rate(scaled, "matrix-pass-scaled.png",
                  "KV state reproducibility, scaled prompt (about half the context)")
    fig_pass_rate(builtin, "matrix-pass-builtin.png",
                  "KV state reproducibility, built in prompt (100 tokens)")
    fig_nmse(scaled, "matrix-nmse.png",
             "Worst NMSE of a restored context, scaled prompt")

    bench06 = load_bench("bench-qwen06.csv")
    fig_throughput(bench06, "pp", "bench-pp-qwen06.png",
                   "Prompt processing, Qwen3-0.6B Q4_K_M", "tokens/s")
    fig_throughput(bench06, "tg", "bench-tg-qwen06.png",
                   "Token generation, Qwen3-0.6B Q4_K_M", "tokens/s")
    fig_speedup(bench06, "bench-speedup.png",
                "Throughput per topology at mid depth")

    bench8 = load_bench("bench-qwen8.csv")
    fig_throughput(bench8, "pp", "bench-pp-qwen8.png",
                   "Prompt processing, Qwen3-8B Q8_0", "tokens/s")
    fig_throughput(bench8, "tg", "bench-tg-qwen8.png",
                   "Token generation, Qwen3-8B Q8_0", "tokens/s")

    bench30 = load_bench("bench-qwen30.csv")
    fig_throughput(bench30, "tg", "bench-tg-qwen30.png",
                   "Token generation, Qwen3-30B-A3B Q4_K_M", "tokens/s")

    print(markdown_summary(scaled, scaled, bench06))
    print(f"charts written to {CHARTS}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Turn scripts/bench-macpro31.sh output into a table and SVG charts.

Every llama-bench run emits two records: prompt processing (n_gen == 0) and token generation
(n_gen > 0). Each record carries that run's throughput samples. This script pools the samples of
all repetitions of a condition into a mean and a sample standard deviation, and draws the standard
deviation as the error bars.

The two phases are analysed separately: the context phase sweeps context depth across three
models, the quant phase compares four quantization levels of one model at depth 0.

Usage: scripts/plot-macpro31.py <context.jsonl> <quant.jsonl> <out-dir>

Requires matplotlib. Set MPLCONFIGDIR if HOME is not writable.
"""

import csv
import json
import os
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl-macpro31"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "Qwen2.5-0.5B q4_K_M",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "Qwen2.5-1.5B q4_K_M",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf": "Llama-3.2-1B q4_K_M",
    "qwen2.5-0.5b-instruct-q4_0.gguf": "q4_0",
    "qwen2.5-0.5b-instruct-q8_0.gguf": "q8_0",
    "qwen2.5-0.5b-instruct-fp16.gguf": "f16",
}

CONTEXT_ORDER = ["Qwen2.5-0.5B q4_K_M", "Llama-3.2-1B q4_K_M", "Qwen2.5-1.5B q4_K_M"]
QUANT_ORDER = ["q4_0", "q4_K_M", "q8_0", "f16"]
# The quant phase reuses the 0.5B q4_K_M file, so its labels are phase-specific
QUANT_LABELS = {
    "qwen2.5-0.5b-instruct-q4_0.gguf": "q4_0",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "q4_K_M",
    "qwen2.5-0.5b-instruct-q8_0.gguf": "q8_0",
    "qwen2.5-0.5b-instruct-fp16.gguf": "f16",
}
COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
KINDS = [("prompt", "prompt processing"), ("generation", "token generation")]


def load(path, table):
    """-> {(label, kind, depth): [samples]}, {label: n_params}"""
    groups, params = {}, {}
    if not path or not os.path.exists(path):
        return groups, params
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = os.path.basename(rec["model_filename"])
            label = table.get(name, name)
            params[label] = rec.get("model_n_params", 0)
            kind = "generation" if rec["n_gen"] > 0 else "prompt"
            groups.setdefault((label, kind, rec["n_depth"]), []).extend(
                rec.get("samples_ts") or [rec["avg_ts"]])
    return groups, params


def stats(samples):
    n = len(samples)
    mean = statistics.fmean(samples)
    return mean, (statistics.stdev(samples) if n > 1 else 0.0), n, min(samples), max(samples)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_vs_depth(groups, kind, human, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    depths = sorted({d for (_, k, d) in groups if k == kind})
    present = [l for l in CONTEXT_ORDER if any(kk[0] == l and kk[1] == kind for kk in groups)]
    for i, label in enumerate(present):
        xs, ys, es = [], [], []
        for depth in depths:
            samples = groups.get((label, kind, depth))
            if not samples:
                continue
            mean, sd, _, _, _ = stats(samples)
            xs.append(0.5 if depth == 0 else depth)  # keep depth 0 visible on a log axis
            ys.append(mean)
            es.append(sd)
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=5, linewidth=1.5, capsize=4,
                    elinewidth=1.1, color=COLORS[i % len(COLORS)], label=label)
    ax.set_xscale("log")
    ax.set_xticks([0.5] + [d for d in depths if d > 0])
    ax.set_xticklabels(["0"] + [str(d) for d in depths if d > 0])
    style(ax, f"{human} throughput against context depth", "context depth (tokens)", "tokens/s")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_quantisation(groups, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    present = [l for l in QUANT_ORDER if any(kk[0] == l for kk in groups)]
    width = 0.36
    for j, (kind, human) in enumerate(KINDS):
        xs, means, errs = [], [], []
        for i, label in enumerate(present):
            samples = groups.get((label, kind, 0))
            if not samples:
                continue
            mean, sd, _, _, _ = stats(samples)
            xs.append(i + (j - 0.5) * width)
            means.append(mean)
            errs.append(sd)
        ax.bar(xs, means, width, yerr=errs, capsize=4, label=human, color=COLORS[j],
               edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 1.0})
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present)
    style(ax, "Qwen2.5-0.5B throughput by quantization (depth 0)", "quantization", "tokens/s")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_scaling(groups, params, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for j, (kind, human) in enumerate(KINDS):
        points = []
        for label in CONTEXT_ORDER:
            samples = groups.get((label, kind, 0))
            if not samples:
                continue
            mean, sd, _, _, _ = stats(samples)
            points.append((params.get(label, 0) / 1e6, mean, sd))
        points.sort()
        if not points:
            continue
        ax.errorbar([p[0] for p in points], [p[1] for p in points], yerr=[p[2] for p in points],
                    marker="s", markersize=5, linewidth=1.5, capsize=4, color=COLORS[j], label=human)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "Throughput against model size (q4_K_M, depth 0)", "parameters (millions)", "tokens/s")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_csv(rows, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "model", "test", "depth", "n", "mean_ts", "sd_ts", "min_ts", "max_ts"])
        w.writerows(rows)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ctx_path, quant_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    ctx, ctx_params = load(ctx_path, LABELS)
    quant, quant_params = load(quant_path, QUANT_LABELS)
    if not ctx and not quant:
        sys.exit("no benchmark records found")

    if ctx:
        plot_vs_depth(ctx, "prompt", "Prompt processing", os.path.join(outdir, "prompt-throughput-vs-depth.svg"))
        plot_vs_depth(ctx, "generation", "Token generation", os.path.join(outdir, "generation-throughput-vs-depth.svg"))
        plot_scaling(ctx, ctx_params, os.path.join(outdir, "scaling.svg"))
    if quant:
        plot_quantisation(quant, os.path.join(outdir, "quantisation.svg"))

    rows = []
    for phase, groups in (("context", ctx), ("quant", quant)):
        for (label, kind, depth) in sorted(groups):
            mean, sd, n, lo, hi = stats(groups[(label, kind, depth)])
            rows.append([phase, label, kind, depth, n, f"{mean:.3f}", f"{sd:.3f}", f"{lo:.3f}", f"{hi:.3f}"])
            print(f"{phase:8s} {label:22s} {kind:10s} depth={depth:5d} n={n} mean={mean:8.2f} sd={sd:6.2f}")
    write_csv(rows, os.path.join(outdir, "results.csv"))


if __name__ == "__main__":
    main()

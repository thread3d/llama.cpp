#!/usr/bin/env python3
"""Charts for the macpro31 larger-model and larger-context sweep.

Reads the JSONL produced by `scripts/bench-macpro31.sh large`:

  <base>-scale.jsonl   six models at depth 0, prompt and generation
  <base>-depth.jsonl   the same small models at depths up to 4096

The scale file supplies the depth-0 baseline; the depth file supplies the deeper points, so the
charts combine them. Each point is the mean of its runs and the error bar is one sample standard
deviation, so the bars show run-to-run spread, not a confidence interval for the mean.

Usage: scripts/plot-macpro31-large.py <scale.jsonl> <depth.jsonl> <out-dir>

Requires matplotlib. Set MPLCONFIGDIR if HOME is not writable.
"""

import os
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl-macpro31-large"))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "Qwen2.5-0.5B",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf": "Llama-3.2-1B",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "Qwen2.5-1.5B",
    "qwen2.5-3b-instruct-q4_k_m.gguf": "Qwen2.5-3B",
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf": "Mistral-7B",
    "Qwen3-8B-Q4_K_M.gguf": "Qwen3-8B",
}

ORDER = ["Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Mistral-7B", "Qwen3-8B"]
COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
KINDS = [("prompt", "prompt processing"), ("generation", "token generation")]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load(path):
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
            label = LABELS.get(name, name)
            params[label] = rec.get("model_n_params", 0)
            kind = "generation" if rec["n_gen"] > 0 else "prompt"
            groups.setdefault((label, kind, rec["n_depth"]), []).extend(
                rec.get("samples_ts") or [rec["avg_ts"]])
    return groups, params


def stats(samples):
    n = len(samples)
    mean = statistics.fmean(samples)
    return mean, (statistics.stdev(samples) if n > 1 else 0.0), n, samples


def merge(*group_sets):
    out = {}
    for groups in group_sets:
        for k, v in groups.items():
            out.setdefault(k, []).extend(v)
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_scaling(groups, params, out):
    """Throughput against parameter count at depth 0."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for j, (kind, human) in enumerate(KINDS):
        pts = []
        for label in ORDER:
            samples = groups.get((label, kind, 0))
            if not samples:
                continue
            mean, sd, _, _ = stats(samples)
            pts.append((params.get(label, 0) / 1e6, mean, sd))
        pts.sort()
        if not pts:
            continue
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    marker="o", markersize=5, linewidth=1.5, capsize=4, color=COLORS[j], label=human)
        for k, label in enumerate(ORDER):
            samples = groups.get((label, kind, 0))
            if kind == "generation" and samples:
                x = params.get(label, 0) / 1e6
                y = stats(samples)[0]
                offset = (6, -12) if k % 2 == 0 else (-8, 8)
                ha = "left" if k % 2 == 0 else "right"
                ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                            fontsize=7, ha=ha)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "Throughput against model size (q4_K_M, depth 0)", "parameters (millions)", "tokens/s")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_vs_depth(groups, kind, human, out, logy=False):
    """Throughput against context depth, one line per model."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    depths_all = sorted({d for (_, k, d) in groups if k == kind})
    present = [l for l in ORDER if any(kk[0] == l and kk[1] == kind for kk in groups)]
    for i, label in enumerate(present):
        xs, ys, es = [], [], []
        for depth in depths_all:
            samples = groups.get((label, kind, depth))
            if not samples:
                continue
            mean, sd, _, _ = stats(samples)
            xs.append(depth)
            ys.append(mean)
            es.append(sd)
        ax.errorbar(xs, ys, yerr=es, marker=MARKERS[i % len(MARKERS)], markersize=5, linewidth=1.5,
                    capsize=4, color=COLORS[ORDER.index(label) % len(COLORS)], label=label)
    ax.set_xscale("symlog", linthresh=256)
    ax.set_xticks([d for d in depths_all])
    ax.set_xticklabels([str(d) for d in depths_all])
    if logy:
        ax.set_yscale("log")
    style(ax, f"{human} against context depth", "context depth (tokens)", "tokens/s")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_retention(groups, out):
    """Throughput as a percentage of the depth-0 value."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ls = {"prompt": "--", "generation": "-"}
    model_handles = []
    for i, label in enumerate(ORDER):
        # a model measured only at depth 0 has nothing to say about retention
        if not any(k[0] == label and k[2] > 0 for k in groups):
            continue
        handle = None
        for kind, _ in KINDS:
            base = groups.get((label, kind, 0))
            if not base:
                continue
            base_mean = stats(base)[0]
            xs, ys, es = [], [], []
            for depth in sorted({d for (_, k, d) in groups if k == kind}):
                samples = groups.get((label, kind, depth))
                if not samples:
                    continue
                mean, sd, _, _ = stats(samples)
                xs.append(max(depth, 1))
                ys.append(100.0 * mean / base_mean)
                es.append(100.0 * sd / base_mean)
            line = ax.errorbar(xs, ys, yerr=es, marker=MARKERS[i % len(MARKERS)], markersize=4,
                               linewidth=1.3, linestyle=ls[kind], capsize=3,
                               color=COLORS[i % len(COLORS)],
                               label=label if kind == "generation" else None)
            if kind == "generation":
                handle = line
        if handle is not None:
            model_handles.append(handle)
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([], [], color="black", linestyle="-", linewidth=1.3, label="token generation"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.3, label="prompt processing"),
    ]
    ax.set_xscale("log")
    ax.axhline(100.0, color="black", linewidth=0.8, alpha=0.5)
    style(ax, "Throughput retained against context depth (100% = depth 0)",
          "context depth (tokens)", "percent of depth-0 throughput")
    ax.legend(handles=model_handles + style_handles, fontsize=7, frameon=False,
              loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_depth_bars(groups, out, depth=2048):
    """Generation throughput at depth 0 against one deep depth."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    labels = [l for l in ORDER if groups.get((l, "generation", 0)) and groups.get((l, "generation", depth))]
    width = 0.36
    for j, d in enumerate((0, depth)):
        xs, means, errs = [], [], []
        for i, label in enumerate(labels):
            samples = groups[(label, "generation", d)]
            mean, sd, _, _ = stats(samples)
            xs.append(i + (j - 0.5) * width)
            means.append(mean)
            errs.append(sd)
        ax.bar(xs, means, width, yerr=errs, capsize=4, label=f"depth {d}",
               color=COLORS[j], edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 1.0})
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    style(ax, f"Token generation at depth 0 and depth {depth}", "model", "tokens/s")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    scale_path, depth_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    scale, params = load(scale_path)
    deep, params_deep = load(depth_path)
    params.update({k: v for k, v in params_deep.items() if v})
    groups = merge(scale, deep)
    if not groups:
        sys.exit("no records found")

    plot_scaling(groups, params, os.path.join(outdir, "scale-throughput-vs-size.svg"))
    plot_vs_depth(groups, "generation", "Token generation", os.path.join(outdir, "depth-generation.svg"))
    plot_vs_depth(groups, "prompt", "Prompt processing", os.path.join(outdir, "depth-prompt.svg"), logy=True)
    plot_retention(groups, os.path.join(outdir, "depth-retention.svg"))
    plot_depth_bars(groups, os.path.join(outdir, "depth-bars-2048.svg"))

    for (label, kind, depth) in sorted(groups):
        mean, sd, n, _ = stats(groups[(label, kind, depth)])
        print(f"{label:14s} {kind:10s} depth={depth:5d} n={n} mean={mean:8.2f} sd={sd:6.2f}")


if __name__ == "__main__":
    main()

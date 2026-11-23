#!/usr/bin/env python3
"""Compare sweeps that differ only in thread count.

Every set is a `bench-macpro31.sh coverage` dataset (one or more JSONL files) tagged with the thread
count it was measured at. The first set is the baseline that the ratios refer to. Draws one grouped
bar chart per test plus a long-format table, so the whole 4/6/7-thread series is one artefact.

Usage:
    scripts/plot-macpro31-threads.py <out-dir> <label>=<jsonl>[,<jsonl>...] [<label>=<jsonl>...]

Example:
    scripts/plot-macpro31-threads.py docs/macpro31 \
        7t=docs/macpro31/bench-coverage.jsonl,docs/macpro31/bench-coverage-xl.jsonl \
        6t=docs/macpro31/bench-coverage-6t.jsonl \
        4t=docs/macpro31/bench-coverage-4t.jsonl

Requires matplotlib. Set MPLCONFIGDIR if HOME is not writable.
"""

import json
import os
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl-macpro31-thr"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "0.5B q4_K_M",
    "qwen2.5-0.5b-instruct-q4_0.gguf": "0.5B q4_0",
    "qwen2.5-0.5b-instruct-q8_0.gguf": "0.5B q8_0",
    "qwen2.5-0.5b-instruct-fp16.gguf": "0.5B f16",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf": "1B q4_K_M",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "1.5B q4_K_M",
    "qwen2.5-1.5b-instruct-q4_0.gguf": "1.5B q4_0",
    "qwen2.5-3b-instruct-q4_k_m.gguf": "3B q4_K_M",
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf": "7B q4_K_M",
    "Mistral-7B-Instruct-v0.3-Q4_0.gguf": "7B q4_0",
    "Qwen3-8B-Q4_K_M.gguf": "8B q4_K_M",
    "Qwen2.5-14B-Instruct-Q4_K_M.gguf": "14B q4_K_M",
    "Qwen2.5-32B-Instruct-Q4_K_M.gguf": "32B q4_K_M",
    "stories260K.gguf": "stories260K",
}
COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b"]


def load(paths, groups, meta):
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                name = os.path.basename(r["model_filename"])
                label = LABELS.get(name, os.path.splitext(name)[0])
                kind = "generation" if r["n_gen"] > 0 else "prompt"
                groups.setdefault((label, kind), []).extend(
                    r.get("samples_ts") or [r["avg_ts"]])
                meta[label] = (r.get("model_n_params", 0), r.get("model_size", 0))


def stats(samples):
    n = len(samples)
    return statistics.fmean(samples), (statistics.stdev(samples) if n > 1 else 0.0), n


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    outdir = sys.argv[1]
    sets, meta = [], {}
    for spec in sys.argv[2:]:
        if "=" not in spec:
            sys.exit("bad set spec %r; expected label=file[,file]" % spec)
        label, files = spec.split("=", 1)
        groups = {}
        load([p for p in files.split(",") if p], groups, meta)
        if not groups:
            sys.exit("no records for set %s" % label)
        sets.append((label, groups))
    os.makedirs(outdir, exist_ok=True)

    base_label, base = sets[0]
    order = sorted(meta, key=lambda le: (meta[le][0] or 0))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.6), sharey=True)
    for ax, (kind, title) in zip(axes, (("prompt", "Prompt processing (128 tokens)"),
                                        ("generation", "Token generation (32 tokens)"))):
        ys = list(range(len(order)))
        n = len(sets)
        h = 0.8 / n
        for i, (label, groups) in enumerate(sets):
            vals = []
            for le in order:
                v = groups.get((le, kind))
                vals.append(stats(v)[0] if v else float("nan"))
            ax.barh([y + (i - (n - 1) / 2.0) * h for y in ys], vals, h, label="%s threads" % label,
                    color=COLORS[i % len(COLORS)], edgecolor="black", linewidth=0.3)
        ax.set_yticks(ys)
        ax.set_yticklabels(order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("tokens/s", fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        ax.grid(True, axis="x", which="both", alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Thread count against every model file (depth 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "threads-throughput.svg"))
    plt.close(fig)

    others = [label for label, _ in sets[1:]]
    print("| model | test | " + " | ".join("%s" % l for l, _ in sets) +
          " | " + " | ".join("%s/%s" % (o, base_label) for o in others) + " |")
    print("|---|---|" + "---|" * len(sets) + "---|" * len(others))
    for le in order:
        for kind in ("prompt", "generation"):
            if (le, kind) not in base:
                continue
            bm = stats(base[(le, kind)])[0]
            cells = ["%.2f" % bm]
            ratios = []
            for label, groups in sets[1:]:
                v = groups.get((le, kind))
                if v:
                    m = stats(v)[0]
                    cells.append("%.2f" % m)
                    ratios.append("%.0f%%" % (100.0 * m / bm))
                else:
                    cells.append("-")
                    ratios.append("-")
            print("| %s | %s | %s | %s |" % (le, "prefill" if kind == "prompt" else "generation",
                                             " | ".join(cells), " | ".join(ratios)))


if __name__ == "__main__":
    main()

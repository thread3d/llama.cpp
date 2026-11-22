#!/usr/bin/env python3
"""Compare two sweep runs that differ only in thread count.

Reads two sets of `bench-macpro31.sh coverage` JSONL files (the "base" and the "alternate" thread
count) and draws one grouped bar chart per test, plus a table with the ratio. The point of the
chart is the shape of the scaling, so both panels share the same model order and use a log axis.

Usage: scripts/plot-macpro31-threads.py <out-dir> <base.jsonl> [base2.jsonl ...] -- <alt.jsonl> [...]
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


def load(paths):
    """-> {(label, kind): [samples]}, {label: (params, size)}"""
    groups, meta = {}, {}
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
    return groups, meta


def stats(samples):
    n = len(samples)
    return statistics.fmean(samples), (statistics.stdev(samples) if n > 1 else 0.0), n


def main():
    if "--" not in sys.argv or len(sys.argv) < 5:
        sys.exit(__doc__)
    split = sys.argv.index("--")
    outdir = sys.argv[1]
    base_paths = sys.argv[2:split]
    alt_paths = sys.argv[split + 1:]
    os.makedirs(outdir, exist_ok=True)

    base, meta = load(base_paths)
    alt, meta_alt = load(alt_paths)
    meta.update(meta_alt)
    if not base or not alt:
        sys.exit("need records in both sets")

    order = sorted(meta, key=lambda le: (meta[le][0] or 0))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharey=True)
    for ax, (kind, title) in zip(axes, (("prompt", "Prompt processing (128 tokens)"),
                                        ("generation", "Token generation (32 tokens)"))):
        ys = list(range(len(order)))
        h = 0.38
        b = [stats(base[(le, kind)])[0] if (le, kind) in base else float("nan") for le in order]
        a = [stats(alt[(le, kind)])[0] if (le, kind) in alt else float("nan") for le in order]
        ax.barh([y - h / 2 for y in ys], b, h, label="7 threads", color="#1f77b4",
                edgecolor="black", linewidth=0.4)
        ax.barh([y + h / 2 for y in ys], a, h, label="4 threads", color="#d95f02",
                edgecolor="black", linewidth=0.4)
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

    print("| model | 7t prefill | 4t prefill | 4t/7t | 7t gen | 4t gen | 4t/7t |")
    print("|---|---|---|---|---|---|---|")
    for le in order:
        row = [le]
        for kind in ("prompt", "generation"):
            if (le, kind) in base and (le, kind) in alt:
                bm = stats(base[(le, kind)])[0]
                am = stats(alt[(le, kind)])[0]
                row += ["%.2f" % bm, "%.2f" % am, "%.0f%%" % (100.0 * am / bm)]
            else:
                row += ["-", "-", "-"]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()

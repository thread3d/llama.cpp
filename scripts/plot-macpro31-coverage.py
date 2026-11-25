#!/usr/bin/env python3
"""Chart of every model file on macpro31, from `scripts/bench-macpro31.sh coverage`.

Reads the JSONL that the coverage phase writes: one `both` run per GGUF file in MODELS_DIR, so
the chart answers "how does each model on this machine actually run", including the duplicates
that differ only in quantization. Prompt processing and token generation are separate panels
because they differ by more than an order of magnitude.

Each bar is the mean of the recorded runs and the whisker is one sample standard deviation, i.e.
run-to-run spread rather than a confidence interval for the mean.

Usage: scripts/plot-macpro31-coverage.py <coverage.jsonl> [more.jsonl ...] <out-dir>
Requires matplotlib. Set MPLCONFIGDIR if HOME is not writable.
"""

import json
import os
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl-macpro31-cov"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Short labels; anything not listed falls back to the file name without its extension.
LABELS = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "Qwen2.5-0.5B q4_K_M",
    "qwen2.5-0.5b-instruct-q4_0.gguf": "Qwen2.5-0.5B q4_0",
    "qwen2.5-0.5b-instruct-q8_0.gguf": "Qwen2.5-0.5B q8_0",
    "qwen2.5-0.5b-instruct-fp16.gguf": "Qwen2.5-0.5B f16",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf": "Llama-3.2-1B q4_K_M",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "Qwen2.5-1.5B q4_K_M",
    "qwen2.5-1.5b-instruct-q4_0.gguf": "Qwen2.5-1.5B q4_0",
    "qwen2.5-3b-instruct-q4_k_m.gguf": "Qwen2.5-3B q4_K_M",
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf": "Mistral-7B q4_K_M",
    "Mistral-7B-Instruct-v0.3-Q4_0.gguf": "Mistral-7B q4_0",
    "Qwen3-8B-Q4_K_M.gguf": "Qwen3-8B q4_K_M",
    "stories260K.gguf": "stories260K (toy)",
    "Qwen2.5-14B-Instruct-Q4_K_M.gguf": "Qwen2.5-14B q4_K_M",
    "Qwen2.5-32B-Instruct-Q4_K_M.gguf": "Qwen2.5-32B q4_K_M",
    "Phi-3.5-mini-instruct-Q4_K_M.gguf": "Phi-3.5-mini",
    "granite-3.1-2b-instruct-Q4_K_M.gguf": "granite-3.1-2b",
    "gemma-2-2b-it-Q4_K_M.gguf": "gemma-2-2b-it",
    "OLMo-2-1124-7B-Instruct-Q4_K_M.gguf": "OLMo-2-7B",
    "aya-expanse-8b-Q4_K_M.gguf": "aya-expanse-8b",
    "Falcon3-3B-Instruct-Q4_K_M.gguf": "Falcon3-3B",
    "smollm2-1.7b-instruct-q4_k_m.gguf": "SmolLM2-1.7B",
    "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf": "R1-Distill-Qwen-1.5B",
    "internlm2_5-7b-chat-Q4_K_M.gguf": "internlm2.5-7b-chat",
    "Yi-1.5-6B-Chat-Q4_K_M.gguf": "Yi-1.5-6B-Chat",
    "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf": "Mistral-Nemo-12B",
    "qwen2.5-coder-7b-instruct-q4_k_m.gguf": "Qwen2.5-Coder-7B",
    "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf": "DeepSeek-Coder-V2-Lite",
    "Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf": "Qwen2.5-Math-7B",
    "gemma-2-9b-it-Q4_K_M.gguf": "gemma-2-9b-it",
    "Phi-3-medium-4k-instruct-Q4_K_M.gguf": "Phi-3-medium-14B",
}


def load(path, groups=None, meta=None):
    """-> {(label, kind): [samples]}, {label: (n_params, size)}"""
    groups = {} if groups is None else groups
    meta = {} if meta is None else meta
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = os.path.basename(rec["model_filename"])
            label = LABELS.get(name, os.path.splitext(name)[0])
            kind = "generation" if rec["n_gen"] > 0 else "prompt"
            groups.setdefault((label, kind), []).extend(rec.get("samples_ts") or [rec["avg_ts"]])
            meta[label] = (rec.get("model_n_params", 0), rec.get("model_size", 0), name)
    return groups, meta


def stats(samples):
    n = len(samples)
    mean = statistics.fmean(samples)
    return mean, (statistics.stdev(samples) if n > 1 else 0.0), n


def panel(ax, groups, meta, labels, kind, title):
    xs, means, errs = [], [], []
    for i, label in enumerate(labels):
        samples = groups.get((label, kind))
        if not samples:
            continue
        mean, sd, _n = stats(samples)
        xs.append(i)
        means.append(mean)
        errs.append(sd)
    ax.barh(xs, means, xerr=errs, capsize=3, color="#1f77b4" if kind == "prompt" else "#d95f02",
            edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.9})
    ax.set_yticks(xs)
    ax.set_yticklabels([labels[i] for i in xs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("tokens/s", fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.grid(True, axis="x", which="both", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    paths, outdir = sys.argv[1:-1], sys.argv[-1]
    os.makedirs(outdir, exist_ok=True)

    groups, meta = {}, {}
    for path in paths:
        load(path, groups, meta)
    if not groups:
        sys.exit("no coverage records found")

    # order smallest model first; the toy model has almost no parameters and belongs at the top
    order = sorted(meta, key=lambda le: (meta[le][0] or 0, meta[le][2]))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    panel(axes[0], groups, meta, order, "prompt", "Prompt processing (128 tokens)")
    panel(axes[1], groups, meta, order, "generation", "Token generation (32 tokens)")
    fig.suptitle("Every model file on macpro31, 7 threads, depth 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "coverage-throughput.svg"))
    plt.close(fig)

    for label in order:
        for kind in ("prompt", "generation"):
            samples = groups.get((label, kind))
            if not samples:
                continue
            mean, sd, n = stats(samples)
            print(f"{label:24s} {kind:10s} n={n} mean={mean:8.2f} sd={sd:6.2f}")


if __name__ == "__main__":
    main()

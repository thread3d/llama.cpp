#!/usr/bin/env python3
"""One run, two backends: ROCm and Vulkan in the same process.

A build with both GGML_HIP and GGML_VULKAN carries six device entries on this
machine, and llama.cpp will happily put half the layers on a ROCm device and half
on a Vulkan one. Nothing in the tree is tested for that: the backend tests take
one backend at a time. This script measures whether it works and what it costs.

The build this needs is separate from the two single backend builds:

    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_COMPILER=clang \\
    cmake -S . -B build-mixed -G Ninja -DGGML_HIP=ON -DGGML_VULKAN=ON \\
        -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++

Note the two selectors, which are not the same in the two tools:

    llama-bench        -dev 'ROCm0/Vulkan2'   a comma there starts a second test
    llama-perplexity   -dev  ROCm0,Vulkan2    the usual common_params list

    ./mixed.py
"""

import csv
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, PARENT, REPO

BIN = os.path.join(REPO, "build-mixed", "bin")
MODEL = "Qwen3-0.6B-Q4_K_M.gguf"
CORPUS = PARENT + "/logs/corpus.txt"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mixed.csv")

# keep the Vega II off the HIP side: it has no device code in this build
ENV = dict(os.environ)
ENV["HIP_VISIBLE_DEVICES"] = "0,1"

CONFIGS = [
    ("rocm only",   "ROCm0",           "ROCm0"),
    ("vulkan only", "Vulkan2",         "Vulkan2"),
    ("mixed A",     "ROCm0/Vulkan2",   "ROCm0,Vulkan2"),
    ("mixed B",     "Vulkan1/ROCm1",   "Vulkan1,ROCm1"),
]

RE_KLD = re.compile(r"Mean\s+KLD:\s*(-?[0-9.]+)\s*[\u00b1+/-]+\s*([0-9.]+)")
RE_TOP = re.compile(r"Same\s+top\s+p:\s*(-?[0-9.]+)\s*[\u00b1+/-]+\s*([0-9.]+)")


def bench(bench_dev):
    argv = [os.path.join(BIN, "llama-bench"), "-m", os.path.join(MODELS, MODEL),
            "-ngl", "99", "-dev", bench_dev, "-p", "64", "-n", "16", "-r", "3",
            "-o", "csv"]
    p = subprocess.run(argv, env=ENV, capture_output=True, text=True)
    rows = list(csv.DictReader(p.stdout.splitlines()))
    pp = next((float(r["avg_ts"]) for r in rows if r["n_gen"] == "0"), float("nan"))
    tg = next((float(r["avg_ts"]) for r in rows if r["n_gen"] != "0"), float("nan"))
    return pp, tg


def kld(ppl_dev, ref):
    argv = [os.path.join(BIN, "llama-perplexity"), "-m", os.path.join(MODELS, MODEL),
            "-f", CORPUS, "-c", "512", "--chunks", "8", "-ngl", "99",
            "-dev", ppl_dev, "--kl-divergence", "--kl-divergence-base", ref]
    p = subprocess.run(argv, env=ENV, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m, t = RE_KLD.findall(out), RE_TOP.findall(out)
    return (float(m[-1][0]) if m else float("nan"),
            float(t[-1][0]) if t else float("nan"))


def main():
    ref = PARENT + "/logs/agreement/ref-mixed.bin"
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    argv = [os.path.join(BIN, "llama-perplexity"), "-m", os.path.join(MODELS, MODEL),
            "-f", CORPUS, "-c", "512", "--chunks", "8", "-dev", "none",
            "--save-all-logits", ref]
    p = subprocess.run(argv, env=ENV, capture_output=True, text=True)
    if not os.path.exists(ref):
        raise SystemExit("reference logits were not written: " + p.stderr[-400:])
    print(f"reference logits: {os.path.getsize(ref)/1e6:.0f} MB")

    rows = []
    for label, bench_dev, ppl_dev in CONFIGS:
        pp, tg = bench(bench_dev)
        k, top = kld(ppl_dev, ref)
        rows.append({"config": label, "bench_dev": bench_dev, "ppl_dev": ppl_dev,
                     "pp_ts": round(pp, 1), "tg_ts": round(tg, 1),
                     "mean_kld": k, "same_top_p": top})
        print(f"{label:12s} pp={pp:8.1f} tg={tg:7.1f} kld={k:.6f} top_p={top:.2f}%",
              flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.remove(ref)
    print(f"wrote {OUT}")
    print("the six rows of `llama-bench --list-devices` are what makes this possible")


if __name__ == "__main__":
    main()

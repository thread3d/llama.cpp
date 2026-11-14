#!/usr/bin/env python3
"""Do the devices compute the same numbers?

test-llama-context asks whether a context survives a save and a restore on one
device. This asks the other question: given the same text, does each die and the
pair of dies produce the same distribution the CPU produces?

The measurement is llama-perplexity's KL divergence mode. One topology saves its
per token logits, then every other topology scores the same corpus against that
file and reports

    Mean KLD    mean Kullback-Leibler divergence of its distribution from the reference
    RMS dp      root mean square difference of the token probabilities
    Same top p  share of positions where both pick the same most likely token

Zero KLD would mean identical arithmetic, which no two kernels provide. What
matters is that all GPU topologies sit at the same small KLD: that says the two
dies together are as faithful as one die, which is exactly what a bad layer
split breaks.

    ./agreement.py --model Qwen3-0.6B-Q4_K_M.gguf --ctx 512 --chunks 2
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANSI, MODELS, ORDER, PARENT, launch

CORPUS = PARENT + "/logs/corpus.txt"
OUT = PARENT + "/logs/agreement"

# the summary block prints "label: value +/- error"; keep both halves
def _stat(label):
    return re.compile(label + r"\s*:\s*(-?[0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)")

RE_KLD = _stat(r"Mean\s+KLD")
RE_RMS = _stat(r"RMS\s+Δp")
RE_TOP = _stat(r"Same\s+top\s+p")
RE_PPL_RATIO = _stat(r"Mean\s+PPL\(Q\)/PPL\(base\)")
RE_PPL = re.compile(r"Final estimate: PPL = ([0-9.eE+\-]+)")


def measure(topology, model, ctx, chunks, ref, timeout):
    extra = [
        "-m", os.path.join(MODELS, model),
        "-f", CORPUS,
        "-c", str(ctx),
        "--chunks", str(chunks),
    ]
    if ref is None:
        extra += ["--save-all-logits", os.path.join(OUT, "ref-logits.bin")]
    else:
        extra += ["--kl-divergence", "--kl-divergence-base", ref]

    argv, env = launch(topology, "llama-perplexity", extra)
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
        out, rc = ANSI.sub("", p.stdout + p.stderr), p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "", -99
    dt = time.monotonic() - t0

    def grab(rx):
        """(value, error) of the last match, or (nan, nan)."""
        m = rx.findall(out)
        if not m:
            return float("nan"), float("nan")
        v, e = m[-1] if isinstance(m[-1], tuple) else (m[-1], "nan")
        return float(v), float(e)

    kld, kld_e = grab(RE_KLD)
    rms, rms_e = grab(RE_RMS)
    top, top_e = grab(RE_TOP)
    ratio, ratio_e = grab(RE_PPL_RATIO)
    ppl = grab(RE_PPL)[0]

    return {
        "topology": topology,
        "role": "reference" if ref is None else "scored",
        "ppl": ppl,
        "ppl_ratio": ratio,
        "ppl_ratio_err": ratio_e,
        "mean_kld": kld,
        "mean_kld_err": kld_e,
        "rms_dp": rms,
        "rms_dp_err": rms_e,
        "same_top_p": top,
        "same_top_p_err": top_e,
        "exit": rc,
        "seconds": round(dt, 1),
        "status": "ok" if rc == 0 else ("timeout" if rc == -99 else "error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-0.6B-Q4_K_M.gguf")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--reference", default="cpu")
    ap.add_argument("--topologies", default="all")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if not os.path.exists(CORPUS):
        raise SystemExit(f"missing corpus: {CORPUS}")
    os.makedirs(OUT, exist_ok=True)
    ref_path = os.path.join(OUT, "ref-logits.bin")

    topos = ORDER if args.topologies == "all" else args.topologies.split(",")
    if args.reference not in topos:
        topos = [args.reference] + topos

    rows = []
    for t in topos:
        if t == args.reference:
            if os.path.exists(ref_path):
                os.remove(ref_path)
            r = measure(t, args.model, args.ctx, args.chunks, None, args.timeout)
            if r["status"] != "ok" or not os.path.exists(ref_path):
                raise SystemExit(f"reference run on {t} failed: {r}")
            size = os.path.getsize(ref_path) / 1e6
            print(f'reference {t:10s} ppl={r["ppl"]:.6f} logits={size:.0f} MB '
                  f'{r["seconds"]:.1f}s', flush=True)
        else:
            r = measure(t, args.model, args.ctx, args.chunks, ref_path, args.timeout)
            print(f'scored    {t:10s} {r["status"]:7s} '
                  f'kld={r["mean_kld"]:.5f}+/-{r["mean_kld_err"]:.5f} '
                  f'rms_dp={r["rms_dp"]:.3f}+/-{r["rms_dp_err"]:.3f}% '
                  f'top_p={r["same_top_p"]:.2f}+/-{r["same_top_p_err"]:.2f}% '
                  f'{r["seconds"]:6.1f}s', flush=True)
        rows.append(r)

    path = os.path.join(OUT, "kld.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")
    print(f"reference logits kept at {ref_path} "
          f"({os.path.getsize(ref_path)/1e6:.0f} MB), a derived artifact")


if __name__ == "__main__":
    main()

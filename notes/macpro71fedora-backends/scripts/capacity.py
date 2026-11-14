#!/usr/bin/env python3
#!/usr/bin/env python3
"""The largest context each topology will accept, and why that is not capacity.

This probe creates a context of size N and runs one token. It measures what
llama.cpp will *accept*, which on this tree is not what it can *hold*: the KV
cache grows on demand, so every topology here accepts 1048576 tokens for an
18.5 GiB model on a single 32 GiB die without allocating anything near that. Run
it to confirm that for a new build; the number it prints is an acceptance limit,
not a capacity limit.

Real capacity is therefore a runtime limit: what it takes to fail is to fill the
cache. That is what `llama-bench -d` does, and it is why the depth sweeps in this
study run to 32768 tokens and stop there rather than to a hard wall.

    ./capacity.py --model Qwen3-30B-A3B-Q4_K_M.gguf
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

OUT = PARENT + "/logs/capacity"
RE_OOM = re.compile(r"(failed to allocate|out of memory|not enough|failed to create|error loading model)[^\n]*", re.I)


def try_ctx(topology, model, ctx, timeout):
    """Does a context of this size allocate and run one token?"""
    argv, env = launch(topology, "llama-cli", [
        "-m", os.path.join(MODELS, model),
        "-c", str(ctx),
        "-p", "hi",
        "-n", "1",
        "--no-warmup",
        "--no-display-prompt",
    ])
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
        out, rc = ANSI.sub("", p.stdout + p.stderr), p.returncode
    except subprocess.TimeoutExpired:
        return False, "timeout", time.monotonic() - t0
    dt = time.monotonic() - t0

    if rc == 0:
        return True, "", dt
    m = RE_OOM.search(out)
    reason = m.group(0)[:90] if m else f"exit {rc}"
    return False, reason, dt


def probe(topology, model, lo, hi, timeout):
    """Largest tested size that works: double up from lo, then bisect below hi."""
    size = lo
    last_ok = 0
    while size <= hi:
        ok, why, _ = try_ctx(topology, model, size, timeout)
        print(f"    {topology:10s} ctx={size:8d} {'ok' if ok else 'fail: ' + why}", flush=True)
        if not ok:
            break
        last_ok = size
        size *= 2

    if last_ok == 0:
        return 0, "even the smallest size failed"

    # bisect between the largest working size and the first failing one
    low, high = last_ok, min(size, hi)
    while high - low > 2048:
        mid = (low + high) // 2
        ok, why, _ = try_ctx(topology, model, mid, timeout)
        print(f"    {topology:10s} ctx={mid:8d} {'ok' if ok else 'fail: ' + why}", flush=True)
        if ok:
            low = mid
        else:
            high = mid
    return low, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-30B-A3B-Q4_K_M.gguf")
    ap.add_argument("--lo", type=int, default=4096)
    ap.add_argument("--hi", type=int, default=1048576)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--topologies", default="all")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    topos = ORDER if args.topologies == "all" else args.topologies.split(",")

    rows = []
    for t in topos:
        print(f"{t}:", flush=True)
        best, why = probe(t, args.model, args.lo, args.hi, args.timeout)
        rows.append({"topology": t, "model": args.model, "max_ctx": best, "note": why})
        print(f"  -> {t:10s} max ctx {best}", flush=True)

    path = os.path.join(OUT, "capacity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topology", "model", "max_ctx", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

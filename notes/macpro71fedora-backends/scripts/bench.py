#!/usr/bin/env python3
"""Performance sweep: llama-bench over topology x model x context depth.

llama-bench fills the KV cache to `-d` tokens before it measures, so `-d` is the
context size the run actually works at: at depth 32768 a token generation step
attends over 32768 rows instead of 128. Two tests are taken per depth, the
prompt processing of 512 tokens (pp512) and the generation of 128 tokens (tg128).

    ./bench.py --label v1 --csv v1-qwen06.csv --model Qwen3-0.6B-Q4_K_M.gguf \
        --depths 0,1024,4096,16384 --reps 3
"""

import argparse
import csv
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, ORDER, launch

FIELDS = [
    "topology", "model", "depth", "rep", "n_prompt", "n_gen",
    "avg_ts", "stddev_ts", "test_time_s", "exit", "status",
]


def parse_csv_output(text):
    """llama-bench's own csv rows, as dicts."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    header = next(csv.reader([lines[0]]))
    rows = []
    for line in lines[1:]:
        vals = next(csv.reader([line]))
        if len(vals) != len(header):
            continue
        rows.append(dict(zip(header, vals)))
    return rows


def run_cell(topology, model, depth, reps, timeout):
    extra = [
        "-m", os.path.join(MODELS, model),
        "-p", "512", "-n", "128",
        "-d", str(depth),
        "-r", str(reps),
        "-o", "csv",
    ]
    argv, env = launch(topology, "llama-bench", extra)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, env=env, capture_output=True, text=True,
                              timeout=timeout)
        out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        rc = -99
    dt = time.monotonic() - t0

    rows = []
    for r in parse_csv_output(out):
        n_prompt = int(r["n_prompt"])
        n_gen = int(r["n_gen"])
        # llama-bench reports one row per test: pp has n_gen 0, tg has n_prompt 0
        kind = "pp" if n_gen == 0 else "tg"
        rows.append({
            "topology": topology,
            "model": model,
            "kind": kind,
            "depth": depth,
            "rep": r.get("rep", ""),
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "avg_ts": float(r["avg_ts"]),
            "stddev_ts": float(r["stddev_ts"]),
            # test_time is an ISO timestamp in this version; avg_ns is the
            # wall clock of one evaluation, which is what a chart wants
            "test_time_s": float(r["avg_ns"]) / 1e9,
            "seconds": round(dt, 2),
            "exit": rc,
            "status": "ok" if rc == 0 else ("timeout" if rc == -99 else "error"),
            "flash_attn": r.get("flash_attn", ""),
            "split_mode": r.get("split_mode", ""),
            "n_gpu_layers": r.get("n_gpu_layers", ""),
            "devices": r.get("devices", ""),
        })
    if not rows:
        rows.append({
            "topology": topology, "model": model, "kind": "", "depth": depth,
            "rep": "", "n_prompt": "", "n_gen": "", "avg_ts": float("nan"),
            "stddev_ts": float("nan"), "test_time_s": float("nan"),
            "seconds": round(dt, 2), "exit": rc,
            "status": "timeout" if rc == -99 else "error",
            "flash_attn": "", "split_mode": "", "n_gpu_layers": "", "devices": "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="Qwen3-0.6B-Q4_K_M.gguf")
    ap.add_argument("--depths", default="0,1024,4096,16384")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--topologies", default="all")
    ap.add_argument("--timeout", type=int, default=1800, help="per cell seconds")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    topos = ORDER if args.topologies == "all" else args.topologies.split(",")

    rows = []
    for topology in topos:
        for depth in depths:
            cell = run_cell(topology, args.model, depth, args.reps, args.timeout)
            for r in cell:
                r["label"] = args.label
                rows.append(r)
                if r["status"] == "ok":
                    print(f'{topology:10s} d={depth:6d} {r["kind"]:2s} '
                          f'{r["avg_ts"]:9.1f} t/s (sd {r["stddev_ts"]:6.2f}) '
                          f'{r["seconds"]:6.1f}s', flush=True)
                else:
                    print(f'{topology:10s} d={depth:6d} {r["status"]:7s} exit={r["exit"]}',
                          flush=True)
            write_csv(args.csv, rows)

    write_csv(args.csv, rows)
    print(f"wrote {args.csv} ({len(rows)} rows)")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(FIELDS)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())

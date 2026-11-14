#!/usr/bin/env python3
"""Correctness sweep: test-llama-context over topology x context size x prompt size.

test-llama-context saves a KV state, restores it and checks that the logits of
the restored run match the run that saved it. It runs 18 cases per model (13
state cases plus 5 that need a recurrent model), and the quantity of interest is
the NMSE each case reports against its own saved baseline.

The matrix answers one question: is a restored context reproducible on every
device of this machine, at every context size, and does adding the second die
change the answer.

    ./matrix.py --label v1 --csv v1-scaled.csv --prompt-mode scaled
    ./matrix.py --label v1 --csv v1-builtin.csv --prompt-mode builtin
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANSI, MODELS, ORDER, REPO, TOPOLOGIES, WORK, launch

MODELS_SRC = REPO + "/build-vulkan/tests/test-models"

# A curated slice of the shipped test models, chosen so that every case of the
# test can run somewhere: plain and sliding-window dense attention, MLA, four
# kinds of MoE, and recurrent/hybrid models that are the only ones able to run
# the rollback, replay and shared cases. The remaining models SKIP those cases.
DEFAULT_MODELS = [
    "arcee-dense",       # plain dense attention
    "llama-dense",       # plain dense, the reference architecture
    "qwen3-dense",       # dense, GQA, the model the perf sweep uses
    "gemma3-dense",      # sliding window attention
    "deepseek2-moe",     # MLA + MoE
    "deepseek4-moe",     # the only family that runs the comp_padding case
    "arctic-moe",        # fine grained MoE
    "glm4moe-moe",       # MoE with dense prefix
    "qwen3next-moe",     # recurrent + MoE hybrid
    "mamba2-dense",      # recurrent, runs rollback/replay/shared
]

WORDS = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
RE_NMSE = re.compile(r"generate_tokens_compare: step \d+ nmse = ([0-9eE+\-.]+)")
RE_PROMPT = re.compile(r"the input prompt is (\d+) tokens")


def make_prompt(n_tokens):
    """A prompt of about n_tokens tokens.

    These test models split a word into ~1.37 tokens, measured from the count the
    test prints. Dividing by a slightly larger number keeps the realised prompt
    close to the target from below, which the batch below then covers.
    """
    n_words = max(8, int(n_tokens / 1.4))
    words = (WORDS * (n_words // len(WORDS) + 1))[:n_words]
    return " ".join(words)


def prepare_models(model_list):
    d = os.path.join(WORK, "models")
    os.makedirs(d, exist_ok=True)
    for m in model_list:
        dst = os.path.join(d, m)
        src = os.path.join(MODELS_SRC, m)
        if not os.path.exists(src):
            raise SystemExit(f"missing test model: {src}")
        if not os.path.exists(dst):
            os.symlink(src, dst)
    return d


def parse_table(text):
    """Case verdict counts and the names of the failing models."""
    pass_n = fail_n = skip_n = 0
    failed = []
    for line in text.splitlines():
        cell = line.split()
        if not cell or not cell[0].endswith(".gguf"):
            continue
        verdicts = [c for c in cell[1:] if c in ("OK", "FAIL", "SKIP")]
        if not verdicts:
            continue
        pass_n += verdicts.count("OK")
        fail_n += verdicts.count("FAIL")
        skip_n += verdicts.count("SKIP")
        if "FAIL" in verdicts:
            failed.append(cell[0])
    return pass_n, fail_n, skip_n, failed


def run_once(topology, ctx, prompt_ratio, models_dir, logs_root, jobs):
    logs = os.path.join(logs_root, "logs")
    shutil.rmtree(logs, ignore_errors=True)

    extra = ["--models", models_dir, "-j", str(jobs), "-c", str(ctx), "--logs", logs]

    if prompt_ratio > 0:
        n_tokens = int(ctx * prompt_ratio)
        # With --logs the test decodes the whole prompt as one batch before saving
        # the state, so n_batch has to cover the realised token count, which runs
        # above the request. The margin is proportional for that reason.
        n_batch = min(ctx, int(n_tokens * 1.25) + 512)
        extra += ["-b", str(n_batch), "-p", make_prompt(n_tokens)]
        prompt_mode = "scaled"
    else:
        # the built in prompt: 100 random tokens whatever the context is
        extra += ["-b", str(min(ctx, 128))]
        prompt_mode = "builtin"

    argv, env = launch(topology, "test-llama-context", extra)

    t0 = time.monotonic()
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    dt = time.monotonic() - t0

    out = ANSI.sub("", proc.stdout)
    pass_n, fail_n, skip_n, failed = parse_table(out)

    nmse = []
    n_prompt = 0
    if os.path.isdir(logs):
        for name in os.listdir(logs):
            with open(os.path.join(logs, name), "r", errors="replace") as f:
                for line in f:
                    line = ANSI.sub("", line)
                    m = RE_NMSE.search(line)
                    if m:
                        nmse.append(float(m.group(1)))
                    m = RE_PROMPT.search(line)
                    if m:
                        n_prompt = max(n_prompt, int(m.group(1)))

    if proc.returncode < 0:
        status = "signal"          # killed by a signal, e.g. assert abort or segfault
    elif proc.returncode > 0:
        status = "error"
    elif fail_n:
        status = "fail"
    elif pass_n == 0:
        status = "empty"           # nothing ran, so a pass is not evidence
    else:
        status = "pass"

    return {
        "topology": topology,
        "ctx": ctx,
        "prompt_mode": prompt_mode,
        "prompt_tokens": n_prompt,
        "exit": proc.returncode,
        "status": status,
        "seconds": round(dt, 3),
        "cases_pass": pass_n,
        "cases_fail": fail_n,
        "cases_skip": skip_n,
        "n_nmse": len(nmse),
        "max_nmse": max(nmse) if nmse else float("nan"),
        "n_nmse_over_1e-5": sum(1 for v in nmse if v > 1e-5),
        "failed_models": ";".join(failed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ctxs", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--topologies", default="all")
    ap.add_argument("--prompt-mode", default="scaled", choices=["scaled", "builtin"])
    ap.add_argument("--resume", action="store_true", help="skip rows already in the csv")
    ap.add_argument("--models", default=",".join(m + ".gguf" for m in DEFAULT_MODELS),
                    help="comma separated test model file names")
    ap.add_argument("--jobs", "-j", type=int, default=4, help="parallel workers")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    models_dir = prepare_models(args.models.split(","))
    logs_root = os.path.join(WORK, "run-logs")
    os.makedirs(logs_root, exist_ok=True)

    ctxs = [int(c) for c in args.ctxs.split(",")]
    topos = ORDER if args.topologies == "all" else args.topologies.split(",")
    ratio = 0.5 if args.prompt_mode == "scaled" else 0.0

    rows = []
    done = set()
    if args.resume and os.path.exists(args.csv):
        with open(args.csv) as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((r["topology"], int(r["ctx"]), r["prompt_mode"]))

    total = len(topos) * len(ctxs) * args.reps - len(done)
    n = 0
    for topology in topos:
        for ctx in ctxs:
            for rep in range(1, args.reps + 1):
                key = (topology, ctx, args.prompt_mode)
                n += 1
                if args.resume and key in done:
                    continue
                row = run_once(topology, ctx, ratio, models_dir, logs_root, args.jobs)
                row["label"] = args.label
                row["rep"] = rep
                rows.append(row)
                print(
                    f"{topology:10s} ctx={ctx:6d} rep={rep} prompt={row['prompt_tokens']:6d} "
                    f"pass={row['cases_pass']:4d} fail={row['cases_fail']:3d} "
                    f"max_nmse={row['max_nmse']:.3e} {row['seconds']:7.2f}s "
                    f"{row['failed_models'][:40]}",
                    flush=True,
                )
                write_csv(args.csv, rows)

    write_csv(args.csv, rows)
    print(f"wrote {args.csv} ({len(rows)} rows)")


def write_csv(path, rows):
    if not rows:
        return
    # keep the first row's key order, adding any later keys at the end
    fields = list(rows[0].keys())
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

#!/usr/bin/env python3
"""Sweep test-llama-context over device topology, context size and repetitions.

Each context size C runs with a prompt of about C/2 tokens, so that a larger
context also means more occupied cells, a larger KV state and more rows in the
attention, not just a larger allocation.

Writes one CSV row per run with the case verdicts, the worst NMSE the run
produced and the wall clock, so a later step can aggregate them into charts with
error bars.

    ./ctxsweep.py --label post --csv post.csv [--ctxs 128,256] [--reps 5]
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

REPO = "/home/threaded/projects/Llama.cpp/llama.cpp"
BIN = REPO + "/build-vulkan/bin/test-llama-context"
MODELS_SRC = REPO + "/build-vulkan/tests/test-models"
WORK = "/home/threaded/projects/Llama.cpp/logs/ctx-sweep"

# topology -> GGML_VK_VISIBLE_DEVICES, in the Vulkan backend's own numbering
# topology -> (GGML_VK_VISIBLE_DEVICES, extra command line arguments)
TOPOLOGIES = {
    "die1":     ("1",   []),             # W6800X die 1 alone
    "die2":     ("2",   []),             # W6800X die 2 alone
    "both":     ("1,2", []),             # both dies, default layer split
    "both-row": ("1,2", ["-sm", "row"]), # both dies, row split
}

MODELS = [
    "afmoe-moe.gguf",
    "arcee-dense.gguf",
    "arctic-moe.gguf",
    "baichuan-dense.gguf",
    "bailingmoe-moe.gguf",
    "beam-moe.gguf",
    "bloom-dense.gguf",
    "chatglm-dense.gguf",
]

WORDS = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE_NMSE = re.compile(r"generate_tokens_compare: step \d+ nmse = ([0-9eE+\-.]+)")
RE_PROMPT = re.compile(r"the input prompt is (\d+) tokens")


def make_prompt(n_tokens):
    """A prompt of roughly n_tokens tokens.

    These test models tokenize a word into about 1.3 tokens, measured with the
    tokenizer the test itself uses."""
    n_words = max(8, int(n_tokens/1.3))
    words = (WORDS * (n_words // len(WORDS) + 1))[:n_words]
    return " ".join(words)


def prepare_models():
    d = os.path.join(WORK, "models")
    os.makedirs(d, exist_ok=True)
    for m in MODELS:
        dst = os.path.join(d, m)
        if not os.path.exists(dst):
            os.symlink(os.path.join(MODELS_SRC, m), dst)
    return d


def parse_table(text):
    """Case verdict counts and failing model names, from the result table."""
    pass_n = fail_n = skip_n = 0
    failed_models = []

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
            failed_models.append(cell[0])

    return pass_n, fail_n, skip_n, failed_models


def run_once(topology, ctx, rep, models_dir, bin_path, prompt_ratio):
    logs = os.path.join(WORK, "logs")
    shutil.rmtree(logs, ignore_errors=True)

    env = dict(os.environ)
    devices, extra = TOPOLOGIES[topology]
    env["GGML_VK_VISIBLE_DEVICES"] = devices

    cmd = [bin_path, "--models", models_dir, "-j", "4", "-c", str(ctx)] + extra

    if prompt_ratio > 0:
        # a prompt that grows with the context: more occupied cells, a larger KV state
        n_tokens = int(ctx*prompt_ratio)
        prompt   = make_prompt(n_tokens)
        # the test decodes the prompt in one batch, so n_batch has to cover it.
        # the tokenizer returns slightly more tokens than requested, hence the margin
        cmd += ["-b", str(min(ctx, n_tokens + 256)), "-p", prompt]
        prompt_mode = "scaled"
    else:
        # the built in prompt of the test: 100 random tokens, whatever the context is
        cmd += ["-b", str(min(ctx, 128))]
        prompt_mode = "builtin"

    cmd += ["--logs", logs]

    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.monotonic() - t0

    out = ANSI.sub("", proc.stdout)
    pass_n, fail_n, skip_n, failed_models = parse_table(out)

    nmse = []
    n_prompt = 0
    for name in os.listdir(logs) if os.path.isdir(logs) else []:
        with open(os.path.join(logs, name), "r", errors="replace") as f:
            for line in f:
                line = ANSI.sub("", line)
                m = RE_NMSE.search(line)
                if m:
                    nmse.append(float(m.group(1)))
                m = RE_PROMPT.search(line)
                if m:
                    n_prompt = max(n_prompt, int(m.group(1)))

    return {
        "topology": topology,
        "ctx": ctx,
        "prompt_mode": prompt_mode,
        "rep": rep,
        "prompt_tokens": n_prompt,
        "exit": proc.returncode,
        "seconds": round(dt, 3),
        "cases_pass": pass_n,
        "cases_fail": fail_n,
        "cases_skip": skip_n,
        "n_nmse": len(nmse),
        "max_nmse": max(nmse) if nmse else float("nan"),
        "n_nmse_over": sum(1 for v in nmse if v > 1e-5),
        "failed_models": ";".join(failed_models),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ctxs", default="128,256,512,1024,2048,4096")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--topologies", default="die1,die2,both")
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--prompt-ratio", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    models_dir = prepare_models()

    ctxs = [int(c) for c in args.ctxs.split(",")]
    topos = args.topologies.split(",")

    rows = []
    total = len(topos) * len(ctxs) * args.reps
    n = 0

    for topology in topos:
        for ctx in ctxs:
            for rep in range(1, args.reps + 1):
                n += 1
                row = run_once(topology, ctx, rep, models_dir, args.bin, args.prompt_ratio)
                row["label"] = args.label
                rows.append(row)
                print(f"[{n}/{total}] {args.label:4s} {topology:5s} ctx={ctx:5d} rep={rep} "
                      f"prompt={row['prompt_tokens']:5d} pass={row['cases_pass']:3d} "
                      f"fail={row['cases_fail']:3d} max_nmse={row['max_nmse']:.3e} "
                      f"{row['seconds']:6.2f}s", flush=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())

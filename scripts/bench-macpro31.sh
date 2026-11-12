#!/usr/bin/env bash
# Collect throughput statistics for the macpro31 CPU build of llama.cpp.
#
# Each repetition is a separate llama-bench process, and the whole list of runs is shuffled with a
# fixed seed before it starts. Drift on the machine (thermals, other load) therefore spreads across
# every model and depth instead of biasing whichever condition ran first. The shuffled plan is
# written next to the results, so a run can be replayed exactly.
#
# llama-bench emits two records per run: prompt processing (n_gen == 0) and token generation
# (n_gen > 0), each with its own samples_ts. scripts/plot-macpro31.py pools them.
#
# Note: llama-bench builds the depth context once for each of those two tests, so wall time grows
# with roughly twice the depth prefill cost.
#
# Usage: scripts/bench-macpro31.sh [context|quant|all]
#   context  three models, depths 0/256/1024
#   quant    one model at four quantization levels, depth 0
#
# Environment: MODELS_DIR, OUT, SEED, REPS, THREADS, PP, TG

set -euo pipefail

export PATH="/opt/local/bin:/usr/local/bin:$PATH"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

BENCH=$ROOT/build-macpro31-cpu/bin/llama-bench
MODELS_DIR=${MODELS_DIR:-$HOME/models}
OUT=${OUT:-$HOME/bench-macpro31.jsonl}
SEED=${SEED:-20261112}
REPS=${REPS:-4}
THREADS=${THREADS:-8}
PP=${PP:-128}
TG=${TG:-32}

[ -x "$BENCH" ] || { echo "missing $BENCH; run scripts/build-macpro31.sh first" >&2; exit 1; }

# plan <phase> -> "<phase> <model> <depth> <rep>" lines in shuffled order
plan() {
    python3 - "$SEED" "$REPS" "$1" <<'PY'
import random
import sys

seed, reps, phase = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

context_runs = [(m, d) for m in (
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
) for d in (0, 256, 1024)]

quant_runs = [(m, 0) for m in (
    "qwen2.5-0.5b-instruct-q4_0.gguf",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen2.5-0.5b-instruct-q8_0.gguf",
    "qwen2.5-0.5b-instruct-fp16.gguf",
)]

runs = []
if phase in ("context", "all"):
    runs += [("context", m, d) for m, d in context_runs]
if phase in ("quant", "all"):
    runs += [("quant", m, d) for m, d in quant_runs]

work = [(*run, rep) for run in runs for rep in range(reps)]
random.Random(seed).shuffle(work)
for ph, model, depth, rep in work:
    print(ph, model, depth, rep)
PY
}

run_phase() {
    local phase=$1 plan_file=$2 ctx_out=$3 quant_out=$4
    local total n=0
    plan "$phase" > "$plan_file"
    total=$(wc -l < "$plan_file" | tr -d ' ')
    echo "# phase=$phase seed=$SEED reps=$REPS threads=$THREADS pp=$PP tg=$TG runs=$total start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    : > "$ctx_out"
    : > "$quant_out"
    while read -r ph model depth rep; do
        n=$((n + 1))
        printf '[%d/%d] %s %s depth=%s rep=%s\n' "$n" "$total" "$ph" "$model" "$depth" "$rep" >&2
        if [ "$ph" = context ]; then out=$ctx_out; else out=$quant_out; fi
        "$BENCH" -m "$MODELS_DIR/$model" -p "$PP" -n "$TG" -d "$depth" \
            -t "$THREADS" -r 1 -o jsonl >> "$out" 2>> "${OUT%.jsonl}.log"
    done < "$plan_file"
    echo "# done=$(date -u +%Y-%m-%dT%H:%M:%SZ) context=$(wc -l < "$ctx_out" | tr -d ' ') quant=$(wc -l < "$quant_out" | tr -d ' ')" >&2
}

BASE=${OUT%.jsonl}

case "${1:-all}" in
    context) run_phase context "${BASE}-plan.tsv" "${BASE}-context.jsonl" /dev/null ;;
    quant)   run_phase quant   "${BASE}-plan.tsv" /dev/null "${BASE}-quant.jsonl" ;;
    all)     run_phase all     "${BASE}-plan.tsv" "${BASE}-context.jsonl" "${BASE}-quant.jsonl" ;;
    *)       echo "usage: $0 [context|quant|all]" >&2; exit 1 ;;
esac

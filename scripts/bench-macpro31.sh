#!/usr/bin/env bash
# Collect throughput statistics for the macpro31 CPU build of llama.cpp.
#
# The run list is built as randomized rounds: every round contains each selected condition once,
# the order is shuffled inside the round, and no condition is ever run twice in a row. Each
# condition therefore appears once in every `reps` slice of the sweep, so drift on the machine is
# spread evenly instead of landing on whichever condition ran first.
#
# The machine must be quiet for the numbers to mean anything, so before every single run the script
# waits until no backup or indexer is running and the 1-minute load average is under GATE_LOAD.
# Runs already recorded in the done file are skipped, so an interrupted sweep resumes.
#
# llama-bench emits one record per test: prompt processing has n_gen == 0, token generation has
# n_prompt == 0. A condition with test=both is one process that runs both tests.
#
# Usage: scripts/bench-macpro31.sh [large|scale|depth|all]
#   large   scale and depth in one interleaved schedule (default)
#   scale   six models from 0.5B to 8B at depth 0
#   depth   generation and prompt processing at contexts up to 4096
#   all     the older context and quant phases as well
#
# Environment: MODELS_DIR, OUT, SEED, THREADS, PP, TG, GATE, GATE_LOAD, GATE_MAX_WAIT

set -euo pipefail

export PATH="/opt/local/bin:/usr/local/bin:$PATH"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

BENCH=$ROOT/build-macpro31-cpu/bin/llama-bench
MODELS_DIR=${MODELS_DIR:-$HOME/models}
OUT=${OUT:-$HOME/bench-macpro31.jsonl}
SEED=${SEED:-20261113}
THREADS=${THREADS:-8}
PP=${PP:-128}
TG=${TG:-32}
GATE=${GATE:-1}
# The desktop session alone keeps this machine near load 1.5, and the 1-minute load average still
# shows our own previous run for a while, so the gate relies on the CPU of the jobs that actually
# ruin a measurement. GATE_LOAD is only a runaway safety net.
GATE_LOAD=${GATE_LOAD:-12.0}
GATE_CPU=${GATE_CPU:-20}
GATE_MAX_WAIT=${GATE_MAX_WAIT:-21600}

BASE=${OUT%.jsonl}

[ -x "$BENCH" ] || { echo "missing $BENCH; run scripts/build-macpro31.sh first" >&2; exit 1; }

# plan <mode> -> "<group> <model> <depth> <test> <rep>" lines, one randomized round at a time
plan() {
    python3 - "$SEED" "$1" <<'PY'
import random
import sys

seed, mode = int(sys.argv[1]), sys.argv[2]

Q05 = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
L1B = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
Q15 = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
Q3B = "qwen2.5-3b-instruct-q4_k_m.gguf"
Q7B = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
Q8B = "Qwen3-8B-Q4_K_M.gguf"

# (group, model, depth, test, reps)
conditions = []

if mode in ("scale", "large", "all"):
    for m in (Q05, L1B, Q15, Q3B, Q7B, Q8B):
        conditions.append(("scale", m, 0, "both", 4))

if mode in ("depth", "large", "all"):
    # both tests where the depth is cheap, generation only where prefill costs minutes
    deep = [
        (Q05, 1024, "both"), (Q05, 2048, "gen"), (Q05, 4096, "gen"),
        (Q05, 2048, "prompt"), (Q05, 4096, "prompt"),
        (L1B, 1024, "both"), (L1B, 2048, "gen"),
        (Q15, 1024, "both"), (Q15, 2048, "gen"),
        (Q3B, 1024, "gen"),
    ]
    for m, d, t in deep:
        conditions.append(("depth", m, d, t, 3))

if mode == "all":
    for m in (Q05, L1B, Q15):
        for d in (0, 256, 1024):
            conditions.append(("context", m, d, "both", 4))
    for m in ("qwen2.5-0.5b-instruct-q4_0.gguf", Q05,
              "qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-instruct-fp16.gguf"):
        conditions.append(("quant", m, 0, "both", 4))

def key(c):
    return (c[0], c[1], c[2], c[3])

def shuffle_without_neighbours(batch, previous, rng):
    """Shuffle until no two neighbours, and no wrap from the previous round, share a condition."""
    best = list(batch)
    for _ in range(500):
        rng.shuffle(best)
        ok = True
        for i in range(len(best)):
            prev = best[i - 1] if i > 0 else previous
            if prev is not None and key(best[i]) == key(prev):
                ok = False
                break
        if ok:
            return best
    return best

rng = random.Random(seed)
rounds = max(c[4] for c in conditions)
work = []
previous = None
for r in range(rounds):
    batch = [c for c in conditions if c[4] > r]
    batch = shuffle_without_neighbours(batch, previous, rng)
    previous = batch[-1] if batch else None
    for g, m, d, t, _ in batch:
        work.append("%s %s %s %s %d" % (g, m, d, t, r))

print("\n".join(work))
PY
}

# combined CPU percent of the background jobs that ruin a measurement; parked helpers cost nothing
blocker_cpu() {
    ps -Ao pcpu,comm | grep -E '/(rsync|mds|mds_stores|mdworker|MRT|fseventsd)$' |
        awk '{ s += $1 } END { printf "%d", s + 0 }'
}

machine_idle() {
    local load cpu
    cpu=$(blocker_cpu)
    [ "$cpu" -le "$GATE_CPU" ] || return 1
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk -v l="$load" -v lim="$GATE_LOAD" 'BEGIN { exit !(l <= lim) }'
}

wait_for_idle() {
    [ "$GATE" = "1" ] || return 0
    local waited=0 load
    while ! machine_idle; do
        if [ "$waited" -ge "$GATE_MAX_WAIT" ]; then
            echo "[gate] still busy after ${GATE_MAX_WAIT}s" >&2
            return 1
        fi
        load=$(sysctl -n vm.loadavg | awk '{print $2}')
        printf '[gate] busy (load %s, background cpu %s%%, waited %ss)\n' "$load" "$(blocker_cpu)" "$waited" >&2
        sleep 60
        waited=$((waited + 60))
    done
    if [ "$waited" -gt 0 ]; then
        echo "[gate] idle after ${waited}s" >&2
    fi
    return 0
}

run_phase() {
    local mode=$1 plan_file=$2 done_file=$3
    local total n=0 key group model depth test rep
    local args=()

    plan "$mode" > "$plan_file"
    total=$(wc -l < "$plan_file" | tr -d ' ')
    touch "$done_file"
    echo "# mode=$mode seed=$SEED threads=$THREADS pp=$PP tg=$TG runs=$total start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2

    while read -r group model depth test rep; do
        n=$((n + 1))
        key="$group $model $depth $test $rep"
        if grep -qxF "$key" "$done_file"; then
            printf '[%d/%d] skip (done) %s\n' "$n" "$total" "$key" >&2
            continue
        fi
        printf '[%d/%d] %s\n' "$n" "$total" "$key" >&2
        wait_for_idle || { echo "[gate] stopping" >&2; exit 1; }
        case "$test" in
            both)   args=(-p "$PP" -n "$TG") ;;
            gen)    args=(-p 0 -n "$TG") ;;
            prompt) args=(-p "$PP" -n 0) ;;
            *)      echo "bad test mode $test" >&2; exit 1 ;;
        esac
        "$BENCH" -m "$MODELS_DIR/$model" "${args[@]}" -d "$depth" -t "$THREADS" -r 1 -o jsonl \
            >> "${BASE}-${group}.jsonl" 2>> "${BASE}.log"
        echo "$key" >> "$done_file"
    done < "$plan_file"
    echo "# done=$(date -u +%Y-%m-%dT%H:%M:%SZ) scale=$(wc -l < "${BASE}-scale.jsonl" 2>/dev/null | tr -d ' ') depth=$(wc -l < "${BASE}-depth.jsonl" 2>/dev/null | tr -d ' ')" >&2
}

case "${1:-large}" in
    large) run_phase large "${BASE}-large-plan.tsv" "${BASE}-large-done.tsv" ;;
    scale) run_phase scale "${BASE}-scale-plan.tsv" "${BASE}-scale-done.tsv" ;;
    depth) run_phase depth "${BASE}-depth-plan.tsv" "${BASE}-depth-done.tsv" ;;
    all)   run_phase all   "${BASE}-all-plan.tsv"   "${BASE}-all-done.tsv" ;;
    *)     echo "usage: $0 [large|scale|depth|all]" >&2; exit 1 ;;
esac

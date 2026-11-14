#!/usr/bin/env bash
# The models on this machine that do not fit one die, forced across both, and
# several runs on the ones that work.
#
# Placement here comes from capacity, not from the fit's preference: a model
# larger than one die has to span the pair, so -dev Vulkan0/Vulkan1 is passed
# explicitly and the layer split is the only option.
#
#   ./fit-large-models.sh [outdir]
#
# Loads 45 to 85 GiB per run. Allow several minutes each, and note that
# llama-bench reloads the model for every test row.

set -u
REPO=/home/threaded/projects/Llama.cpp/llama.cpp
PARENT=/home/threaded/projects/Llama.cpp
MODELS=$PARENT/.models
LMSTUDIO=$HOME/.lmstudio/models/lmstudio-community
OUT=${1:-$PARENT/logs/fit-large}
BIN=$REPO/build-vulkan/bin

export GGML_VK_VISIBLE_DEVICES=1,2

mkdir -p "$OUT"

CODER=$LMSTUDIO/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf
GEMMA=$LMSTUDIO/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf
GPTOSS=$LMSTUDIO/gpt-oss-120b-GGUF/gpt-oss-120b-MXF4-00001-of-00002.gguf
QN80=$LMSTUDIO/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q8_0-00001-of-00003.gguf

bench() {
    local name=$1; shift
    local log=$OUT/$name.log
    [ -s "$log" ] && { echo "skip $name (have log)"; return; }
    echo "=== $name load=$(cut -d' ' -f1 /proc/loadavg)"
    timeout 3600 "$BIN/llama-bench" "$@" > "$log" 2>&1
    echo "[$name exit=$?]"
}

# can a model load and generate at all, and where does the fit put it
probe() {
    local name=$1 model=$2
    [ -f "$model" ] || { echo "skip $name (no model)"; return; }
    local log=$OUT/probe-$name.log
    [ -s "$log" ] && { echo "skip $name (have log)"; return; }
    echo "=== probe-$name"
    timeout 3000 "$BIN/llama-cli" -m "$model" -c 4096 -n 16 \
        -p "The capital of France is" -st --no-warmup -v > "$log" 2>&1
    local rc=$?
    local pick
    pick=$(tr '\r' '\n' < "$log" | grep -oE "model fits on [A-Za-z0-9]+|does not fit on [A-Za-z0-9]+" | tail -1)
    local spill
    spill=$(tr '\r' '\n' < "$log" | grep -oE "CPU_Mapped model buffer size = +[0-9.]+ MiB" | tail -1)
    local speed
    speed=$(tr '\r' '\n' < "$log" | grep -oE "Generation: [0-9.]+ t/s" | tail -1)
    echo "    exit=$rc | ${pick:-no decision} | ${spill:-no host spill} | ${speed:-no tokens}"
}

# 45 GiB across two dies: depth, and flash attention on and off
[ -f "$CODER" ] && {
    bench coder-d0-4k -m "$CODER" -dev Vulkan0/Vulkan1 -ngl 99 -fa 1 -p 512 -n 128 -d 0,4096  -r 3
    bench coder-d16k  -m "$CODER" -dev Vulkan0/Vulkan1 -ngl 99 -fa 1 -p 512 -n 128 -d 16384   -r 2
    bench coder-fa0   -m "$CODER" -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -p 512 -n 128 -d 0,4096  -r 2
}

# a dense model that stays on one die, for comparison
bench gemma-1die -m "$GEMMA" -dev Vulkan0 -ngl 99 -fa 1 -p 512 -n 128 -d 0,4096 -r 3

# the ones that exceed the pair and have to spill
probe gptoss    "$GPTOSS"
probe qwen3next "$QN80"

echo DONE

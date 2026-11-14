#!/usr/bin/env bash
# What --fit decides about device count, and what that decision is worth.
#
# Two halves. The first asks, for each model, whether the fit keeps it on one
# device or spreads it, by reading the fit's own log line and the per device
# model buffer sizes. The second measures the same generation twice, once with
# the fit deciding and once with the devices pinned, which is the A/B that shows
# what the layer split costs.
#
# Run it on an otherwise idle machine. Several of these models are tens of GiB
# and take minutes to load from a cold page cache.
#
#   ./fit-placement.sh [outdir]
#
# Reads $PARENT/.models for the small models and $HOME/.lmstudio/models for the
# large ones. A model that is not present is skipped, not an error.

set -u
REPO=/home/threaded/projects/Llama.cpp/llama.cpp
PARENT=/home/threaded/projects/Llama.cpp
MODELS=$PARENT/.models
LMSTUDIO=$HOME/.lmstudio/models/lmstudio-community
OUT=${1:-$PARENT/logs/fit-placement}
BIN=$REPO/build-vulkan/bin

# the duo only: the Vega II carries no device code in this build
export GGML_VK_VISIBLE_DEVICES=1,2

mkdir -p "$OUT"

# one placement run: name, extra args, model path
place() {
    local name=$1 extra=$2 model=$3
    [ -f "$model" ] || { echo "skip $name (no model)"; return; }
    local log=$OUT/place-$name.log
    [ -s "$log" ] && { echo "skip $name (have log)"; return; }
    echo "=== $name [$extra]"
    timeout 2400 "$BIN/llama-cli" -m "$model" $extra -c 4096 -n 2 -p hi -st --no-warmup -v > "$log" 2>&1
    local rc=$?
    local pick
    pick=$(tr '\r' '\n' < "$log" | grep -oE "model fits on [A-Za-z0-9]+|does not fit on [A-Za-z0-9]+" | tail -1)
    local bufs
    bufs=$(tr '\r' '\n' < "$log" | grep -oE "Vulkan[01] model buffer size = +[0-9.]+ MiB" | tail -2 | tr '\n' ' ')
    echo "    exit=$rc | ${pick:-no decision} | $bufs"
}

# the same generation twice, fit deciding vs devices pinned
ab() {
    local name=$1 model=$2
    [ -f "$model" ] || { echo "skip ab-$name (no model)"; return; }
    local log=$OUT/ab-$name.log
    [ -s "$log" ] && { echo "skip ab-$name (have log)"; return; }
    echo "=== ab-$name"
    : > "$log"
    for cfg in "fit:" "pinned:-dev Vulkan0,Vulkan1"; do
        local label=${cfg%%:*} args=${cfg#*:}
        for i in 1 2 3; do
            printf '%s ' "$label" >> "$log"
            timeout 900 "$BIN/llama-cli" -m "$model" $args -c 2048 -n 128 \
                -p "Write a short paragraph about the ocean and nothing else." -st --no-warmup 2>&1 \
                | tr '\r' '\n' | grep -oE "Generation: [0-9.]+ t/s" | tail -1 >> "$log"
        done
    done
    cat "$log"
}

place q6      ""                 "$MODELS/Qwen3-0.6B-Q4_K_M.gguf"
place q8b     ""                 "$MODELS/Qwen3-8B-Q8_0.gguf"
place q30b    ""                 "$MODELS/Qwen3-30B-A3B-Q4_K_M.gguf"
place gemma31 ""                 "$LMSTUDIO/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf"
place coder   ""                 "$LMSTUDIO/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf"

# the user's own choice wins: neither of these should be overridden
place q6-userdev "-dev Vulkan0,Vulkan1" "$MODELS/Qwen3-0.6B-Q4_K_M.gguf"
place q6-userngl "-ngl 99"              "$MODELS/Qwen3-0.6B-Q4_K_M.gguf"
place q6-cpu    "-dev none"             "$MODELS/Qwen3-0.6B-Q4_K_M.gguf"

ab q6  "$MODELS/Qwen3-0.6B-Q4_K_M.gguf"
ab q8b "$MODELS/Qwen3-8B-Q8_0.gguf"

echo DONE

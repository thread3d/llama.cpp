#!/usr/bin/env bash
# Device level correctness tests on every topology of this machine.
#
# Runs the shipped test binaries that check a backend against itself and against
# the CPU reference, one topology at a time, and writes one log per topology plus
# a one line summary. Run it on an otherwise idle machine: the tests compare
# numbers, but a contended GPU is a slow GPU and the logs then take much longer.
#
#   ./device-tests.sh [outdir]
#
# A full test-backend-ops sweep is long. Set OPS_FILTER to restrict it, for
# example OPS_FILTER='FLASH_ATTN_EXT|MUL_MAT' when only those ops matter.

set -u
set -o pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:-$ROOT/../logs/device-tests}
FILTER=${OPS_FILTER:-}
JOBS=${OPS_JOBS:-4}

mkdir -p "$OUT"
: > "$OUT/summary.txt"

# topology : build : env assignment : backend name as test-backend-ops sees it
run_one() {
    local name=$1 build=$2 envs=$3 backend=$4
    local log="$OUT/$name.log"
    printf '=== %s (%s, backend %s) ===\n' "$name" "$build" "$backend" | tee -a "$OUT/summary.txt"

    # cross device copy: only meaningful when the topology has two devices
    echo "--- test-backend-copy" >> "$log"
    env $envs "$ROOT/build-$build/bin/test-backend-copy" >> "$log" 2>&1
    local rc_copy=$?

    local ops_arg=()
    [ -n "$FILTER" ] && ops_arg=(-o "$FILTER")
    echo "--- test-backend-ops ${ops_arg[*]:-}" >> "$log"
    env $envs timeout 7200 "$ROOT/build-$build/bin/test-backend-ops" \
        -b "$backend" -j "$JOBS" "${ops_arg[@]}" >> "$log" 2>&1
    local rc_ops=$?

    echo "--- test-fusion" >> "$log"
    env $envs timeout 1800 "$ROOT/build-$build/bin/test-fusion" >> "$log" 2>&1
    local rc_fusion=$?

    local ops_line
    ops_line=$(grep -E "^[0-9]+/[0-9]+ tests passed" "$log" | tail -1)
    local copy_line
    copy_line=$(grep -E "copies arrived intact" "$log" | tail -1)
    local fusion_line
    fusion_line=$(grep -E "tests passed|FAIL" "$log" | tail -1)

    printf '  copy   rc=%s %s\n' "$rc_copy" "${copy_line:-no result}" | tee -a "$OUT/summary.txt"
    printf '  ops    rc=%s %s\n' "$rc_ops" "${ops_line:-no result}" | tee -a "$OUT/summary.txt"
    printf '  fusion rc=%s %s\n' "$rc_fusion" "${fusion_line:-no result}" | tee -a "$OUT/summary.txt"
}

# The Vega II is not a build target, so every filter below keeps it hidden.
run_one cpu        vulkan ""                             CPU
run_one rocm-die1  rocm   "HIP_VISIBLE_DEVICES=0"        ROCm0
run_one rocm-die2  rocm   "HIP_VISIBLE_DEVICES=1"        ROCm0
run_one rocm-both  rocm   "HIP_VISIBLE_DEVICES=0,1"      ROCm0
run_one vk-die1    vulkan "GGML_VK_VISIBLE_DEVICES=1"    Vulkan0
run_one vk-die2    vulkan "GGML_VK_VISIBLE_DEVICES=2"    Vulkan0
run_one vk-both    vulkan "GGML_VK_VISIBLE_DEVICES=1,2"  Vulkan0

echo "logs in $OUT" | tee -a "$OUT/summary.txt"

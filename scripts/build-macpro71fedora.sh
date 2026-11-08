#!/usr/bin/env bash
# Build llama.cpp on MacPro71Fedora (Fedora 44, Xeon W-3275, ROCm 7.1, RADV).
# GPUs: 2x Radeon Pro W6800X (gfx1030) + Radeon Pro Vega II (gfx906), 32 GiB each.
#
# Usage: scripts/build-macpro71fedora.sh [rocm|vulkan|both]

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# ccache sits on PATH via /usr/lib64/ccache and needs a writable cache dir
# set CCACHE_DIR when HOME is read-only
export CCACHE_DIR="${CCACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/ccache}"
mkdir -p "$CCACHE_DIR"

JOBS=$(nproc)

build_rocm() {
    export ROCM_PATH=/opt/rocm
    export HIP_PATH=/opt/rocm
    export HIP_COMPILER=clang

    cmake -S . -B build-rocm -G Ninja \
        -DGGML_HIP=ON \
        -DGGML_RPC=ON \
        -DAMDGPU_TARGETS="gfx906;gfx1030" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
        -DCMAKE_PREFIX_PATH=/usr/lib64/cmake
    cmake --build build-rocm --parallel "$JOBS"
}

build_vulkan() {
    cmake -S . -B build-vulkan -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON
    cmake --build build-vulkan --parallel "$JOBS"
}

case "${1:-both}" in
    rocm)   build_rocm ;;
    vulkan) build_vulkan ;;
    both)   build_rocm; build_vulkan ;;
    *)      echo "usage: $0 [rocm|vulkan|both]" >&2; exit 1 ;;
esac

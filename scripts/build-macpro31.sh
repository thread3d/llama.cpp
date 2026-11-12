#!/usr/bin/env bash
# Build llama.cpp on macpro31 (Mac Pro 3,1, OS X 10.11.6 El Capitan, Darwin 15.6).
# CPUs: 2x Xeon X5482 (Harpertown: SSE4.1, no AVX), 24 GiB RAM.
# GPU: NVIDIA Quadro 4000 (Fermi GF100) - not Metal capable, so no GPU backend can run here.
#
# Apple clang cannot build this tree before macOS 10.15, because libc++ marks std::filesystem
# unavailable there. Use GCC 10 from MacPorts: sudo port install gcc10
#
# Usage: scripts/build-macpro31.sh [cpu|check]

set -euo pipefail

# MacPorts and Homebrew install outside the default PATH, so this also works over ssh or cron
export PATH="/opt/local/bin:/usr/local/bin:$PATH"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

CC_MP=${CC_MP:-/opt/local/bin/gcc-mp-10}
CXX_MP=${CXX_MP:-/opt/local/bin/g++-mp-10}
JOBS=${JOBS:-$(sysctl -n hw.ncpu)}
BUILD=build-macpro31-cpu

# GGML_NATIVE=ON gives -march=native, which on Harpertown is core2 plus SSE4.1 and no AVX.
# GGML_METAL=OFF: the Quadro 4000 exposes no Metal device.
# GGML_ACCELERATE=OFF: the 10.11 Accelerate.h headers do not compile with GCC.
# GGML_BLAS=OFF: BLAS only serves F32 tensors, so it does nothing for quantized models.
cmake -S . -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC_MP" \
    -DCMAKE_CXX_COMPILER="$CXX_MP" \
    -DGGML_NATIVE=ON \
    -DGGML_METAL=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DLLAMA_CURL=OFF

cmake --build "$BUILD" --parallel "$JOBS"

case "${1:-cpu}" in
    cpu)
        ;;
    check)
        # The CPU backend is the reference, so test-backend-ops skips it. Run the tests that
        # do check CPU kernels against reference code.
        for t in test-quantize-fns test-gguf test-sampling test-tiled-mulmat; do
            echo "== $t"
            "./$BUILD/bin/$t"
        done
        ;;
    *)
        echo "usage: $0 [cpu|check]" >&2
        exit 1
        ;;
esac

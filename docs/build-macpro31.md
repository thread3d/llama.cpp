# llama.cpp on Mac Pro 3,1 (OS X 10.11)

This branch makes llama.cpp build and run on a Mac Pro 3,1 with OS X 10.11.6 El Capitan, using
the CPU backend. GPU acceleration is not possible on this machine; the evidence is below.

## Hardware and software

| Item | Value |
|---|---|
| CPU | 2x Xeon X5482 (Harpertown), 8 cores, SSE4.1, no AVX |
| RAM | 24 GiB |
| GPU | NVIDIA Quadro 4000 (Fermi GF100), 2 GiB |
| OS | OS X 10.11.6, Darwin 15.6 |
| Compiler | GCC 10.2.0 (MacPorts), because Apple clang cannot build this tree |
| CMake / Ninja | 3.19.1 / 1.10.2 |

## Why GCC 10 and not Apple clang

Apple clang 8.0 (Xcode 8.2) rejects `-std=c++17`, and libc++ marks `std::filesystem` as
`unavailable: introduced in macOS 10.15`, so this tree cannot be built with it on El Capitan:

```
.../include/c++/v1/filesystem:738:24: error: 'path' is unavailable: introduced in macOS 10.15
```

GCC 10 ships its own libstdc++, which implements C++17 and `std::filesystem` with no deployment
target restriction. Two consequences follow from using it:

- The Accelerate framework headers (`Accelerate.h` -> `vecLib.h` -> `vBasicOps.h`) use vector
  types that GCC rejects, so `GGML_ACCELERATE` must stay off. `ggml-blas` includes
  `Accelerate.h` too, so the BLAS backend is off as well. This costs almost nothing: `ggml-blas`
  only accelerates F32 tensors, so it never applies to quantized models.
- The Metal backend cannot compile at all: the 10.11 Foundation headers use Objective-C
  generics and nullability, which GCC 10 does not support.

## Changes in this branch

- `ggml/src/ggml.c`: `ggml_time_ms()` and `ggml_time_us()` call `clock_gettime(CLOCK_MONOTONIC)`,
  which only exists from macOS 10.12. On Apple platforms whose headers do not define
  `CLOCK_MONOTONIC` they now use `mach_absolute_time()` instead. Without this the file does not
  compile on El Capitan: `error: 'CLOCK_MONOTONIC' undeclared`.
- `tests/test-tiled-mulmat.cpp`: the same missing `clock_gettime`. It now uses `ggml_time_us()`,
  which removes the timer `#ifdef` and the `windows.h` and `time.h` includes.
- `ggml/CMakeLists.txt` and `docs/build.md`: Metal is no longer on by default for Darwin < 22,
  because the backend relies on Metal 3 era APIs that older SDKs do not declare and older
  runtimes do not implement. Pass `-DGGML_METAL=ON` to build it anyway.

## GPU acceleration: not available on this machine

No GPU backend in this tree can run on a Mac Pro 3,1. This is a property of the hardware and the
OS, not of the build:

| Backend | Why it cannot run |
|---|---|
| Metal | `MTLCreateSystemDefaultDevice()` returns nil. The Quadro 4000 is Fermi (GF100), and Metal needs Kepler or newer on NVIDIA. The backend also needs Metal 3 era APIs that 10.11 does not have. |
| Vulkan | macOS has no native Vulkan and MoltenVK is built on Metal. There is no Vulkan loader on the machine: `Could NOT find Vulkan (missing: Vulkan_LIBRARY Vulkan_INCLUDE_DIR)`. |
| CUDA | The installed toolkits are 4.1 and 7.0. `ggml-cuda` needs CUDA 11 or newer and compute capability 5.0 or newer; Fermi is 2.0, and CUDA 8.0 was the last release supporting macOS. |
| OpenCL | The backend targets OpenCL 3.0 with Adreno kernels. The Quadro 4000 offers OpenCL 1.1. |
| SYCL, HIP, MUSA, CANN, Hexagon | Vendor platforms that do not exist for macOS. |

So there is one build and not two. The GPU still drives the display; it just cannot compute here.

## Building

```
sudo port install gcc10 cmake ninja
scripts/build-macpro31.sh          # build build-macpro31-cpu
scripts/build-macpro31.sh check    # build, then run the tests that check CPU kernels
```

Binaries land in `build-macpro31-cpu/bin`.

## Verification on this machine

- `test-quantize-fns`, `test-gguf` (101/101), `test-sampling` and `test-tiled-mulmat` pass.
  `test-tiled-mulmat` reports `max_err: 0.000000` for every quantized type, so the SSE4.1
  kernels agree with the reference implementation.
- `test-backend-ops` skips the CPU backend by design and therefore proves nothing here.
- Greedy generation with Qwen2.5-0.5B-Instruct Q4_K_M (630M parameters, 463 MiB):
  `The capital of France is` -> `Paris.`
- `llama-bench -p 64 -n 32 -t 8` on the same model: **pp64 15.79 t/s, tg32 4.45 t/s**.

Use `-t 8`: the two sockets have four cores each and there is no SMT, so more threads only adds
contention.

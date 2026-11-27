# llama.cpp on Mac Pro 3,1 (OS X 10.11)

This branch makes llama.cpp build and run on a Mac Pro 3,1 with OS X 10.11.6 El Capitan, using the
CPU backend. GPU acceleration is not possible on this machine; the evidence is below, and the
measured throughput of the CPU backend is at the end.

The numbers here are from build 12743 (`57654b8`, upstream `abca6920`) at **7 threads**. The
one-paragraph reason for 7 and not 8 is under "Measured throughput"; the full argument, with the
measurements, is in [`build-macpro31-perf.md`](build-macpro31-perf.md).

## Hardware and software

| Item | Value |
|---|---|
| CPU | 2x Xeon X5482 (Harpertown), 8 cores, SSE4.1, no AVX |
| RAM | 32 GiB |
| GPU | NVIDIA Quadro 4000 (Fermi GF100), 2 GiB |
| OS | OS X 10.11.6, Darwin 15.6 |
| Compiler | GCC 10.2.0 (MacPorts), because Apple clang cannot build this tree |
| CMake / Ninja | 3.19.6 / 1.10.2 |

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
- `ggml/src/ggml-cpu/arch/x86/quants.c`: SSSE3 dot-product kernels for `q4_K`, `q6_K`, `q5_0` and
  `q8_0`, in the same per-ISA style the file already uses for the other formats. Before this,
  those four formats fell through to scalar reference code on any CPU without AVX2, which is
  this one. It is the largest single change in throughput: prefill 3.5x to 3.7x on the K-quant
  models, and generation 1.3x. [`build-macpro31-perf.md`](build-macpro31-perf.md) has the A/B
  measurements and the correctness checks (perplexity, fixed-seed greedy output, and each kernel
  against its generic reference).
- `scripts/build-macpro31.sh`: builds with `-DGGML_OPENMP=OFF` and `-DLLAMA_USE_PREBUILT_UI=OFF`.
  OpenMP is the other large change: MacPorts GCC ships libgomp, so the default configuration
  found it and put a barrier at every compute node, which costs 2x to 3.2x on generation.
  The prebuilt-UI download fails against the current Hugging Face bundle layout; skipping it
  still builds `llama-server`, it just embeds an empty UI.

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

The branch tip (`3a66096`, llama.cpp build 12833, upstream `ca123ba`, 2026-11-27) was verified with
the full CTest suite and the build script's own checks. The 87-commit upstream merge applied without
textual conflicts, and the result matches the build 12743 suite exactly — nothing new fails. This
merge rebuilt and re-tested the tree; it did not re-measure throughput, so the numbers below still
describe build 12743.

- `ctest`: **71 of 74 tests pass**, one is skipped and two cannot run on this OS.
- `scripts/build-macpro31.sh check` (`test-quantize-fns`, `test-gguf`, `test-sampling`,
  `test-tiled-mulmat`) passes. `test-tiled-mulmat` reports `max_err: 0.000000` for every quantized
  type, including the `q4_K` path, so the SSSE3 kernels still agree with the reference after the
  upstream merge.
- `test-backend-ops` skips the CPU backend by design and therefore proves nothing here.
- **Skipped:** `test-host-buffer-release` measures `phys_footprint` with `task_info(TASK_VM_INFO)`
  before and after freeing a 256 MiB buffer. Darwin 15 does not populate that field (it reports
  zero), so the test is registered with `SKIP_RETURN_CODE` 77 and skips instead of failing.
- **Blocked by the platform:** `test-conversion-fp8` and `test-kv-cache-scales` need PyTorch 2.1 or
  newer for `torch.float8_e4m3fn`. The PyTorch 2.2 wheel fails to load here because `libc10.dylib`
  needs a newer `/usr/lib/libc++.1.dylib` than El Capitan has; PyTorch 1.13.1 imports but predates
  the float8 dtype.
- **Python version matters.** Several tests run Python scripts, and `scripts/chat_tts_norm.py` uses
  `tuple[int, ...]`. Python 3.8 stops with `TypeError: 'type' object is not subscriptable`, so the
  suite was run with Homebrew Python 3.9.1 and jinja2 3.1.4; with Python 3.8, `test-jinja-py` fails
  52 assertions purely on that import. `test-jinja -py` shells out to `python3` from `PATH`, while
  `scripts/build-macpro31.sh` puts MacPorts first, so run ctest with `/usr/local/bin` ahead of
  `/opt/local/bin` (e.g. `PATH=/usr/local/bin:/opt/local/bin:/usr/bin:/bin:/usr/sbin:/sbin`):
  otherwise `test-jinja-py` fails on Python 3.8 even though nothing about the tree changed.
- The four models added to the sweep (Qwen2.5-32B Q6_K, gemma-2-27B Q6_K, Qwen2.5-14B Q8_0,
  Mistral-7B Q8_0) each load and run; the 25 GiB 32B file fits in this machine's 32 GiB.

## Measured throughput

This section covers models up to 1.5B and contexts up to 1024 tokens. The larger sweep, up to 32B
across nine architectures and contexts up to 4096, is in
[`build-macpro31-scale.md`](build-macpro31-scale.md).

Every number comes from `scripts/bench-macpro31.sh`, which is committed so the sweep can be
repeated. The raw llama-bench records are in `docs/macpro31/bench-context.jsonl`,
`bench-quant.jsonl` and `bench-wide-7t.jsonl`; `scripts/plot-macpro31.py` turns them into the
charts and `docs/macpro31/results.csv`.

### Method

- `llama-bench`, CPU backend, **7 threads**, batch and ubatch at their defaults, Flash Attention
  `auto`. The build is 12743 (`57654b8`, upstream `abca6920`) with GCC 10.2 and `-DGGML_OPENMP=OFF`.
- **Why 7 and not 8.** Eight worker threads on eight cores leaves no headroom for the rest of the
  system, and generation collapses: on this machine 0.5B q4_K_M measures about 15 tokens/s at 7
  threads and 3.3 at 8, with the same build and the same model, reproducibly. 7 threads leaves one
  core free; 6 leaves two and is now the recommended default, at 88 percent of the prefill and no
  loss of generation. The full 4/6/7 series is in `build-macpro31-scale.md`.
- A prompt processing test is 128 tokens, a token generation test is 32 tokens.
- Context depth is swept with `-d`: llama-bench fills the context with that many tokens first, then
  runs the test. It builds the context once for the prompt test and again for the generation test,
  so wall time grows with roughly twice the depth prefill cost.
- Each repetition is a separate `llama-bench` process. The full run list is shuffled with a fixed
  seed before it starts, so drift on the machine (thermal, background work) spreads across every
  model and depth instead of landing on whichever condition ran first. The realized order is
  committed as `macpro31/bench-all-plan.tsv`.
- Four independent processes per condition. Error bars and the `sd` column are plus or minus one
  sample standard deviation of those four values, so they show run-to-run spread. They are not a
  confidence interval for the mean, and they do not cover conditions the sweep did not vary.
- Before every run the script waits until background CPU (Spotlight, Time Machine, virus scan) is
  under 20 percent and the load average is under 12. Spotlight indexes external volumes on this
  machine and Time Machine thins them, so the gate is what keeps that noise out of the numbers.

### Prompt processing against context depth

![Prompt processing throughput against context depth](macpro31/prompt-throughput-vs-depth.svg)

| model | depth | n | tokens/s | sd |
|---|---|---|---|---|
| Qwen2.5-0.5B q4_K_M | 0 | 4 | 58.50 | 0.63 |
| Qwen2.5-0.5B q4_K_M | 256 | 4 | 56.60 | 0.64 |
| Qwen2.5-0.5B q4_K_M | 1024 | 4 | 53.11 | 0.78 |
| Llama-3.2-1B q4_K_M | 0 | 4 | 33.08 | 0.06 |
| Llama-3.2-1B q4_K_M | 256 | 4 | 31.90 | 0.35 |
| Llama-3.2-1B q4_K_M | 1024 | 4 | 29.15 | 0.35 |
| Qwen2.5-1.5B q4_K_M | 0 | 4 | 23.51 | 0.13 |
| Qwen2.5-1.5B q4_K_M | 256 | 4 | 22.71 | 0.14 |
| Qwen2.5-1.5B q4_K_M | 1024 | 4 | 20.53 | 0.27 |

### Token generation against context depth

![Token generation throughput against context depth](macpro31/generation-throughput-vs-depth.svg)

| model | depth | n | tokens/s | sd |
|---|---|---|---|---|
| Qwen2.5-0.5B q4_K_M | 0 | 4 | 16.00 | 0.45 |
| Qwen2.5-0.5B q4_K_M | 256 | 4 | 14.22 | 0.46 |
| Qwen2.5-0.5B q4_K_M | 1024 | 4 | 12.59 | 0.35 |
| Llama-3.2-1B q4_K_M | 0 | 4 | 7.80 | 0.07 |
| Llama-3.2-1B q4_K_M | 256 | 4 | 7.17 | 0.19 |
| Llama-3.2-1B q4_K_M | 1024 | 4 | 6.41 | 0.25 |
| Qwen2.5-1.5B q4_K_M | 0 | 4 | 6.11 | 0.26 |
| Qwen2.5-1.5B q4_K_M | 256 | 4 | 5.79 | 0.14 |
| Qwen2.5-1.5B q4_K_M | 1024 | 4 | 5.21 | 0.05 |

### Quantization at depth 0

All four are the same Qwen2.5-0.5B model, so this isolates the quantization format.

![Throughput by quantization](macpro31/quantisation.svg)

| quantization | test | n | tokens/s | sd |
|---|---|---|---|---|
| q4_0 | prompt | 4 | 70.88 | 0.44 |
| q4_0 | generation | 4 | 17.41 | 0.15 |
| q4_K_M | prompt | 4 | 58.38 | 0.74 |
| q4_K_M | generation | 4 | 15.87 | 0.59 |
| q8_0 | prompt | 4 | 57.42 | 0.87 |
| q8_0 | generation | 4 | 12.11 | 0.18 |
| f16 | prompt | 4 | 13.53 | 0.23 |
| f16 | generation | 4 | 5.48 | 0.03 |

### Throughput against model size

![Throughput against model size](macpro31/scaling.svg)

## What the numbers say

- **Prompt processing is now compute bound at a much higher rate.** Q4_K_M gives 58.4, 33.1 and
  23.5 tokens/s for 0.63B, 1.24B and 1.78B, which is 73.6, 81.7 and 83.5 GFLOP/s effective (two
  operations per parameter). Before the SSSE3 kernels the same sweep worked out to about 21-26 GFLOP/s: the
  K-quant formats were running scalar reference code, so the vector unit sat idle.
- **Generation is bandwidth bound and now at the memory ceiling.** 0.5B, 1B and 1.5B generate at
  15.8, 7.8 and 6.2 tokens/s. With 0.49, 0.80 and 1.11 GB of weights those are 7.7, 6.3 and
  6.8 GB/s read, against 7.14 GB/s measured for this machine with an 8-thread streaming probe. There
  is no headroom left; generation cannot be made faster without a faster memory system.
- **Context costs both, and generation more.** From depth 0 to 1024, generation retains 80, 82 and
  83 percent on the three models, while prompt processing retains 91, 89 and 87 percent. The
  attention pass reads the KV cache on every generated token and that read grows with the context,
  so the cost lands on generation.
- **The old 4.4x advantage of Q4_0 over Q4_K_M is gone.** It was an artifact of `q4_K_M` having no
  vector kernel on a CPU without AVX2. With kernels for both, q4_0 leads q4_K_M by 21 percent on
  prefill (70.9 against 58.4) and 10 percent on generation (17.4 against 15.9). Q4_K_M remains the
  better quality per byte, and on this machine it is no longer a throughput penalty.
- **f16 is now the slow choice**, at 13.5 tokens/s prefill and 5.4 generation: four times the bytes
  per weight of a 4-bit format, and no integer dot-product path to exploit.
- **Practical advice for this machine:** the 0.5B is the model this CPU can hold a conversation with
  (12-17 tokens/s generation, 53-58 prefill), the 1.5B is usable in batch or with patience (5.1-6.2
  generation, 21-23 prefill), and anything larger belongs to the batch section of
  [`build-macpro31-scale.md`](build-macpro31-scale.md). On a machine that is somebody's desktop,
  `-t 6` is the better default: 88 percent of this prefill and no loss of generation.

## Reproducing the numbers

```
scripts/build-macpro31.sh
MODELS_DIR=~/models THREADS=7 scripts/bench-macpro31.sh all      # a few hours on this machine
python3 scripts/plot-macpro31.py \
    docs/macpro31/bench-context.jsonl docs/macpro31/bench-quant.jsonl docs/macpro31
python3 scripts/plot-macpro31-large.py \
    docs/macpro31/bench-scale.jsonl docs/macpro31/bench-depth.jsonl docs/macpro31
```

`bench-macpro31.sh` needs `MODELS_DIR` to contain the GGUF files it names, plus a `coverage` phase
that walks every `*.gguf` in it. The sweep is slow because llama-bench rebuilds the depth context
for each of its two tests, so depth dominates the cost. It is resumable: runs already listed in
the done file are skipped, so an interrupted sweep continues where it stopped.

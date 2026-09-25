# llama.cpp on macOS with AMD GPUs

This branch adds working Metal GPU acceleration for **Intel Macs with discrete AMD
GPUs** - Mac Pro 2019, iMac Pro, 2019 16" MacBook Pro, and Thunderbolt eGPUs. It does
not apply to Apple Silicon, which upstream already supports well.

## Why a separate build is needed

Upstream's Metal backend is written for Apple Silicon. Its kernels assume 32-lane
simdgroups and use `simdgroup_matrix` for the matrix-unit matmul. AMD GPUs report
`simdgroup matrix mul. = false`, so on those cards:

- prefill falls back to the matrix-vector path instead of a tiled matmul
- Flash Attention is gated off and runs on the CPU, which collapses generation
- the driver accumulates a Metal resource per host-to-device copy until it stalls
- GCN/Vega cards, whose wavefront is 64 lanes wide, mis-execute 32-lane kernels

The measured result on a Mac Pro 2019 with a Radeon PRO W6800X Duo was about
**5 t/s prefill and 3 t/s generation on a 1B model** - an order of magnitude slower
than the CPU.

This branch integrates the AMD Metal implementation from the
[ToshLLM](https://github.com/engeldlgado/toshllm) project:

| Component | What it fixes |
|---|---|
| **ToshGEMM** | a tiled matmul that restores fast prefill without `simdgroup_matrix` |
| **AMD attention kernel** | runs attention on the GPU instead of falling back to the CPU |
| **Persistent staging buffer** | one reusable staging buffer per device instead of a new Metal resource per copy |
| **wave64 path** | correct, fast kernels for GCN/Vega cards with 64-lane wavefronts |
| **Device selection, multi-GPU collectives** | Metal previously only ever used the system-default GPU |

## Prerequisites

- macOS 14 or newer, Intel (x86_64)
- CMake 3.14+ and the Xcode Command Line Tools
- Enough system RAM to hold the models; GPU VRAM is optional because experts and
  layers can stay on the host

## Building

Two builds are useful: a GPU build and a CPU build. The CPU build keeps native ISA
flags, which the GPU build deliberately pins for reproducibility, so the CPU build is
a few percent faster on CPU-only work.

### GPU build

```sh
cmake -B build-amd -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_OSX_ARCHITECTURES=x86_64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DLLAMA_OPENSSL=OFF
cmake --build build-amd -j "$(sysctl -n hw.ncpu)"
```

`GGML_METAL_EMBED_LIBRARY=ON` embeds the shader source and compiles it on first use,
which costs roughly 70 seconds once per process. To avoid that, precompile the
kernels with Xcode's Metal tool (`xcrun -sdk macosx metal`) into a `default.metallib`
next to the binaries and build with `-DGGML_METAL_EMBED_LIBRARY=OFF`.

Add `-DTOSH_ENABLE_DYNAMIC_MOE=ON` to build the optional bounded-VRAM MoE expert
cache. It is inert unless enabled at runtime.

### CPU build

```sh
cmake -B build-cpu -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build-cpu -j "$(sysctl -n hw.ncpu)"
```

## Running

**`TOSH_FA_AMD=1` is required.** The application that ships these kernels sets it
itself; a raw `llama-cli`/`llama-server` invocation does not. Without it the AMD
attention kernels stay disabled and generation collapses. On a gemma-4-26B-A4B this
is the difference between 16 t/s and 75 t/s.

```sh
# one GPU
TOSH_FA_AMD=1 ./build-amd/bin/llama-server \
    -m model.gguf -ngl 99 -fa auto -t 14 -c 4096 --port 8080

# all GPUs
TOSH_FA_AMD=1 GGML_METAL_DEVICE_LIST=0,1,2 TOSH_MGPU_EVENTS=1 \
    ./build-amd/bin/llama-server \
    -m model.gguf -ngl 99 -fa auto -t 14 -c 4096 --split-mode layer --port 8080
```

### Environment variables

| Variable | Effect |
|---|---|
| `TOSH_FA_AMD` | enable the AMD attention kernels. **Required on AMD** |
| `GGML_METAL_DEVICE_INDEX=N` | use physical GPU `N` only |
| `GGML_METAL_DEVICE_LIST=a,b` | register the listed GPUs; slot `i` maps to physical `list[i]` |
| `TOSH_MGPU_EVENTS=1` | event-based cross-device synchronisation; use with `DEVICE_LIST` |
| `TOSH_MGPU_PEER=1` | peer-direct transfers where the cards allow it |
| `GGML_METAL_CONCURRENCY_ENABLE=1` | force concurrent encoders back on (only correct on unified memory) |
| `GGML_METAL_MM_MANUAL_DISABLE=1` | fall back to the matrix-vector prefill path |
| `GGML_METAL_MM_DOUBLE_BUFFER_DISABLE=1` | keep ToshGEMM but disable its double buffering |
| `TOSH_MGPU_SHAPE_CACHE=1` | keep a second prepared graph shape to avoid re-allocation |
| `GGML_METAL_WAVE64_DECODE_DISABLE=1` | disable the wave64 GPU path on GCN cards |
| `TOSH_MV_ALIGN=0\|1` | force aligned quant reads on/off (diagnostic) |

### Multi-GPU and `n_cpu_moe`

Use `--split-mode layer` with more than one device. `-ncmoe` (`--n-cpu-moe`) keeps the
experts of the first N layers on the host; it is what makes a model larger than the
available VRAM fit at all. It is not a tuning knob to leave unset. For the models in
this repository:

| Model | size | 1 GPU | 2 GPUs | 3 GPUs |
|---|---:|---:|---:|---:|
| gemma-4-26B-A4B Q4_K_M | 16 GB | 0 | 0 | 0 |
| Ornith-1.5-9B BF16 | 17 GB | 0 | 0 | 0 |
| gpt-oss-120b MXFP4 | 59 GB | 22 | 4 | 0 |
| Ornith-1.5-35B BF16 | 66 GB | 28 | 8 | 0 |

### Device order is not stable

The Metal device indices change across reboots, so probe them rather than hard-coding:

```sh
TOSH_FA_AMD=1 GGML_METAL_DEVICE_LIST=0,1,2 ./build-amd/bin/llama-bench \
    -m small.gguf -p 0 -n 1 -ngl 99 -r 1 2>&1 >/dev/null | grep 'device '
```

Each line reports the index, the card name, and the probed SIMD-group width
(`32` = Apple/AMD RDNA, `64` = AMD GCN/Vega). Cards sharing a board, such as the two
dies of a W6800X Duo, have a faster link between them than to a separate card, so a
pair of same-board devices scales better than a pair spanning two cards.

## Performance

## Performance

Measured on this machine: Mac Pro 2019 (Xeon W-3275, 256 GB) with a Radeon PRO W6800X
Duo and a Radeon Pro Vega II, macOS 26.6.2. `llama-bench -p 512 -n 128 -r 3`, so each
cell is the mean and standard deviation of three repetitions.

`cpu` is the CPU build at `-t 14`. `gpu1` is one W6800X die, `gpu2` is both dies of that
card, `gpu3` is all three GPUs including the Vega II. `n_cpu_moe` is 0 everywhere except
Ornith-1.5-35B (28/8/0) and gpt-oss-120b (22/4/0) for gpu1/gpu2/gpu3 - those keep experts
on the host so the model fits the VRAM of the configuration.

| model | size | test | cpu | gpu1 | gpu2 | gpu3 |
|---|---:|---|---:|---:|---:|---:|
| Llama-3.2-1B Q8_0 | 1.2 GB | pp512 | 345.0 &plusmn; 2.74 | 6637.7 &plusmn; 7.30 | 6180.4 &plusmn; 41.01 | 4867.5 &plusmn; 37.40 |
|  |  | tg128 | 36.6 &plusmn; 0.12 | 248.4 &plusmn; 0.16 | 228.1 &plusmn; 0.70 | 224.6 &plusmn; 1.15 |
| Ornith-1.5-9B BF16 | 17 GB | pp512 | 74.2 &plusmn; 0.28 | 714.1 &plusmn; 0.76 | 516.8 &plusmn; 2.26 | 480.4 &plusmn; 0.65 |
|  |  | tg128 | 3.8 &plusmn; 0.01 | 27.3 &plusmn; 0.01 | 26.9 &plusmn; 0.05 | 28.5 &plusmn; 0.02 |
| gemma-4-26B-A4B Q4_K_M | 16 GB | pp512 | 94.5 &plusmn; 0.72 | 1513.8 &plusmn; 1.72 | 1197.8 &plusmn; 1.93 | 979.6 &plusmn; 1.36 |
|  |  | tg128 | 12.3 &plusmn; 0.02 | 75.1 &plusmn; 0.05 | 72.2 &plusmn; 0.08 | 65.8 &plusmn; 0.02 |
| Ornith-1.5-35B BF16 | 66 GB | pp512 | 50.9 &plusmn; 0.90 | 52.4 &plusmn; 3.52 | 95.2 &plusmn; 4.81 | 591.6 &plusmn; 5.31 |
|  |  | tg128 | 6.8 &plusmn; 0.06 | 12.2 &plusmn; 0.03 | 22.2 &plusmn; 0.18 | 46.6 &plusmn; 0.03 |
| gpt-oss-120b MXFP4 | 59 GB | pp512 | 63.2 &plusmn; 0.34 | 153.2 &plusmn; 1.04 | 359.2 &plusmn; 4.47 | 268.3 &plusmn; 6.06 |
|  |  | tg128 | 13.3 &plusmn; 0.03 | 16.3 &plusmn; 0.39 | 32.3 &plusmn; 4.30 | 69.5 &plusmn; 0.13 |

What the table says:

- **A model that fits one card is fastest on one card.** The 1B, 9B and 26B all peak at
  `gpu1`; spreading them over more devices only adds transfer cost (1B prefill
  6638 -> 4868, 26B 1514 -> 980, 9B 714 -> 480).
- **A model that does not fit scales hard with device count.** The 66 GB 35B gains about
  11x prefill and 7x generation going from one GPU to all three (52 -> 592 t/s and
  6.8 -> 46.7 t/s).
- **The 120B MoE is the exception.** Its experts are the bulk of the weights, so prefill
  peaks at `gpu2` (359 t/s) while generation keeps improving to `gpu3` (69.5 t/s, 5x the CPU).
- **Against the CPU**, generation improves 2-7x on every model and prefill by one to two
  orders of magnitude.
- The two W6800X dies sit on one board and share a faster link than either has to the
  Vega II, which is why `gpu2` often beats `gpu3` for prefill.

## Benchmarking

`scripts/bench-macos-amd.sh` runs the full sweep - CPU plus 1, 2 and 3-GPU
configurations across several models, `-r 3` per cell - and writes a TSV plus the
markdown table above.

```sh
scripts/bench-macos-amd.sh                  # ~1-2 h for the default model set
REPS=5 scripts/bench-macos-amd.sh           # more repetitions per cell
MODELS_DIR=~/models scripts/bench-macos-amd.sh
```

Raw JSON lands in `bench-macos-amd-raw/`, results in `bench-macos-amd.tsv`. The script
is resumable: an existing per-cell JSON is reused.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Garbage output, or a token repeated forever | `TOSH_FA_AMD=1` not set |
| Generation far slower than expected | same; attention is running on the CPU |
| A different card than expected is used | device order changed after a reboot |
| ~70 s before anything happens | embedded shader library compiling; precompile `default.metallib` |
| Allocation failure on a large model | set `-ncmoe` so experts fit in VRAM |
| Prompt speed much worse with all GPUs than with two | the cross-card link; leave the slower card out |
| A run dies with a shader error (`invalid string literal value for 'host_name'`) or a segfault at startup | transient on this hardware - re-run it. One cell of the sweep below failed this way once and passed on retry. This machine has a documented history of GPU resets, so treat a single occurrence as noise, but investigate if it repeats |

## Licensing and provenance

The AMD Metal implementation in this branch is ported from **ToshLLM**
(<https://github.com/engeldlgado/toshllm>) and is
**Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>**,
licensed **GPL-3.0-or-later**.

llama.cpp itself is MIT. Because this branch combines the two, a build or source tree
that contains the AMD kernels is a derivative of GPL-3.0 code, and distributing it
requires that:

- the complete corresponding source is published under GPL-3.0-or-later,
- these copyright and licence notices are kept intact,
- the changes to the files are stated, with dates.

Do not redistribute this branch as MIT-licensed. If you need an MIT-only tree, build
from `master` without the patch series; you will not get AMD Metal acceleration.

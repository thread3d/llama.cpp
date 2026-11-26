# llama.cpp on Mac Pro 3,1: model size and context depth

Second sweep on the same machine as `docs/build-macpro31.md`: models from 0.5B to 32B across nine
architectures, at context depths up to 4096 tokens, plus one pass over every GGUF file on the
machine. This revision adds higher-precision and larger files to that pass: Qwen2.5-32B Q6_K,
gemma-2-27B Q6_K, Qwen2.5-14B Q8_0 and Mistral-7B Q8_0, so the widest set is 34 files. Build, portability fixes and the GPU situation are in that document; this one reports
measurements.

## Hardware

| Item | Value |
|---|---|
| CPU | 2x Xeon X5482 (Harpertown), 8 cores, SSE4.1, no AVX |
| RAM | 32 GiB |
| OS | OS X 10.11.6, Darwin 15.6 |
| Backend | CPU only, **7 threads** |

## Method

- `llama-bench` from the build produced by `scripts/build-macpro31.sh`; batch and ubatch at their
  defaults, Flash Attention `auto`, 7 threads. Seven rather than eight because eight worker threads
  on eight cores leave no headroom and generation collapses to about a third on this machine when
  anything else is runnable; the measurement is in [`build-macpro31-perf.md`](build-macpro31-perf.md).
- Six models, all Q4_K_M so that size is the main variable: Qwen2.5-0.5B, Llama-3.2-1B,
  Qwen2.5-1.5B, Qwen2.5-3B, Mistral-7B, Qwen3-8B.
- A prompt processing test is 128 tokens, a token generation test is 32 tokens. Depths of 1024 and
  above build a context of that many tokens before the test runs.
- **Interleaved schedule.** The run list is built as randomized rounds: each round holds every
  selected condition once, the order inside a round is shuffled, and no condition is ever placed
  next to itself, including across a round boundary. Every condition therefore appears once in each
  slice of the sweep, so a slow change on the machine is spread evenly across all conditions instead
  of biasing one. The realized order is committed as `macpro31/bench-all-plan.tsv`.
- **A quiet machine before every run.** Before each individual run the script waits until the CPU
  used by `rsync`, `backupd`, `mds`, `mdworker`, `MRT` and `fseventsd` is under 20 percent and the
  1-minute load average is under 12, then starts the run. The load limit matters as much as the CPU
  one on this machine: Spotlight indexes several external volumes and Time Machine periodically
  thins a 5.5 TiB backup volume, and both put the system into I/O wait without showing high CPU.
- Repeats: 4 per condition in the size sweep, 3 per condition in the depth and coverage sweeps. Each
  point is the mean of its runs and the error bar is one sample standard deviation, so bars show
  run-to-run spread rather than a confidence interval for the mean.
- Deep contexts cost real time: `llama-bench` rebuilds the depth context once for the prompt test and
  again for the generation test, so a 4096-token condition pays two 4096-token prefills. Where that
  is minutes of work, only the test of interest was run: generation only at 2048 and 4096 for the
  smaller models, and prompt processing at depth measured separately for Qwen2.5-0.5B. Each chart
  states which depths it covers.
- The depth-0 baseline for the deeper points comes from the size sweep, which measures the same
  models at depth 0 with 4 runs.
- **The whole suite was re-run at other thread counts,** on the pre-merge build: 6, 5, 4, 3, 2 and 1
  threads over the 22-model set, in `macpro31/bench-*-{6,5,4,3,2,1}t.jsonl`. Those records, and the
  thread comparison below, describe build 12330 (`2c333cf`). The 2026-11-25 re-baseline after the
  upstream merge re-measured the **7-thread** suite only; the thread series was not re-run on build
  12743.

## Throughput against model size

![Throughput against model size](macpro31/scale-throughput-vs-size.svg)

| model | params | n | prompt t/s | sd | generation t/s | sd |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.63B | 4 | 58.38 | 0.89 | 15.77 | 0.88 |
| Llama-3.2-1B | 1.24B | 4 | 33.05 | 0.04 | 7.84 | 0.02 |
| Qwen2.5-1.5B | 1.78B | 4 | 23.48 | 0.27 | 6.16 | 0.21 |
| Qwen2.5-3B | 3.40B | 4 | 11.07 | 0.14 | 3.20 | 0.07 |
| Mistral-7B | 7.25B | 4 | 4.55 | 0.07 | 1.50 | 0.04 |
| Qwen3-8B | 8.19B | 4 | 4.52 | 0.04 | 1.36 | 0.04 |

## Token generation against context depth

![Token generation against context depth](macpro31/depth-generation.svg)

| model | depth | n | t/s | sd | retained |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0 | 4 | 15.77 | 0.88 | 100.0% |
| Qwen2.5-0.5B | 1024 | 3 | 12.64 | 0.46 | 80.1% |
| Qwen2.5-0.5B | 2048 | 3 | 10.83 | 0.33 | 68.7% |
| Qwen2.5-0.5B | 4096 | 3 | 8.56 | 0.02 | 54.3% |
| Llama-3.2-1B | 0 | 4 | 7.84 | 0.02 | 100.0% |
| Llama-3.2-1B | 1024 | 3 | 6.42 | 0.19 | 81.9% |
| Llama-3.2-1B | 2048 | 3 | 5.58 | 0.24 | 71.2% |
| Qwen2.5-1.5B | 0 | 4 | 6.16 | 0.21 | 100.0% |
| Qwen2.5-1.5B | 1024 | 3 | 5.09 | 0.24 | 82.5% |
| Qwen2.5-1.5B | 2048 | 3 | 4.56 | 0.06 | 74.0% |
| Qwen2.5-3B | 0 | 4 | 3.20 | 0.07 | 100.0% |
| Qwen2.5-3B | 1024 | 3 | 2.76 | 0.03 | 86.2% |
| Mistral-7B | 0 | 4 | 1.50 | 0.04 | 100.0% |
| Qwen3-8B | 0 | 4 | 1.36 | 0.04 | 100.0% |

## Prompt processing against context depth

![Prompt processing against context depth](macpro31/depth-prompt.svg)

| model | depth | n | t/s | sd | retained |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0 | 4 | 58.38 | 0.89 | 100.0% |
| Qwen2.5-0.5B | 1024 | 3 | 53.10 | 0.90 | 91.0% |
| Qwen2.5-0.5B | 2048 | 3 | 48.71 | 0.71 | 83.4% |
| Qwen2.5-0.5B | 4096 | 3 | 41.85 | 0.60 | 71.7% |
| Llama-3.2-1B | 0 | 4 | 33.05 | 0.04 | 100.0% |
| Llama-3.2-1B | 1024 | 3 | 29.30 | 0.25 | 88.6% |
| Qwen2.5-1.5B | 0 | 4 | 23.48 | 0.27 | 100.0% |
| Qwen2.5-1.5B | 1024 | 3 | 20.45 | 0.35 | 87.1% |
| Qwen2.5-3B | 0 | 4 | 11.07 | 0.14 | 100.0% |
| Mistral-7B | 0 | 4 | 4.55 | 0.07 | 100.0% |
| Qwen3-8B | 0 | 4 | 4.52 | 0.04 | 100.0% |

## Throughput retained

![Throughput retained against context depth](macpro31/depth-retention.svg)

Solid lines are token generation, dashed lines are prompt processing, and 100 percent is each
model's own depth-0 figure.

## Depth 0 against depth 2048

![Token generation at depth 0 and depth 2048](macpro31/depth-bars-2048.svg)

## Every model file on the machine

`scripts/bench-macpro31.sh coverage` walks every `*.gguf` in `MODELS_DIR` at depth 0, so the chart and
table below answer "how does each file actually run" rather than following the six-model comparison
above. The duplicates that differ only in quantization are included on purpose: they are what is on
the disk. This is the `bench-wide-7t.jsonl` measurement: 7 threads, 34 files, 3 runs each, on build
12743. The older 22-model `bench-coverage*.jsonl` files are the pre-merge thread-count series and are
kept only for the thread chart.

![Throughput of every model file](macpro31/coverage-throughput.svg)

| model | prompt t/s | sd | generation t/s | sd |
|---|---|---|---|---|
| stories260K (toy) | 19276.33 | 77.07 | 4118.11 | 18.97 |
| Qwen2.5-0.5B q4_0 | 71.03 | 0.74 | 17.38 | 0.30 |
| Qwen2.5-0.5B q4_K_M | 58.50 | 0.87 | 15.60 | 1.18 |
| Qwen2.5-0.5B q8_0 | 57.59 | 0.41 | 12.20 | 0.14 |
| Llama-3.2-1B q4_K_M | 32.78 | 0.58 | 7.65 | 0.36 |
| SmolLM2-1.7B (llama) | 19.45 | 0.13 | 5.75 | 0.33 |
| Qwen2.5-1.5B q4_0 | 20.66 | 0.28 | 6.42 | 0.03 |
| Qwen2.5-1.5B q4_K_M | 23.36 | 0.35 | 6.20 | 0.05 |
| R1-Distill-Qwen-1.5B | 23.36 | 0.31 | 6.20 | 0.03 |
| Qwen2.5-0.5B f16 | 13.67 | 0.02 | 5.52 | 0.03 |
| granite-3.1-2b | 12.77 | 0.15 | 3.95 | 0.17 |
| gemma-2-2b-it | 15.23 | 0.24 | 3.51 | 0.15 |
| Falcon3-3B | 13.02 | 0.08 | 3.56 | 0.03 |
| Qwen2.5-3B q4_K_M | 11.08 | 0.12 | 3.18 | 0.12 |
| Phi-3.5-mini | 5.00 | 0.06 | 2.53 | 0.02 |
| Yi-1.5-6B-Chat | 5.71 | 0.08 | 1.80 | 0.05 |
| Mistral-7B q4_0 | 4.10 | 0.05 | 1.62 | 0.06 |
| Mistral-7B q4_K_M | 4.55 | 0.06 | 1.48 | 0.05 |
| OLMo-2-7B | 4.83 | 0.04 | 1.48 | 0.05 |
| Qwen2.5-Coder-7B | 4.81 | 0.06 | 1.46 | 0.04 |
| Qwen2.5-Math-7B | 4.80 | 0.07 | 1.46 | 0.05 |
| internlm2.5-7b-chat | 4.54 | 0.07 | 1.42 | 0.05 |
| Qwen3-8B q4_K_M | 4.52 | 0.07 | 1.35 | 0.06 |
| aya-expanse-8b | 4.51 | 0.04 | 1.24 | 0.05 |
| gemma-2-9b-it | 3.69 | 0.06 | 1.08 | 0.04 |
| Mistral-Nemo-12B | 2.89 | 0.04 | 0.91 | 0.03 |
| Mistral-7B q8_0 | 2.95 | 0.03 | 0.93 | 0.03 |
| Phi-3-medium-14B | 1.73 | 0.02 | 0.74 | 0.03 |
| Qwen2.5-14B q4_K_M | 2.38 | 0.03 | 0.76 | 0.02 |
| DeepSeek-Coder-V2-Lite (MoE) | 9.72 | 0.03 | 3.35 | 0.05 |
| Qwen2.5-14B q8_0 | 1.69 | 0.03 | 0.47 | 0.02 |
| Qwen2.5-32B q4_K_M | 0.99 | 0.02 | 0.33 | 0.01 |
| gemma-2-27B q6_K | 0.78 | 0.07 | 0.27 | 0.01 |
| Qwen2.5-32B q6_K | 0.68 | 0.01 | 0.23 | 0.01 |

`stories260K` is a 1.1 MB toy model included because it is on the disk; its numbers only show that the
harness itself has no fixed overhead worth worrying about.

Four higher-precision files were added on 2026-11-25: Qwen2.5-32B Q6_K (25.0 GiB), gemma-2-27B Q6_K
(20.8 GiB), Qwen2.5-14B Q8_0 (14.6 GiB) and Mistral-7B Q8_0 (7.2 GiB). They behave exactly as the
bandwidth model predicts: less quantisation means more bytes per weight, so both phases slow down.
Mistral-7B falls from 4.55 to 2.95 prefill and 1.48 to 0.93 generation; the 14B from 2.38 to 1.69 and
0.76 to 0.47; the 32B from 0.99 to 0.68 and 0.33 to 0.23 (Q6_K against Q4_K_M); gemma-2-27B Q6_K is
the slowest file on the machine at 0.78/0.27. The 32B Q6_K is 25 GiB in a 32 GiB machine and still
runs, but it has left interactive use far behind: one token every four seconds.

- **Every architecture on the disk loads and runs.** The set spans `llama` (Llama-3.2, SmolLM2,
  Mistral, Mistral-Nemo, Yi, Starling-class), `qwen2`/`qwen3` (Qwen, Qwen-Coder, Qwen-Math, DeepSeek R1
  distill), `phi3` (Phi-3.5-mini and Phi-3-medium), `granite` (IBM), `gemma2` (Google, 2b, 9b and
  27b), `olmo2` (AllenAI), `command-r` (Cohere aya), `falcon` (TII), `internlm2` (InternLM) and
  `deepseek2` (the mixture-of-experts Coder-V2-Lite). None needed a build change, on a CPU whose only
  SIMD is SSE4.1.
- **Generation is uniformly at the memory ceiling for dense models.** Reading every weight once per
  token gives 5.6 to 7.7 GB/s across the dense files, whatever the family or format, so throughput is
  set by file size and nothing else: Qwen2.5-Coder-7B, Qwen2.5-Math-7B, internlm2.5-7B and Mistral-7B
  all land between 1.42 and 1.48 tokens/s.
- **The mixture-of-experts model breaks that rule, and it is the most interesting line in the table.**
  DeepSeek-Coder-V2-Lite is a 16B model in a 10.36 GB file, but only about 2.4B parameters are active
  per token, so it generates at **3.35 tokens/s** — faster than every dense 7-8B model here, despite
  occupying more disk than any of them. Prefill is 9.72 tokens/s, also ahead of the dense 7B class. On
  a bandwidth-limited machine, sparse activation is worth more than a smaller file.
- **Prefill efficiency still varies by architecture.** Qwen, Llama, Gemma-2, OLMo-2, Falcon3,
  SmolLM2, Yi and aya sit in the 59 to 84 GFLOP/s band. The Q8_0 and Q6_K files fall to 42-50: they
  move twice the bytes per parameter, so they are no longer purely compute bound. Phi-3.5-mini is the
  low outlier at ~38 and Phi-3-medium reaches ~48. Same CPU, same threads, so those are the models'
  graphs and kernel layouts.
- **Practical read**: the 0.5B to 1.5B models are interactive (5.5 to 17.4 tokens/s generation), the
  3B class is marginal, and everything from 7B up is batch-only — except the MoE, which behaves like
  a 2.4B model while occupying 10 GB.

## What thread count changes

The whole suite was re-run at 6, 5, 4, 3, 2 and 1 threads over the same 22-model set. Seven down to
two are committed (`bench-coverage.jsonl` and `bench-coverage-{6,5,4,3,2}t.jsonl`); one thread was
still running when this was written.

![Throughput at 7, 6, 5, 4, 3 and 2 threads](macpro31/threads-throughput.svg)

Across the 21 real models (the toy is excluded), as a percentage of the 7-thread rate:

| threads | prefill median | prefill range | generation median | generation range | thread ratio |
|---|---|---|---|---|---|
| 6 | 88.0% | 85.7-88.9% | 98.1% | 93.9-106.9% | 85.7% |
| 5 | 73.8% | 72.7-74.9% | 89.2% | 83.5-104.2% | 71.4% |
| 4 | 59.7% | 58.9-60.8% | 78.3% | 72.4-102.6% | 57.1% |
| 3 | 45.0% | 44.4-46.3% | 63.8% | 57.3-83.2% | 42.9% |
| 2 | 30.2% | 29.7-31.4% | 47.0% | 40.8-65.5% | 28.6% |

Representative absolute figures, tokens/s:

| model | test | 7t | 6t | 5t | 4t | 3t | 2t |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B q4_0 | prefill | 70.29 | 61.39 | 51.63 | 42.13 | 31.81 | 21.51 |
| Qwen2.5-0.5B q4_0 | generation | 16.70 | 17.64 | 17.07 | 17.14 | 13.89 | 10.32 |
| Qwen2.5-0.5B q4_K_M | prefill | 57.95 | 49.65 | 42.57 | 34.34 | 26.01 | 17.59 |
| Qwen2.5-0.5B q4_K_M | generation | 15.63 | 15.59 | 14.67 | 13.05 | 10.72 | 8.16 |
| Qwen2.5-3B | prefill | 10.96 | 9.69 | 8.15 | 6.62 | 4.96 | 3.35 |
| Qwen2.5-3B | generation | 3.16 | 3.10 | 2.84 | 2.48 | 2.03 | 1.49 |
| Qwen3-8B | prefill | 4.47 | 3.95 | 3.31 | 2.68 | 2.01 | 1.35 |
| Qwen3-8B | generation | 1.34 | 1.29 | 1.20 | 1.04 | 0.85 | 0.62 |
| Qwen2.5-32B | prefill | 0.99 | 0.87 | 0.73 | 0.59 | 0.44 | 0.30 |
| Qwen2.5-32B | generation | 0.34 | 0.32 | 0.30 | 0.26 | 0.21 | 0.15 |

- **Prefill tracks the thread count and beats it slightly.** The medians sit 1-2 points above the
  thread ratio at every count (88.0 against 85.7, 73.8 against 71.4, 59.7 against 57.1, 45.0 against
  42.9, 30.2 against 28.6), and the spread across 21 models and nine architectures stays under 1.7
  points. That is what a purely compute-bound test looks like: the only variable is how many cores
  are working, with less per-thread contention as the count falls.
- **Generation is sublinear because it is bandwidth bound.** 98.1, 89.2, 78.3, 63.8 and 47.0 percent
  at 6, 5, 4, 3 and 2 threads. From 6 down to 3 those numbers track `sqrt(threads / 7)` closely
  (predicted 92.6, 84.5, 75.6, 65.5), which is the shape of a memory system that saturates before the
  cores do. At 2 threads the measurement falls below that curve (47.0 against 53.5), so the fit is a
  guide rather than a law.
- **The spread in generation is the format effect, not noise.** At 4 threads the range is 72 to 103
  percent: `q4_0` is unchanged from 7 threads (17.14 against 16.70) and `q8_0` keeps 92 percent,
  while `f16` falls to 74 percent and keeps falling (60 percent at 3, 45 at 2). Formats whose kernels
  need less bandwidth per token keep their speed when cores are removed.
- **Six threads is the sweet spot on a machine that is somebody's desktop.** 98 percent of the
  7-thread generation rate for 88 percent of the prefill, with two cores left for the desktop. The
  8-thread collapse documented in `build-macpro31-perf.md` is why "one more than 6" is not an option.
- The 1-thread count was never completed, and that pre-merge series is now stopped; only 7
  and 6 threads have been measured on build 12743 (below).

### The 34-file set on build 12743

The 7- and 6-thread counts were re-measured after the upstream merge, on build 12743 with the full
34-file set (`bench-wide-7t.jsonl` and `bench-wide-6t.jsonl`). These are the only thread numbers
measured on the current binary; the five-count series above describes the earlier 22-model set on
build 12330 and is kept for the shape of the curve.

![Throughput at 7 and 6 threads over 34 files, build 12743](macpro31/threads-throughput-12743.svg)

Across all 34 files, as a percentage of the 7-thread rate:

| threads | prefill median | prefill range | generation median | generation range | thread ratio |
|---|---|---|---|---|---|
| 6 | 87.3% | 84.8-92.0% | 97.1% | 92.0-104.6% | 85.7% |

Representative absolute figures, tokens/s:

| model | test | 7t | 6t | 6t/7t |
|---|---|---|---|---|
| Qwen2.5-0.5B q4_K_M | prefill | 58.50 | 49.61 | 85% |
| Qwen2.5-0.5B q4_K_M | generation | 15.60 | 15.46 | 99% |
| Qwen2.5-3B | prefill | 11.08 | 9.67 | 87% |
| Qwen2.5-3B | generation | 3.18 | 3.07 | 97% |
| Qwen3-8B | prefill | 4.52 | 3.94 | 87% |
| Qwen3-8B | generation | 1.35 | 1.31 | 97% |
| Mistral-7B Q8_0 | prefill | 2.95 | 2.57 | 87% |
| Mistral-7B Q8_0 | generation | 0.93 | 0.93 | 100% |
| DeepSeek-Coder-V2-Lite (MoE) | prefill | 9.72 | 8.71 | 90% |
| DeepSeek-Coder-V2-Lite (MoE) | generation | 3.35 | 3.27 | 98% |
| gemma-2-27B Q6_K | prefill | 0.78 | 0.71 | 92% |
| gemma-2-27B Q6_K | generation | 0.27 | 0.26 | 96% |
| Qwen2.5-32B Q6_K | prefill | 0.68 | 0.59 | 87% |
| Qwen2.5-32B Q6_K | generation | 0.23 | 0.22 | 94% |

- **The shape holds on the wider set.** Prefill lands at 87.3 percent, 1.6 points above the thread
  ratio and within 0.7 points of the 22-model figure (88.0), spread 84.8 to 92.0 across 34 files.
- **Generation keeps 97.1 percent, so the sweet spot survives the wider set** — 2.9 percent down,
  against 1.9 percent on the 22-model series. Read the claim as "a couple of percent", not zero.
- **Four files measured above 7 threads**: Mistral-7B q4_0 at 105 percent, qwen2.5-1.5b q4_0 and the
  stories260K toy at 103, Qwen2.5-14B Q8_0 at 101. Small or bandwidth-light files, where a few
  percent is run-to-run noise rather than a gain from dropping a core.
- **The high-precision additions sit in the same band.** The Q8_0 and Q6_K files take 87 to 92
  percent of their prefill, so twice the bytes per parameter does not change how the machine divides
  work across cores; it only halves the absolute rate.
- The run took 3 h 36 min for 102 conditions (34 files x 3 repeats) at 6 threads, with the
  quiet-machine gate blocking once.

## What the numbers say

- **Prompt processing delivers a nearly constant number of operations per second.** Throughput
  falls from 58.4 to 4.52 tokens/s across a 13x range of model size, and the product with two
  operations per parameter stays in one band:

  | model | params | prompt t/s | effective GFLOP/s |
  |---|---|---|---|
  | Qwen2.5-0.5B | 0.63B | 58.38 | 73.6 |
  | Llama-3.2-1B | 1.24B | 33.05 | 81.7 |
  | Qwen2.5-1.5B | 1.78B | 23.48 | 83.5 |
  | Qwen2.5-3B | 3.40B | 11.07 | 75.2 |
  | Mistral-7B | 7.25B | 4.55 | 66.0 |
  | Qwen3-8B | 8.19B | 4.52 | 74.0 |

  About 59 to 84 GFLOP/s for the 4-bit files, so the rough rule here is `prompt tokens/s ~ 75 / (2 x
  params in billions)`. The pre-SSSE3 builds on this machine managed 21-26 GFLOP/s, so the kernel work
  is most of the difference.
- **Generation is bandwidth bound and sits at the measured ceiling.** Reading every weight once per
  token, the effective read rate is 7.7 GB/s at 0.5B, 6.3 at 1B, 6.8 at 1.5B, 6.3 at 3B, 6.1 at 7B
  and 6.4 at 8B, against 7.14 GB/s for a pure 8-thread streaming read: the small model slightly
  exceeds that probe, the rest sit at 85-95 percent of it. The flatness over a 50x range of model
  size, 0.49 GB to 25.03 GB, is the whole story of generation on this machine.
- **Depth hurts generation much more than prompt processing.** At 1024 tokens generation retains
  80 to 86 percent while prompt processing retains 87 to 91 percent. At 4096 on Qwen2.5-0.5B,
  generation is down to 54 percent while prompt processing holds 72 percent. The attention pass
  reads the KV cache on every generated token, and that read grows with the context; prompt
  processing pays it once.
- **Format matters less than it used to.** With SSSE3 kernels for both, q4_0 leads q4_K_M by 21
  percent on 0.5B prefill but is level or slightly behind on the larger models. Generation is within
  a few percent either way, and q4_0 wins at 7B because the file is 7 percent smaller. The old 4.4x
  penalty for q4_K_M is gone.
- **This machine's practical ceiling is the 1.5B to 3B range.** 5.2 to 6.2 tokens/s is usable for
  interactive work, and 2.8 at 3B with a 1024-token context is marginal but alive. Mistral-7B at
  1.50 and Qwen3-8B at 1.36 tokens/s are batch-only. Prompt processing at 4.5 tokens/s for a 7B or
  8B model means a 1000-token prompt needs about four minutes before the first token, against the
  12 minutes the OpenMP build took.
- **Beyond 8B the machine still works, but leaves interactive use behind.** The 14B generates at
  0.76 tokens/s and the 32B Q4_K_M at 0.33, with prefill of 2.38 and 0.99 tokens/s, so a 1000-token prompt
  costs about 7 minutes on the 14B and 17 minutes on the 32B before the first token. The 32B fits in
  32 GiB only because llama.cpp maps the weights from disk.
- The two sweeps agree where they overlap: Qwen2.5-0.5B prompt processing at depth 1024 was 53.11 in
  the context sweep and 53.10 in the depth sweep.

## Reproducing the numbers

```
scripts/build-macpro31.sh
MODELS_DIR=~/models THREADS=7 scripts/bench-macpro31.sh all
python3 scripts/plot-macpro31-large.py \
    docs/macpro31/bench-scale.jsonl docs/macpro31/bench-depth.jsonl docs/macpro31
python3 scripts/plot-macpro31.py \
    docs/macpro31/bench-context.jsonl docs/macpro31/bench-quant.jsonl docs/macpro31
python3 scripts/plot-macpro31-coverage.py docs/macpro31/bench-wide-7t.jsonl docs/macpro31

# other thread counts, then the comparison
MODELS_DIR=~/models THREADS=6 OUT=~/bench-macpro31-r6.jsonl scripts/bench-macpro31.sh all
python3 scripts/plot-macpro31-threads.py docs/macpro31 \
    7t=docs/macpro31/bench-coverage.jsonl 6t=docs/macpro31/bench-coverage-6t.jsonl
```

The sweep is resumable: conditions already listed in the done file are skipped, so an interrupted
sweep continues where it stopped.

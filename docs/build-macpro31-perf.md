# llama.cpp performance on Mac Pro 3,1: what actually moves the needle

This document is about *making the CPU build faster* on the machine described in
[`build-macpro31.md`](build-macpro31.md): 2x Xeon X5482 Harpertown, SSE4.1, no AVX, no usable GPU,
OS X 10.11.6. That document covers how to build it and why no GPU backend can run; this one measures
where the time goes and which knobs are worth pulling.

Everything here is measured on the machine with `llama-bench`, alternating variants run by run so
that machine drift cannot land on one side of a comparison. The raw records and the harness scripts
that produced them are machine-local scratch data under `~/perf-2026/` (`logs/`, `builds/`, and the
`*.sh` / `*.c` helpers), so they are not committed here; the reproducible end-to-end sweep is
`scripts/bench-macpro31.sh`. Nothing was measured with a competing build or benchmark running.

## Summary

Ordered by measured effect on this machine:

| Change | Effect | Cost |
|---|---|---|
| `-DGGML_OPENMP=OFF` | generation **2.0x to 3.2x**, prefill +1-4% | one cmake flag |
| SSE4.1 kernel for q4_K and q6_K | prefill **3.5x to 3.7x**, generation 1.3x, on 256-aligned `Q4_K_M` models | ~130 lines of C |
| SSE4.1 kernel for q5_0 | prefill **3.3x**, generation **1.46x**, on models whose "Q4_K_M" file is really q5_0 | ~30 lines of C |
| SSE4.1 kernel for q8_0 | prefill **1.43x** on `Q8_0` models, generation unchanged | ~25 lines of C |
| Keep `-t 8` (the default) | already optimal once OpenMP is off | nothing |
| Profile-guided build | +4% prefill, +1% generation (retrained on the kernel build) | a training run and a second build |
| Use `Q4_K_M` rather than `Q4_0` | now 12.5% faster on prefill, and better quality | nothing |
| `-fa off` | **costs 36% of prefill at depth 1024** - do not disable it | nothing |
| `-ctk/-ctv q8_0` | costs 18% of prefill at depth 1024 for 1.8% generation | nothing |
| `-ub`, `--poll`, `--prio`, `-lm`, `-funroll-loops`, `-flto` | no measurable effect, or worse | - |

Together the first two rows take token generation on the 0.5B from 4.6 to 15.9 tokens/s and prefill
from 16.7 to 65; on the 1.5B, 2.9 to 5.7 and 7.4 to 26; on the 7B, 0.95 to 1.40 and 1.39 to 5.10.
The other two benchmark documents in this directory report this same build; the OpenMP numbers they
originally described are superseded and kept in this section only for comparison.

The kernels are conditional on the model rather than the machine - they cover what the file
actually contains, which is not always what its name says (Section 5).

## Method

- `llama-bench`, both tests. Prefill is `-p 128`, generation `-n 32` (`-n 16` for the 7B, where
  generation is under 1 token/s). Depth work uses `-d`.
- A/B comparisons alternate the two binaries run by run (A, B, then B, A, ...), and every record is
  tagged with which side produced it. Both sides therefore see the same machine conditions.
- The machine is somebody's desktop: a session, Spotlight and Time Machine run on it. Their effect
  shows up as scatter between repeats, which is reported as `sd`. Alternating runs is what keeps
  that scatter from becoming a fake difference.
- All builds use GCC 10.2 from MacPorts with `GGML_NATIVE=ON` (so `-march=native`, which on
  Harpertown is SSE4.1 and no AVX), `Release`, `-O3`.

## 1. The build was using OpenMP, and OpenMP is expensive here

`GGML_OPENMP` defaults to `ON`. On macOS with Apple clang, `find_package(OpenMP)` fails and the
build silently falls back to ggml's own threading, so this never shows up on a normal Mac. MacPorts
GCC ships libgomp, so the Mac Pro build found it and took the OpenMP path instead: `ggml-cpu.c`
opens one `omp parallel` region per graph and runs `#pragma omp barrier` at every compute node
(`ggml_barrier()`, `ggml-cpu.c:591`).

Prefill has enough work per node for that to hide. Generation evaluates one graph per token, and
each node in it is small, so hundreds of barriers per token become the dominant cost.

A/B against the same build with `-DGGML_OPENMP=OFF`:

| model | test | threads | OpenMP | pthread pool | change |
|---|---|---|---|---|---|
| Qwen2.5-0.5B q4_0 | prefill | 8 | 76.20 | 79.64 | +4.5% |
| Qwen2.5-0.5B q4_0 | generation | 8 | 5.34 | 16.87 | **+216%** |
| Qwen2.5-0.5B q4_0 | prefill | 4 | 41.71 | 42.38 | +1.6% |
| Qwen2.5-0.5B q4_0 | generation | 4 | 8.11 | 15.81 | **+95%** |
| Qwen2.5-0.5B q4_K_M | generation | 8 | 4.60 | 10.54 | **+129%** |
| Mistral-7B q4_K_M | generation | 8 | 0.96 | 1.03 | +7% |

The two builds compute the same thing, so this had to be checked rather than assumed:

- `llama-perplexity -c 512` over the same 12000-byte text: **12.2935 +/- 0.99695 from both builds**.
- Greedy generation with `--temp 0 --seed 1234 -n 64`: **byte-identical output** from both.
- The perplexity run itself: OpenMP 78.19 s wall / 337.8 s user, no OpenMP 46.64 s wall / 345.2 s
  user. Same CPU consumed, 1.68x less wall clock. The difference is synchronisation, not arithmetic.

The win is a fixed cost per token, so it shrinks as the model grows: the 0.5B model gains 2.1-3.2x
and the 7B gains 7%. On the models this machine can actually hold a conversation with, it is the
difference between usable and not.

`OMP_WAIT_POLICY=active` does not change it (5.35 against 5.34 tokens/s), so this is not threads
sleeping between graphs; it is the barrier structure inside them.

**Consequence for the existing documents:** every generation number in `build-macpro31.md` and
`build-macpro31-scale.md` was originally produced by the OpenMP build and understated this machine,
by a factor of 1.1 at 7B rising to 3.2 at 0.5B q4_0; it has since been re-measured with this build.

## 2. Thread count

With the pthread pool, 8 threads is best or within 6% of best everywhere:

| model | test | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B q4_0 | prefill | 21.59 | 42.38 | 61.91 | **79.54** |
| Qwen2.5-0.5B q4_0 | generation | 10.15 | 15.63 | **17.95** | 16.85 |
| Qwen2.5-0.5B q4_K_M | prefill | 4.35 | 8.67 | 12.91 | **16.86** |
| Qwen2.5-0.5B q4_K_M | generation | 3.45 | 6.51 | 9.15 | **10.49** |
| Mistral-7B q4_K_M | generation | - | 0.61 | 0.88 | **1.03** |

Prefill scales close to linearly: 7.0x on 8 threads for the `Q4_0` model and 7.7x for `Q4_K_M`,
measured from 1 thread up. Whatever the two-socket topology costs, it is not visible in prefill.

On the OpenMP build the same sweep showed generation *falling* from 4 threads to 8 (8.17 to 5.33 on
0.5B q4_0), and `build-macpro31.md` concluded from it that "more threads add contention rather than
throughput". That conclusion was an artifact of the OpenMP build; it does not hold once the build
uses ggml's own thread pool.

### The 8-thread cliff, re-measured on 2026-11-22

The table above says 8 threads is fine. Re-measuring the finished build at `ncpu` gives a different
answer once the machine is somebody's desktop while the sweep runs: with eight worker threads on
eight cores there is no headroom, so any competing runnable thread (WindowServer, Dock, a browser,
an indexer) delays the barrier at every node. Seven threads leave exactly one core free and
reproduce the generation figures above.

| model | test | 6 threads | 7 threads | 8 threads |
|---|---|---|---|---|
| Qwen2.5-0.5B q4_K_M | generation | 15.4 | **15.0** | 3.3 |
| Qwen2.5-0.5B q4_K_M | prefill | 50.1 | **57.5** | 44.4 |
| Qwen2.5-3B q4_K_M | generation | - | **3.08** | 1.03 |
| Mistral-7B q4_K_M | generation | - | **1.45** | 0.66 |

The cliff reproduced across four repeats, was unchanged by `--poll 0` or `--poll 100`, and was
neither thermal (`CPU_Speed_Limit = 100`, no thermal warning recorded) nor memory bound (`bw`
measured 7.14 GB/s against the 7.39 in Section 3) with nothing else runnable. So `-t 8` is only a
fair choice on a machine with no desktop activity; the committed sweep uses `THREADS=7`, and the
benchmark documents report 7-thread numbers. An 8-thread measurement taken with a live desktop is
~2-3x too low on generation.

A whole second and third sweep — every model up to 32B, every depth — were also run at 6 and at 4
threads. The comparison is in `build-macpro31-scale.md` under "What thread count changes": prefill
lands at 88-90 percent of the 7-thread rate at 6 threads and 59-61 percent at 4, flat across model
size (compute bound), while generation is 97-109 percent at 6 threads and 75-96 percent at 4
depending on the format (bandwidth bound). Six threads is therefore the sweet spot on a machine
that is somebody's desktop: no generation loss, 88 percent of the prefill, two free cores.

## 3. What the memory system can actually deliver

Generation reads every weight once per token, so the ceiling on generation throughput is the
machine's read bandwidth. Measured with a 1 GiB array, warm pages, `~/perf-2026/bw.c`:

| threads | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| GB/s | 3.29 | 3.69 | 5.68 | 7.25 | 7.39 |

Against that ceiling:

| model | generation t/s | model size | GB/s read | share of ceiling |
|---|---|---|---|---|
| Qwen2.5-0.5B q4_0 | 16.85 | 0.42 GB | 7.13 | **96%** |
| Qwen2.5-0.5B q4_K_M | 10.49 | 0.49 GB | 5.15 | 70% |
| Mistral-7B q4_0 | 1.56 | 4.11 GB | 6.41 | **87%** |
| Mistral-7B q4_K_M | 1.03 | 4.37 GB | 4.50 | 61% |

`Q4_0` has an SSSE3 dot product kernel and it now runs at 87-96% of what the memory system can
supply: there is almost nothing left to win there. The K-quant models sit at 61-70%, and the reason
is not DRAM but `ggml_vec_dot_q4_K_q8_K_generic`, a scalar loop that x86 only escapes with AVX2 or
AVX. That is where the remaining performance is.

The same 7B model in two formats makes that point without any cross-model guesswork: quantised to
`Q4_0` it generates at 1.56 tokens/s, as `Q4_K_M` at 1.03. **1.51x from the format alone**, and the
`Q4_K_M` file is only 6.3% larger, so about 1.42x of that is kernel and not bytes. It is the same
weights and the same machine; only the code that multiplies them differs.

**Section 5 closes this gap**, and it is worth seeing how far: with the kernels in place the same
models reach

| model | generation t/s | GB/s | share of the 7.4 GB/s ceiling |
|---|---|---|---|
| Mistral-7B q4_K_M | 1.40 | 6.12 | 83% |
| Qwen2.5-1.5B q4_K_M | 5.71 | 6.38 | 86% |

so every format on this machine now runs at 83-96% of what the memory system can deliver, and
generation is finally bandwidth bound on the formats that matter instead of arithmetic bound.

## 4. Profile-guided optimisation

Built with `-fprofile-generate`, trained on a mix of prefill and generation across four models, then
rebuilt in place with `-fprofile-use -fprofile-correction`. The A/B uses the same sources without
profiles:

| model | test | plain | PGO | change |
|---|---|---|---|---|
| Qwen2.5-0.5B q4_0 | prefill | 78.99 | 79.54 | +0.7% |
| Qwen2.5-0.5B q4_0 | generation | 16.64 | 16.77 | +0.8% |
| Qwen2.5-0.5B q4_K_M | prefill | 16.90 | 17.24 | +2.0% |
| Qwen2.5-0.5B q4_K_M | generation | 10.47 | 10.72 | +2.4% |
| Mistral-7B q4_K_M | generation | 1.10 | 1.17 | **+6.7%** |

Modest but consistent, and it grows with the model, which is where it matters: the 7B is the model
that needs help. It costs one extra build and one training run, both scripted in
`~/perf-2026/pgo.sh`.

Repeat the measurement on top of the kernels, though, and it shrinks to about half, because much of
what the profile was optimising was the scalar loop that no longer exists:

| kernel build, 1.5B q4_K_M | without PGO | with PGO | change |
|---|---|---|---|
| prefill | 26.11 | 27.18 | +4.1% |
| generation | 6.12 | 6.21 | +1.4% |
| kernel build, 7B q4_K_M generation | 1.52 | 1.53 | +0.9% |

So PGO is still worth taking, at a few percent of prefill, but it is no longer the thing to reach
for first: 4% for an extra build and a training run, against 3.5x for a kernel. Retrain against the
*kernel* build if you do it, or the profile is fitted to code that is no longer hot.

## 5. Vector kernels for the formats that had none

This is where the machine was leaving most of its performance. On x86, ggml provides an AVX2 kernel,
then an AVX kernel, then a scalar loop, for each quantised dot product. This CPU has neither, so
every format except `Q4_0` fell through to the scalar loop: `Q4_0` was the only one with an SSSE3
path, and `q4_K`, `q6_K` and `q8_0` were all scalar while the CPU sits at SSE4.1.

Three `__SSSE3__` branches were added, following the shape of the existing `Q4_0` kernel
(`pmaddubsw` for 16 unsigned-by-signed byte products, `pmaddwd` to widen to int32):

- **Q8_0**: the two 16-byte halves of a block, widened to int32 separately so the int16 pair sums
  cannot overflow before being added.
- **Q4_K**: nibbles unpacked into two 16-byte vectors per sub-block, with the per-sub-block 6-bit
  scale applied as a float multiply so every integer step stays well inside int32.
- **Q6_K**: its quants are `(nibble | (two bits of qh) << 4) - 32`, i.e. signed, so the abs/sign
  pair inside the existing `mul_sum_i8_pairs` helper is what makes `maddubs` legal; `|q| <= 32`
  keeps every int16 partial sum exact.

### Q8_0

| 0.5B Q8_0, 8 threads | plain | SSSE3 | change |
|---|---|---|---|
| prefill | 44.80 | 64.22 | **1.43x** |
| generation | 12.20 | 12.30 | 1.01x |

Generation does not move, which is the right answer: it is bandwidth bound (Section 3), so a kernel
that only speeds up arithmetic must leave it alone. That is a check on the measurement as much as on
the kernel.

### Q4_K and Q6_K

`Q4_K_M` stores most tensors as `q4_K` and the rest, including the output layer, as `q6_K`, so both
kernels matter. Same build without them as the reference, alternating runs, medians:

| model | test | plain | q4_K only | q4_K + q6_K |
|---|---|---|---|---|
| Qwen2.5-1.5B q4_K_M | prefill | 7.37 | 2.52x | **3.53x** (26.02 t/s) |
| Qwen2.5-1.5B q4_K_M | generation | 4.48 | 1.27x | 1.27x (5.71 t/s) |
| Qwen2.5-3B q4_K_M | prefill | 3.48 | - | **3.58x** (12.45 t/s) |
| Qwen2.5-3B q4_K_M | generation | 2.36 | - | 1.34x (3.17 t/s) |
| Mistral-7B q4_K_M | prefill | 1.39 | - | **3.66x** (5.10 t/s) |
| Mistral-7B q4_K_M | generation | 1.08 | 1.34x | 1.30x (1.40 t/s) |
| Qwen2.5-0.5B q4_K_M | prefill | 16.89 | 1.07x | - |
| Qwen2.5-0.5B q4_K_M | generation | 10.46 | 1.03x | - |

Prefill is compute bound, so the honest way to read this is operations per second:

| model | GFLOP/s before | after |
|---|---|---|
| Qwen2.5-1.5B | 26.2 | **92.6** |
| Qwen2.5-3B | 23.6 | **84.6** |
| Mistral-7B | 20.2 | **73.9** |
| Qwen2.5-0.5B as Q4_0, for scale (already SSE) | 100.2 | - |

Three and a half times more arithmetic per second from the same silicon, and the K-quant formats now
reach the same class as the `Q4_0` path they used to trail by 4.4x. Practically, on the 7B a
1000-token prompt went from about twelve minutes to about three.

### Which format to use, after this

Before the kernels, `Q4_0` beat `Q4_K_M` by 4.4x on prefill and `build-macpro31.md` recommended it.
That is no longer true. Same 1.5B weights, same kernel build, alternating runs, medians of four:

| format | prefill | generation |
|---|---|---|
| Q4_0 | 23.01 | 6.03 |
| Q4_K_M | **25.89** (+12.5%) | 6.16 |

`Q4_K_M` is now both the better-quality format and the faster one, so it is the default to pick.
`Q4_0` keeps its advantage only where the model cannot use 256-wide blocks at all, which is the
subject of the next subsection.

### Why the 0.5B barely moved, and the q5_0 kernel that fixed it

Qwen2.5-0.5B has hidden size 896, and 896 is not a multiple of `QK_K` = 256, so its 896-wide tensors
cannot be stored as `q4_K` blocks. The quantiser silently uses something else, and the file that
says `Q4_K_M` on the tin is mostly a different format:

| model | tensor types |
|---|---|
| Qwen2.5-0.5B "Q4_K_M" | `q5_0` 133, `q4_K` 12, `q6_K` 12, `q8_0` 13, `f32` 121 |
| Qwen2.5-1.5B Q4_K_M | `q4_K` 169, `q6_K` 29, `f32` 141 |
| Mistral-7B Q4_K_M | `q4_K` 193, `q6_K` 33, `f32` 65 |
| Qwen2.5-0.5B Q8_0 | `q8_0` 170, `f32` 121 |

So the 0.5B's Q4_K_M file is 84% `q5_0` by tensor count, and `q5_0` had no vector path on this CPU
either - which is exactly why it gained 7% instead of 3.5x. 1.5B (1536 = 6 x 256) and 7B
(4096 = 16 x 256) are 256-aligned and almost entirely `q4_K`/`q6_K`, which is why they move.

That made the next kernel obvious, because the 0.5B is the most interactive model this machine can
run. `q5_0` is `q4_0` plus a fifth bit per weight, packed one per element in `qh[4]`; the file
already has `bytes_from_bits_16`, which spreads 16 packed bits into a `0xFF` mask, so the high
nibble is one AND away and the rest is the same shape as the `q8_0` kernel:

| 0.5B "Q4_K_M" (133 q5_0 tensors), 8 threads | with q8_0+q4_K+q6_K | + q5_0 kernel | change |
|---|---|---|---|
| prefill | 19.47 | 64.99 | **3.34x** |
| generation | 10.89 | 15.90 | **1.46x** |

Verification, as for the others: `test-tiled-mulmat` 15/15 PASS with no failures; against
`ggml_vec_dot_q5_0_q8_0_generic` the worst relative difference on non-cancelling results is **2.3e-5**
and the worst absolute difference is 0.0068 on a result of 1.4e4, i.e. 5e-8 relative; perplexity on
that model moves 11.2174 -> 11.2096.

This is the kind of thing worth checking before believing a kernel result: the win is conditional on
the model's hidden size, not on the machine. `llama-quantize` prints the type mix for any file before
it refuses to requantize, which is the cheapest way to see it.

### Verification

New arithmetic needs more than a benchmark that got faster.

- `test-tiled-mulmat` reports `max_err: 0.000000` for q4_K and q6_K (and for the types that already
  had kernels). It does **not** cover q8_0, so passing it is not the check for that one.
- The real check compares each new kernel against its own generic reference, both exported from the
  library: `ggml_vec_dot_q8_0_q8_0_generic`, `ggml_vec_dot_q4_K_q8_K_generic`,
  `ggml_vec_dot_q6_K_q8_K_generic`. 300-500 random cases per seed, 1 to 32 blocks each, fp16 scales
  spanning several octaves, quants drawn from the range the quantiser actually produces, and `q8_K`
  group sums computed rather than invented (`~/perf-2026/quant_ref_check.c`).
  - worst relative difference on non-cancelling results: **3e-5** for q8_0, **5e-5** for q4_K,
    **7.5e-5** for q6_K.
  - The one case above those thresholds is `vec=586.570312` against `ref=586.890625`, an absolute
    difference of 0.32 on a sum whose terms are of order 1.9e6, i.e. 1.7e-7 relative. A relative
    comparison on a near-zero sum says nothing; the absolute error is the meaningful number and it
    is at float precision.
- The control: the same harness against the *unpatched* library compares each scalar path against
  its generic reference and gets exactly 0.0, so the harness can detect a difference when there is
  one.
- Structural: `pmaddubsw`/`psignb` appear in the disassembly of `ggml_vec_dot_q8_0_q8_0` six times
  in the patched library and zero times in the unpatched one.
- End to end, `llama-perplexity -c 512`: Q8_0 model 10.9769 -> 10.9671, and a 1.5B q4_K_M model
  8.8275 -> 8.8224. Those 0.05-0.09% shifts are the same reordering propagated through a softmax.
  Greedy generation with a fixed seed is byte-identical on the OpenMP/pthread pair, which is the
  comparison where identical output is actually expected.

The kernels are committed in `ggml/src/ggml-cpu/arch/x86/quants.c`; the check program is
`~/perf-2026/quant_ref_check.c`.

## 6. Runtime knobs, measured at depth 1024

Same model and threads; depth 1024 because that is where these knobs have something to do:

| setting | prefill | generation |
|---|---|---|
| default (`-fa auto`) | 66.79 | 13.31 |
| `-fa on` | 66.49 | 13.16 |
| `-fa off` | **42.60** | 12.98 |
| `-fa on -ctk q8_0 -ctv q8_0` | 54.72 | 13.55 |
| `-ub 128` | 66.90 | 13.13 |
| `-ub 2048` | 66.91 | 13.31 |

- **`auto` already means on**, and turning it off costs 36% of prefill. Do not disable flash
  attention on this machine.
- **Quantising the KV cache is a bad trade here.** It buys 1.8% of generation and costs 18% of
  prefill, because the attention path has to unpack the cache.
- **`-ub` does nothing at short prompts**: 128 tokens already fits one micro-batch. With a
  2048-token prompt it starts to matter, but only just: 256 gives 65.6 tokens/s, 512 (the default)
  71.6, 1024 72.4, 2048 70.8. The default is within 1% of the best; leave it alone unless the prompt
  is long and the micro-batch is smaller than 512.
- At depth 1024 generation falls from 16.85 to 13.31 tokens/s (79% retained) on the 0.5B `Q4_0`
  model. That is a steeper fall than `build-macpro31-scale.md` reports for `Q4_K_M` (94.6%), for the
  same reason as Section 3: a faster kernel leaves the KV read as a larger share of each token.

## 7. Compiler flags that do nothing here

Both candidates were built on top of the kernel build and A/B'd against it on a 1.5B `Q4_K_M`
model, alternating, medians of four runs:

| flag | prefill | generation | verdict |
|---|---|---|---|
| `-funroll-loops` | 1.011x | 0.951x | no |
| `-flto` | 1.001x | 0.972x | no |

Both are worth recording because the second one is widely assumed to be impossible here. GCC 10 on
this Darwin does accept `-flto`, links a working binary, and really does run whole-program
optimisation (`lto1` shows up in `ps` during the link). It simply does not pay: the hot loops in
this tree already live inside one translation unit per backend, and what crosses those boundaries
is not in a hot path. The build also takes noticeably longer.

`-funroll-loops` is worse than neutral: prefill is unchanged and generation loses about 5%, which is
consistent with its effect on the scalar and semi-scalar code in the attention path (bigger loops,
more pressure, no fewer flops).

Rejected before measuring, on correctness grounds rather than performance ones: anything in the
`-ffast-math` family. It would change how `expf` behaves inside softmax, and the whole point of the
kernel work above was to make the arithmetic faster without changing its answers. If someone wants
to try it, the verification to run is the same one used in Section 5: perplexity against the
unpatched build, and `quant_ref_check`.

## A build to use

Both changes are now in the tree:

1. **`-DGGML_OPENMP=OFF`** in `scripts/build-macpro31.sh`.
2. The q8_0, q5_0, q4_K and q6_K SSSE3 kernels in
   `ggml/src/ggml-cpu/arch/x86/quants.c`. This is the only source change, and the only item
   conditional on what a model file actually contains.

```
cmake -S . -B build-macpro31-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/local/bin/gcc-mp-10 \
    -DCMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-10 \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_CURL=OFF
```

What that lands, measured on this machine at 8 threads and depth 0, against the OpenMP numbers the
two benchmark documents originally reported:

| model | test | before (OpenMP build) | after | |
|---|---|---|---|---|
| Qwen2.5-0.5B "Q4_K_M" | prefill | 16.67 | **64.99** | 3.9x |
| Qwen2.5-0.5B "Q4_K_M" | generation | 4.55 | **15.90** | 3.5x |
| Qwen2.5-1.5B q4_K_M | prefill | 7.35 | **26.02** | 3.5x |
| Qwen2.5-1.5B q4_K_M | generation | 2.85 | **5.71** | 2.0x |
| Mistral-7B q4_K_M | prefill | 1.39 | **5.10** | 3.7x |
| Mistral-7B q4_K_M | generation | 0.95 | **1.40** | 1.5x |
| Qwen2.5-0.5B q8_0 | prefill | 43.34 | **64.55** | 1.5x |
| Qwen2.5-0.5B q8_0 | generation | 4.83 | **11.86** | 2.5x |
| Qwen2.5-0.5B q4_0 | prefill | 74.14 | 79.65 | 1.07x |
| Qwen2.5-0.5B q4_0 | generation | 5.27 | **16.90** | 3.2x |

The two rows that matter most for daily use are the first two: the 0.5B is the model this machine
can hold a conversation with, and it goes from 4.6 to 15.9 tokens/s.

Profile-guided optimisation on top adds about 4% of prefill and is described in Section 4; worth it
if an extra build and a training run are acceptable, skippable otherwise. If you do it, train on the
kernel build.

## Reproducing

```
scripts/build-macpro31.sh                      # already sets -DGGML_OPENMP=OFF; kernels are in the tree
~/perf-2026/bw.c                               # memory ceiling probe: gcc-mp-10 -O3 -march=native -pthread
~/perf-2026/ab.sh <label> <rounds> <binA> <binB> -- <llama-bench args>
~/perf-2026/fmtcomp.sh <label> <rounds> ...    # same binary, two model files, alternating
~/perf-2026/pgo.sh pgo2                        # two-stage profile-guided build
~/perf-2026/quant_ref_check.c                  # each kernel against its generic reference
llama-quantize <model> /dev/null q4_0          # prints which types a model file really contains
```

One measurement note worth carrying forward: use medians, not means. A single cold-cache run
(0.958 against 2.363 twice, on the 3B model) moves a mean ratio from 1.34x to 1.67x, and the
alternating A/B order is what stops that from biasing one variant.

Long sweeps were run detached in a named `screen` session on the machine, so a crash of the
controlling host does not kill them; each runner script skips steps whose results already exist.

## Open items

- **`q5_K` next, if anyone runs `Q5_K_M`.** It is the same recipe again (unpack five bits, `maddubs`,
  scale in float) and it is now the only common format on this machine with no vector path. Nothing
  here uses it, which is why it was not done: the types that are actually in use - `q4_K`, `q6_K`,
  `q5_0`, `q8_0`, `q4_0` - all have a kernel now.
- **The kernel work is worth upstreaming, carefully.** It is 180 lines of C in one file, it follows
  the existing per-ISA pattern exactly, and it is worth 3.5x on the architectures llama.cpp still
  claims to support without the AVX2 path. But note `AGENTS.md` in that tree: no PR descriptions or
  commits written by an agent, and nothing pushed upstream without a human who understands it at
  the keyboard. The kernels are in this branch and the verification harness is under
  `~/perf-2026/` for whoever wants to own it.
- **`-ffast-math` was not tried**, deliberately. It is the one remaining compiler knob with a
  plausible double-digit number attached, and it changes `expf` inside softmax. The verification to
  run is in Section 5, and the expected gain is small on this machine because the float loops that
  would benefit are already SSE.
- The `Q4_K_M` figures the first two documents originally reported were a floor rather than a
  property of the machine: the same CPU now generates the same 7B model 51% faster in a format
  whose kernel it can vectorise, and 3.7x faster on prefill once the format it actually is has a
  kernel too.

# CUDA/PTX megakernel research and implementation ledger

Research started 2026-09-18. The objective is the main NanoGPT speedrun: minimum
training time while preserving its token streams and reaching the validation
target. A component benchmark is not a speedrun result.

Current target: the maintainer-approved techniques in open PR #360, pinned in
[FRONTIER.md](FRONTIER.md) and `frontier.lock.json`. Sections below through the
SASS/PTX optimization describe the older merged-record MLP unless stated
otherwise. Its 3072-wide E4M3 timings do not apply to the new 2816-wide mixed-FP8
frontier target. The native full model remains incomplete.

## Pinned source trees

| Source | Commit | Local checkout |
| --- | --- | --- |
| KellerJordan/modded-nanogpt | `bc3a0c2d640d0d73dedaef87eae26148d2e32afb` | `../modded-nanogpt-upstream` |
| HazyResearch/Megakernels | `7309cec801537b61fea3b50d7dfe454a6cde578e` | `../Megakernels` |
| Its ThunderKittens submodule | `664c108d16f12707a73d3072ab525f26fb2b4f62` | `../Megakernels/ThunderKittens` |

The upstream checkout is detached and unchanged. The implementation branch is
`hillclimb/main-path`. The reference repositories are not runtime dependencies of
the initial native implementation.

## People and work that determine this design

This is a map of relevant contributions, not a ranking of people. Claims from
unmerged PRs below have not been reproduced here.

| People | Primary work examined | Consequence for this port |
| --- | --- | --- |
| Benjamin Spector, Jordan Juravsky, Stuart Sul, Dylan Lim, Owen Dugan, Simran Arora, Chris Ré | HazyResearch's low-latency and throughput megakernels; controller, loader, consumer, storer, scheduler, page allocator source | Schedule tiles with explicit dependencies; overlap independent resources and eventually pipeline loads across instructions. |
| Tri Dao, Jay Shah and FlashAttention-3 collaborators | Hopper attention paper and implementation description | Attention requires stable online softmax, explicit asynchronous operand lifetimes, and separate forward/backward designs. |
| Keller Jordan, Jeremy Bernstein, Vlado Boza | Muon writeup; current `polar_express()` and `NorMuonAndAdam` | Preserve optimizer grouping, momentum, matrix normalization, polynomial ordering, and Adam's distinct role. |
| Noah Amsel, David Persson, Christopher Musco, Robert M. Gower | Polar Express paper | Polynomial coefficients are part of the algorithm, not a interchangeable implementation detail. |
| You Jiacheng, Franz Cesista (`leloykun`), Braden Koszarsky | NanoGPT record history: FP8 head, packed projections, attention layout/window improvements | Optimize the actual FP8 layouts and scheduled shapes, rather than a generic GPT-2 surrogate. |
| Larry Dial (`classiclarryd`), Varun Neal | Main-path architecture/schedule; FlashAttention integration, MTP, bigrams, FP8 support | Keep data alignment, window transitions, sparse table ownership, and loss normalization in the port contract. |
| Chris McCormick | Unified optimizer, transposed head, flattened forward, transpose reuse, FA3 document bounds | Weight/gradient layout is a lifecycle decision; generating both needed layouts once is better than repeated strided consumption. |
| Andrew Briand, Josh Rauvola, Soren Dunn, `moof2x` | Fused ReLU-squared and softcapped cross-entropy records | Keep activation and loss epilogues in the producing tile, including backward state. |
| `Lisennlp`, Jan Varho, Shivanjan Chakravorty (`theonlyglitch_`) | MUDD/DC attention, prefix targets/canonical masking, packed FP8 QKV and all-six-GEMM FP8 MLP | Current main is substantially more complex than an 11-layer textbook transformer. These are required port surfaces. |
| Subhajit Ghosh (`isubuz`), Warpscale | Open PR #363 | Useful competing kernel/scaling evidence, but it targets an older baseline; do not transplant it as if it matched current main. |
| Deven Pietrzak | Open PR #360 | Sampled softmax, sparse embedding ownership and changed optimizer claim larger gains; these change the model/training algorithm and need a separate experiment from systems-equivalent fusion. |
| Herman Brunborg, Nihir Patel | Open PR titles #367 and #366 | Additional competitors to investigate; exact-match and host-RAM n-gram claims are not verified evidence in this ledger. |

Primary references:

- [HazyResearch Megakernels source](https://github.com/HazyResearch/Megakernels/tree/7309cec801537b61fea3b50d7dfe454a6cde578e).
- [Low-latency megakernel design](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).
- [Throughput megakernel design](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main).
- [HazyResearch's later implementation perspective](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions).
- [FlashAttention-3](https://tridao.me/blog/2024/flash3/) and [paper](https://arxiv.org/abs/2407.08608).
- [Muon](https://kellerjordan.github.io/posts/muon/) and [Polar Express](https://arxiv.org/abs/2505.16932).
- [NanoGPT record history and rules](https://github.com/KellerJordan/modded-nanogpt/blob/bc3a0c2d640d0d73dedaef87eae26148d2e32afb/README.md).
- [Packed QKV / FP8 MLP PR #344](https://github.com/KellerJordan/modded-nanogpt/pull/344).
- [Kernel tuning PR #363](https://github.com/KellerJordan/modded-nanogpt/pull/363).
- [ANVIL2 PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360).
- [NVIDIA PTX memory model and instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## What the HazyResearch reference actually does

`include/megakernel.cuh` instantiates persistent workers. Separate warp roles
load operands, compute, store, and control execution. The controller manages
instruction stages, semaphores, and reusable shared-memory pages.
`megakernels/scheduler.py` constructs a dependency graph and assigns instruction
queues to GPU workers. `demos/low-latency-llama` supplies concrete operator bodies.

Its two published workloads have different limiting resources. The small-model
decode design emphasizes bandwidth; the larger tensor-parallel design also
overlaps matrix computation and communication. Neither is a ready-made training
implementation. Their speedups cannot be applied numerically to NanoGPT training.

For this port, "fully fused" means a device-resident graph executing through one
kernel entry per GPU for a training step. Global memory is still necessary:
training activations and weights cannot all remain in registers/shared memory.
"Fine-grained" means a ready token tile can advance without a whole-layer grid
barrier. It does not mean that every dependency can be removed.

## Main-path contract extracted from the pinned source

The model construction uses 11 layers, residual width 768, six heads, Q/K head
width 96, value width 128, and an MLP intermediate width of 3072. The MLP bank
contains twelve pairs of matrices; actual use follows the explicit forward
routing and must not be inferred from `num_layers` alone.

The main schedule has 1250 scheduled iterations plus 40 extension iterations at
this commit. This differs from the earlier #344 submission's 1315-step total.
Validation covers 10,485,760 tokens. The dataset loader and validation calculation
are part of correctness, not interchangeable benchmark scaffolding.
The per-GPU token counts on eight GPUs are 16,384, 32,768 and 49,152 across the
three scheduled stages. M=8192 in the initial component tests is a development
shape, not one of those full training batches.

The major port surfaces are:

1. Embedding/bigram lookup, sign hash, smear, and residual routing.
2. Packed FP8 QKV, Q/K normalization, RoPE/YaRN, key offset and padding.
3. Causal/local variable-length attention, value embeddings, gates, XSA and DC correction.
4. FP8 ReLU-squared MLP, both projections, and all four backward GEMMs.
5. MUDD residual/value mixing and its gradients.
6. FP8 vocabulary head, softcap, MTP/prefix losses, and canonical validation mask.
7. Distributed gradient ownership, sparse exchange, accumulation, NorMuon/Adam,
   scheduled updates, and regenerated FP8 weights/scales.

### Exact MLP boundaries

Given quantized `X`, `W1`, `W2`, their scales, and output gradient `dY`:

| Result | Operation | Required boundary |
| --- | --- | --- |
| `post` | `relu(X @ W1.T)^2` | Dequantize accumulator, round preactivation to BF16, square/round, quantize to FP8; emit both layouts. |
| `Y` | `post @ W2` | FP32 accumulation with dequantization, BF16 output. |
| `dpre` | `(dY @ W2.T) * 2 * sqrt(post)` | Reconstruct the derivative from stored quantized `post`; preserve BF16 arithmetic boundaries, then FP8 quantization. |
| `dX` | `dpre @ W1` | Consume the saved FP8 weight layout. |
| `dW1` | `dpre.T @ X` | Accumulate across every token before the optimizer may update. |
| `dW2` | `post.T @ dY` | Independent of `dpre`, so overlap it when dependencies allow. |

Upstream's output-gradient scale is exact-current, while post/dpre scales are
delayed. The initial native graph takes quantized inputs and explicit scales;
scale estimation/update is still outside this graph and must be ported before
calling it a complete replacement for `FusedFP8MLPFunction`.

## Native scheduler and synchronization argument

`cuda/megakernel.cuh` implements a ready queue of matrix-tile tasks. An up-projection
row group publishes its down-projection and derivative tasks. A derivative row
group publishes its input-gradient tasks. With `--gradient-chunk=1024`, each
weight-gradient tile advances through 1024-token chunks. A chunk waits for its
own producer tiles and the preceding chunk's accumulator owners, rather than
every token in the layer. Each element retains its FP32 accumulator in global
memory, and only the final chunk applies dequantization and BF16 rounding. The
order of the K=32 MMA instructions is unchanged. The unsplit graph remains a
control selected by `--gradient-chunk=0`.

Each output-writing thread fences before its block signals completion. Dependency
counters use PTX `atom.acq_rel.gpu.global.add`; the last arrival therefore gathers
the prior producers' writes before it publishes children. Queue slots use release
stores and acquire loads. A reserved queue slot is always published without
waiting for a consumer, avoiding a circular queue dependency.

The queue allocates unique consumer tickets with atomic addition. Each worker
keeps one reservation for ready work. While that slot is unpublished, it claims
independent roots with another atomic ticket, allowing newly ready children to
run before roots are exhausted. Completed groups reserve their entire child
range with one atomic operation, then publish each slot with a release store.
Producers never wait for consumers. For this acyclic graph, an unpublished slot
with unfinished work has an earlier producer either executing or ready to run.
Tickets past the known total task count exit without accessing the queue.
Tests include one worker, seven workers, 132 workers, and grids larger than the
occupancy limit. The final queue head must equal task count plus worker count,
the tail must equal task count, and every dependency count is checked. A separate
audited specialization counts every task visit and gradients started before all
roots finish. Small cases shuffle roots to stress out-of-order dependency
arrivals. The production specialization is checked independently after reset;
audit atomics are excluded from timings.

The accepted compute instruction uses FP8 `mma.sync.m16n8k32` with FP32
accumulators and 64-by-64 output tiles. It double-buffers operands using PTX
`cp.async`: wait for the current stage, prefetch the next, compute, and release
the old stage after a block reader barrier. A 16-byte-aligned XOR layout maps MMA
lane quartets to distinct shared-memory banks. Compile-time K=32/64/128 stage
sizes are measured separately. It does not yet have HazyResearch-style independent
loader/consumer/storer warps, TMA, or cross-task prefetch.
`cuda/hopper.cuh` is a separate experimental WGMMA path with two shared-memory
stages and overlap of operand copies with tensor computation. It is not promoted:
see the numerical rejection below.

## Experiments and decision rules

Run on Modal with one H100:

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --sanitize
```

The new Python code only provisions the container, invokes native build/test
commands, and saves their output. All new computation, scheduling, reference
math, and comparisons live in CUDA/C++. No additional Python file is introduced.
Existing upstream Python/Triton modules remain for the baseline until their
replacement passes validation; they are not part of the native test path.

The first run used CUDA 12.8.93, driver 580.95.05 and H100 80GB HBM3. The persistent
kernel used 80 registers/thread, 4100 shared bytes and no spills. All small cases
matched the FP32 cuBLAS-based reference exactly. At M=8192/C=768/H=3072, relative
L2 errors were below 0.000026. Memcheck, initcheck and synccheck reported zero
errors on the small stress matrix.

That first implementation was slow: 45.346 ms fused versus 11.249 ms for six
separate launches of the same tile implementation at M=8192. At M=256 the figures
were 1.658 ms and 0.823 ms. These are warm-cache component measurements, exclude
queue reset, and are not comparisons against the stock trainer or an optimized
FP8 cuBLAS path. They establish a failed performance candidate, not a speedup.

The first revision therefore emits/consumes dual FP8 layouts and uses vector
asynchronous copies for aligned operands. This follows the actual main path,
which already pays to create transposes once instead of re-reading them with
uncoalesced accesses in each GEMM.

Subsequent experiments:

- Modal supplied an H200 for the second H100 request. Its timings are recorded
  separately, not used as a same-device before/after comparison.
- Oversubscribed worker testing exposed a shared task-ID handoff race in the
  original idle-polling queue. A reader barrier fixed it. The test failure is
  preserved in `experiments/cuda-check-20260918T232702Z.log`; the next run passed
  memcheck, initcheck, racecheck and synccheck, including oversized grids.
- Replacing repeated queue CAS attempts with unique tickets improved the H100
  M=8192 case. At 792 workers it measured 2.203392 ms versus 2.314848 ms for the
  six-launch control. At 132 workers it was still 8.079264 ms. Small M=256 was
  slower than the control at every tested worker count. These are component
  medians and do not establish a gain over upstream or an end-to-end speedup.
- The native FP8 WGMMA experiment did not pass the stricter output contract.
  At M=129/C=96/H=160 its raw preactivation relative L2 error was 0.00004391,
  but rounding through BF16 and FP8 amplified the postactivation error to
  0.004414, with 21 of 20,640 values different. Promoting each K=32 partial into
  FP32 and fencing accumulator registers did not remove this discrepancy.
  The control MMA path matches this small case exactly. Do not promote WGMMA by
  raising the 0.0002 output limit. Direct upstream comparison and convergence
  evidence are needed to establish an appropriate alternative contract.

Use `--wgmma` with `--cuda-check` to reproduce that rejected experiment. The
default check selects the validated MMA implementation. Use `--main-shapes` to
check the actual training token counts. Recent runs save source archives beside
their logs and print source SHA-256 digests before compiling.

### Previous accepted-code validation

`experiments/cuda-check-20260918T234040Z.log` and its matching source archive pin
the tested implementation. Modal supplied **H100 NVL**, not the H100 HBM3 device
from the earlier runs. Compare columns within this run, not absolute times across
those machines. The native kernel now copies each task's descriptor into local
state before computing, removing repeated potentially aliased descriptor reads.
It uses 128 registers/thread, 8196 shared bytes and zero spill bytes. Static SASS
inspection found HMMA and asynchronous-copy instructions and no LDL/STL local
memory instructions in the megakernel.

| Tokens per GPU | Fused compute (ms) | Six launches (ms) | Six-node CUDA Graph (ms) | Fused including queue reset (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 4.509584 | 3.926000 | 3.915872 | 4.532928 |
| 32,768 | 9.028208 | 7.768704 | 7.755312 | 9.068592 |
| 49,152 | 13.496064 | 11.459393 | 11.450992 | 13.543808 |

These are warm-cache medians of ten samples after five warmups. All four columns
use this project's tile implementation and prequantized inputs, not the upstream
training implementation. All actual training shapes passed numerical checks.
Memcheck, initcheck, racecheck and synccheck passed the small/ragged/zero-input
stress cases including oversized grids. This candidate is **not a performance
win** at the real shapes: including queue reset, it is about 16–18% slower than
the six-node CUDA Graph control. It remains an isolated native validation path.

### Continuing fine-grained work

The 2026-09-19 UTC experiments preserve rejected candidates as well as passing
ones. `cuda-check-20260919T000823Z.log` measured the first ready-first CAS queue
against FIFO on the same H100 HBM3. It passed correctness and all four sanitizers
but took roughly 19.6/40.1/60.3 ms at the three main shapes, versus
3.49/6.90/10.23 ms for the double-buffered FIFO implementation. Replacing root
CAS with atomic tickets alone did not fix it (`...T001025Z.log`). Keeping a ready
reservation per worker eliminated the ready-head CAS retry loop as well.

`cuda-check-20260919T001350Z.log` first validated chunked weight gradients. Both
weight gradients matched the unsplit MMA computation with zero differing values
at all tested shapes. On the largest main shape, 54,261 gradient tasks started
before all up-projection roots had completed. These are real dependency-driven
tasks within one kernel, not separate per-operation launches. The run still had
an 8-byte register spill and remained slower than its CUDA Graph control.

To reproduce the current stage-size, scheduler and chunk-size comparisons:

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --main-shapes --gradient-chunk=1024 --ablate --sanitize
```

The native default uses K=128 stages and 4096-token gradient chunks.
`--ablate` runs K=32 and K=64 controls, a FIFO control, an unsplit control, and the
other 1024/4096-token chunk size in the same container/GPU. These are fixed-order,
warm-cache component experiments. Every variant computes the same six GEMMs;
the staged and CUDA Graph controls use an unsplit weight-gradient reduction.
Queue reset and partial-accumulator traffic are included in `fused_with_reset`;
input quantization, scale estimation, and full-model work are still absent.

### Latest checkpoint

`experiments/cuda-check-20260919T002026Z.log` compares the choices on one H100
80GB HBM3 with CUDA 12.8.93 and driver 580.95.05. The K=128 and K=64 persistent
kernels were essentially tied at 1024-token chunks, while K=128 made the staged
control faster. K=128 remains selected. Increasing gradient chunks to 4096
reduced the persistent kernel's overhead:

| Tokens per GPU | K=32 / chunk=1024 | K=64 / chunk=1024 | K=128 / chunk=1024 | K=128 / chunk=4096 |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 4.164528 | 3.470352 | 3.455184 | 3.285088 |
| 32,768 | 8.190928 | 6.800480 | 6.774752 | 6.386384 |
| 49,152 | 12.230384 | 10.124512 | 10.170656 | 9.529823 |

Values are milliseconds including queue reset, with ten retained samples and
five warmups. This stage-size comparison is not a comparison against the stock
trainer. The faster K=128 CUDA Graph control still beats the megakernel.

`experiments/cuda-check-20260919T002535Z.log` and its source archive validate the
final checkpoint. The audited and production specializations are independently
run with outputs and FP32 partial accumulators poisoned with NaNs. Both pass the
fixed 0.0002 relative-L2 limit. Chunked weight-gradient outputs have zero differing
values against the unsplit MMA computation. Tests include shuffled roots,
oversubscribed grids, and M=8209/C=96/H=160, which exercises three real-sized
chunks and a 17-token tail.
Memcheck, initcheck, racecheck, and synccheck all pass, including that larger
ragged case. Maximum observed relative L2 error is 0.000047122752. Static SASS
inspection reports 64 HMMA and 28 LDGSTS instructions, with no LDL/STL spills.

| Tokens per GPU | Fused compute (ms) | Fused with reset (ms) | CUDA Graph, same tiles (ms) |
| ---: | ---: | ---: | ---: |
| 16,384 | 3.261344 | 3.297744 | 2.722240 |
| 32,768 | 6.330592 | 6.376512 | 5.240544 |
| 49,152 | 9.475231 | 9.538016 | 7.818128 |

The production kernel uses 162 registers/thread and 32,776 shared bytes, with
three resident blocks/SM and zero spills. At the largest shape, 12,672 of 13,824
weight-gradient tasks started before all up-projection roots finished. The total
graph has 105,984 tasks. This verifies fine-grained execution, but its reset-
inclusive runtime remains about 21–22% slower than the same-tile CUDA Graph.

Promotion requires:

1. Native output/gradient correctness, including small/ragged/zero inputs and
   repeated launches, followed by direct comparison against the pinned upstream
   FP8 implementation (not only the independent cuBLAS reference).
2. Clean CUDA sanitizers and inspected resource/SASS output.
3. Faster representative shapes on the same GPU, including queue initialization,
   scaling and layout costs; compare against upstream and CUDA Graph replay.
4. A complete training step with matching state transitions, then unchanged
   FineWeb validation and same-node multi-run 8-H100 comparisons.

## Quantization, transport, IPC and tokenizer experiments

These experiments are opt-in components, with computation and checks in
CUDA/C++. `train_gpt.py` remains the only new Python entry point. None is enabled
in the canonical trainer, and none establishes a training-time speedup.

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=quantize --sanitize
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=transport --sanitize
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=tokenizer
```

### FP8 scale granularity and inline quantization

`cuda/quantize.cu` measures a BF16-input up-projection with M=16384, N=3072,
K=768 on H100 HBM3. GPU min/max reductions choose symmetric E4M3 scales from
`max(abs(min), abs(max))/448`. The five modes use one scale per operand
(`layer`), per row, per 1x128 segment, per 64x128 tile, or per 64 whole rows.
The inline 64x128 version loads BF16 into shared memory, reduces the range,
quantizes into the existing swizzled FP8 layout, executes PTX MMA, and applies
the scale before adding the next K partial, all in one launch. It recomputes
an operand's scale in every output tile that consumes it.

The unchanged arithmetic gate is relative L2 <=0.0002 against pedantic FP32
cuBLAS on dequantized inputs. Separately, the synthetic quantization screen
requires worst 64x128 input-tile relative L2 <=0.04 and original-BF16-output
relative L2 <=0.06. These new screens do not weaken the arithmetic gate or
replace validation loss. Min/max alone cannot establish output error for
arbitrary dot products: cancellation depends on values, not just their range.
This experiment selects scale granularity, not FP4/FP8/BF16 per-tile precision.

Final evidence: `experiments/cuda-check-20260919T005618491600Z.log` and its source
archive. Times include statistics, quantization and GEMM, with five warmups and
ten CUDA-event samples over the same buffers. Clocks were not locked. Uniform
inputs are normal(0,1); the stress distribution scales each input tile by
`2^(((row/64)*7 + column/128)%25 - 12)`.

| Method | Uniform total (ms) | Different tile ranges total (ms) | Stress screen |
| --- | ---: | ---: | --- |
| One scale per operand | 1.302192 | 1.301696 | Fail: some tiles become all zero |
| One scale per row | 1.424752 | 1.402128 | Fail |
| One scale per 1x128 segment | 4.493424 | 4.558720 | Pass |
| One scale per 64x128 tile | 4.208240 | 4.216416 | Pass |
| One scale per 64 rows | 1.305536 | 1.305872 | Fail |
| Inline 64x128 quantization + MMA | 6.111824 | 6.092976 | Pass |
| BF16 cuBLAS GEMM control | 0.120992 | 0.121008 | No quantization |

All custom MMA arithmetic checks pass. Inline quantization produces bit-identical
BF16 outputs to the separate tile path, including 65x97x160 ragged shapes and
zero inputs. Memcheck, initcheck, racecheck and synccheck pass those small cases.
The inline kernel uses 153 registers/thread, 16,432 static plus 32,768 dynamic
shared bytes, three blocks/SM and zero spills; SASS contains HMMA. These custom
kernels are much slower than the vendor BF16 control. No variant is promoted.

### ZipServ-inspired lossless two-GPU transport

The [ZipServ paper](https://arxiv.org/html/2603.17435v1) and
[author implementation](https://github.com/HPMLL/ZipServ_ASPLOS26/tree/6f8a209bfe46b0905f90d61b0796e352147ff041)
are the references; the checkout is `../ZipServ`. Its TCA-TBE method stores
common BF16 exponents using three code bitmaps and retains full values for
exceptions. The paper targets inference weight compression and fused decode/GEMM.
It does not supply a drop-in training collective.

`cuda/transport.cu` adapts that representation for transport. A GPU histogram
chooses seven consecutive exponents. Every 64-value tile has three 64-bit
bitmaps, byte payloads for common values, two-byte exceptions and a prefix-summed
payload offset. The receiving GPU decodes directly into a BF16 two-rank sum.
This is an independent transport experiment, not a reproduction of ZipGEMM.
All 65,536 BF16 bit patterns round-trip exactly, including NaNs and signed zero;
ragged lengths 1/63/64/65/65543 are covered by all four CUDA sanitizers.
Cross-GPU reductions are also compared bit-for-bit with dense and NCCL controls,
but the sanitizer runs cover the local codec, not NCCL or cross-process IPC.

The initial successful two-H100 run is
`experiments/cuda-check-20260919T004805Z.log`. For 28,311,552 BF16 values (56.6 MB),
normal-distributed data compressed to 73% of original bytes, including metadata.
Dense peer-copy plus sum took 0.249 ms, NCCL reduce 0.206 ms, precompressed
transfer plus decode/sum 0.298 ms, and recompress plus transfer plus decode/sum
0.916 ms. Broad exponent inputs expanded to 120% of the original bytes. The
timing includes metadata copies, payload-size readback, launches and synchronization
on both GPUs. Buffers are reused, with five warmups and ten host-clock samples.
The smaller 4.7 MB case also loses to both dense controls. Compression is rejected
for this path; reduced wire bytes alone do not establish a speedup.

The final source check is `experiments/cuda-check-20260919T005726550234Z.log`,
with NCCL 2.25.1 and an explicit producer synchronization for initial IPC
population timing. It again passes correctness and all four codec sanitizers.
The 56.6 MB normal-data case measured 0.251/0.215/0.315/0.923 ms for dense peer,
NCCL, precompressed, and recompressed paths respectively, preserving the result.

### CUDA IPC and the fast weight-loading reference

The closest match to the user's description is SGLang's
[Weight Cache Daemon](https://www.lmsys.org/blog/2026-08-21-sglang-fast-recovery).
It keeps post-processed weights alive in a separate GPU process and supplies
CUDA IPC mappings to replacement engines. The article reports roughly 495 s
to 0.63 s for one model's weight-loading phase. The exact 80x claim was not
located. Populating the cache still incurs the initial load.

The native IPC experiment actually starts a separate executable with
`posix_spawn`, passes CUDA memory/event handles over a Unix socket, waits for
producer readiness and verifies every value on the GPU. The owner retains its
allocation until the consumer unmaps and acknowledges completion. It compares
first mapping, repeated mapping, retained mapping, and pinned H2D copies, with
the same full-buffer verification included in each timed path.

In the initial successful transport log, a synthetic 248 MB weight buffer took
4.702 ms for pinned H2D plus read, 2.982 ms for remap plus read, and 0.221 ms for
reading an already mapped allocation. First mapping plus read took 4.979 ms.
For a 98 KB token batch, remapping cost 0.202 ms versus only 0.028 ms for pinned
H2D. Mapping a whole resident shard once is the useful candidate; mapping each
batch is rejected. These are GPU-resident synthetic buffer tests, not disk I/O,
checkpoint restoration or measured trainer data-pipeline improvements. Training
weights change each step and require an ownership/update protocol before sharing.

In the final source check, retained-map reads were similar at 0.223 ms for
248 MB, but pinned H2D took 9.186 ms and repeated remapping 3.414 ms, with several
large remapping outliers. Initial population including synchronization took
32.540 ms. Host transfer performance varies across Modal placements, and the
topology diagnostic did not resolve that variation. Both complete sample sets
are retained; a fixed 80x or end-to-end loader improvement is not claimed.

### Snaptokens preprocessing

`cuda/tokenizer.cpp` invokes the existing Rust extension through the CPython C
API; it adds no Python files. Modal builds
[Snaptokens 2d0f8695](https://github.com/coder-2011/snaptokens/tree/2d0f8695f21688a35dd79f3c38372c847c159a99)
using Rust 1.91.1. The GPT-2 tokenizer JSON is pinned to Hugging Face revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, SHA256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
The control is tiktoken 0.12.0, testing both four-thread batch and serial APIs.

Final evidence: `experiments/cuda-check-20260919T005618491592Z.log`. Exact token
IDs and document offsets match for all 2058 documents / 604224 tokens, including
empty strings, whitespace, contractions, numbers, Unicode and the end-of-text
token. Five warmups precede ten measured calls, including C++ output
materialization. On this deliberately repeated synthetic 1.44 MB corpus,
Snaptokens took 9.069 ms, tiktoken serial 175.034 ms, and tiktoken batch 215.702 ms.
The 19.3x ratio is against the faster tiktoken control and reflects warm caches
and these API/output representations. It is not a general text-throughput claim.

The binary can also preprocess JSONL `text` records:
`tokenizer PYTHON_EXECUTABLE TOKENIZER_JSON INPUT_JSONL OUTPUT_BIN`. It emits the
NanoGPT shard header and uint16 tokens with document-start 50256, then verifies
a disk round-trip. The test shard contains 606282 tokens including document
starts. Official FineWeb shards are already GPT-2-tokenized and remain unchanged;
the main loop has no raw-text tokenization to accelerate. Existing vocabulary
tables also remain unchanged.

### Artifact handling and failed attempts

Initial tokenizer linking failed because standalone Python reported a nonexistent
`/install/lib`; the wrapper now resolves the actual libpython location. The first
transport image could not mix CUDA 13 NCCL headers with its pinned CUDA 12 runtime;
both now use NCCL 2.25.1. Failure logs are retained. Concurrent experiments also
exposed second-resolution artifact name collisions at `20260919T005444Z`; the
wrapper now uses microseconds, and both experiments were rerun into unique final
archives. The two original full outputs are retained as
`quant-inline-before-control.log` and `tokenizer-serial-control-first.log`; the
shared timestamp archive is not used as final evidence. Modal's topology-matrix
diagnostic is unavailable; this is recorded separately from mandatory native
peer-access checks.

## SASS/PTX and Nsight Compute optimization (2026-09-19 UTC)

The accepted default now pads the activation-transpose scratch pitch from 64 to
68 bytes, reuses each FP8 quantization for both output layouts, specializes fully
aligned graphs at host dispatch, requests four resident CTAs per SM, and keeps
the inner 128-wide K-stage loop rolled. The general bounds-checked specialization
remains available for ragged shapes. Host dispatch checks every matrix dimension,
operand stride and gradient chunk boundary before selecting the aligned kernel.
Arithmetic and gradient reduction order are unchanged.

### What the machine code and counters showed

Nsight Compute 2025.1.1.0 works on this Modal H100 allocation. The image installs
`cuda-nsight-compute-12-8`; CUDA is 12.8.93 and the driver is 580.95.05. The
wrapper captures PTX using the binary's exact Makefile flags, dumps SASS, and
exports the binary NCU report plus source, metrics, session and details views.
The profile region contains one production megakernel launch after five warmups;
queue reset is outside that region. NCU uses `--set full --clock-control none
--cache-control none --profile-from-start off --launch-count 1`. Ordinary event
timings are collected in a separate invocation before profiling. Timings printed
inside the NCU invocation are retained but are not used for performance claims.
See NVIDIA's [CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
and [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

The PTX FP8 `mma.sync` instruction lowers to FP8-to-FP16 conversions, FP16 HMMA,
and separate FP32 additions on this target/compiler. The baseline executes about
228 million conversion instructions and 228 million FADD instructions, alongside
113 million HMMA instructions. This is not native FP8 WGMMA throughput. DRAM
throughput is only 5.59% of peak in the baseline profile; instruction issue,
register pressure and shared-memory accesses are more useful targets here.

| Counter, M=16384 | Original control | Accepted default |
| --- | ---: | ---: |
| Registers per thread | 162 | 128 |
| Resident CTAs per SM | 3 | 4 |
| Achieved occupancy | 18.69% | 24.86% |
| Tensor pipe active | 21.84% | 26.36% |
| Cycles with no eligible warp | 51.03% | 45.62% |
| Eligible warps per scheduler | 0.75 | 0.95 |
| Source-attributed executed instructions | 1,523,438,320 | 1,403,368,258 |
| Source-attributed excess shared wavefronts | 56,623,104 | 0 |
| Profiled kernel duration | 3.28 ms | 2.74 ms |

These profiles are from separate allocations and explain the mechanism; the
same-device, unprofiled comparison below establishes the timing improvement.
Four transpose `LDS.U8` sites account for 47,185,920 baseline excess shared
wavefronts. Padding eliminates all reported excess shared wavefronts, but alone
saves only about 1% of time. It does not establish that bank conflicts were the
only bottleneck.

The first aligned, unrolled candidate had a 40-byte stack frame and substantial
spill instructions. Rolling the inner K loop reduces its production frame to
8 bytes, with 8 bytes each of static spill stores and loads. Source-attributed
local-memory sectors fall from 18,213,888 to 16,896. The production SASS has
16 static HMMA sites, two LDL sites and two STL sites; it is not spill-free.
Dynamic shared/local counts here are sums of NCU's source-attributed counters,
not inferred byte traffic from static assembly counts.

### Paired final timings and correctness

`experiments/cuda-check-20260919T012837893690Z.log` runs the accepted default and
original control consecutively on the same H100 80GB HBM3. Each uses five warmups
and ten retained CUDA-event samples with reused, warm buffers and unlocked
clocks. The original control retains its prior three-CTA worker count; the
candidate uses four. These are medians, not confidence intervals.

| Tokens | Original megakernel + reset | Default megakernel + reset | Time reduction | Original graph control | Optimized graph control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 3.268544 ms | 2.701136 ms | 17.36% | 2.700304 ms | 2.452128 ms |
| 32,768 | 6.367920 ms | 5.300800 ms | 16.76% | 5.230480 ms | 4.747728 ms |
| 49,152 | 9.510384 ms | 7.922160 ms | 16.70% | 7.795968 ms | 7.069808 ms |

Both graph controls use the same tile arithmetic as their corresponding fused
kernel. The optimized graph remains faster than the megakernel. Inputs are
prequantized, and this benchmark does not include full-model training or input
scale/quantization costs. No leaderboard improvement is established.

All eight output/layout checks pass against the existing reference with the
unchanged relative-L2 limit of 0.0002; the maximum reported error is 0.000047123.
Chunked weight gradients exactly match unsplit MMA results. Production and
audited specializations pass independently. Memcheck, initcheck, racecheck and
synccheck report zero errors/hazards on both aligned and ragged quick cases,
including one-worker and oversubscribed grids and multiple gradient chunks.
The canonical Python trainer, token stream and evaluation remain unchanged.

The separate quantization experiment also passes its arithmetic checks and all
four sanitizers with the current shared CUDA headers. Its regression log is
`experiments/cuda-check-20260919T013303716271Z-92873cbe.log`.

### Rejected candidates and retained evidence

- Replacing the compiler's FP8 lowering with converted FP16 MMA that directly
  accumulates into the running sum changes rounding. At M=49152, dW2 relative-L2
  is 0.00022324526, above 0.0002. It is rejected, remains opt-in as
  `--variant=direct`, and is never enabled by the default build. Its first two
  main shapes pass, which is why all three shapes are required.
- The aligned K64 unrolled candidate takes 3.328816 ms including reset at
  M=16384, versus 2.732992 ms for the rolled K128 candidate on another allocation.
  This is exploratory evidence, not a same-device speed ratio; K64 is not promoted.
- The aligned unrolled version is faster than the original but spills heavily;
  its staged control is also slower. Final reporting retains both the original
  and optimized graph controls instead of claiming a win against that weaker
  intermediate control.

Compressed profile artifacts contain PTX, SASS, the NCU report and decoded views:

- Baseline: `experiments/profile-20260919T010434440014Z.tar.gz`.
- Padding only: `experiments/profile-20260919T011439303866Z.tar.gz`.
- Aligned/unrolled: `experiments/profile-20260919T012248560690Z.tar.gz`.
- Final default: `experiments/profile-20260919T012948492643Z-de08ee7d.tar.gz`.

Each `cuda-check-*` log has a corresponding exact source archive. A concurrent
loop/K64 experiment still collided at timestamp `20260919T012456992498Z` despite
microsecond timestamps. That profile archive contains the loop variant; the
shared `cuda-check` log contains K64, whose artifact-directory creation then
failed. Complete original outputs are preserved as `profile-loop-initial.log`
and `profile-k64-initial.log`. The final default was rerun into the unique archive
above. Artifact names now also contain a random UUID suffix.

Reproduce the accepted comparison and capture a profile:

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --main-shapes --kernel-compare --sanitize
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --profile
```

Use `--variant=control --kernel-compare --main-shapes` to reverse candidate order.
Use `--inspect-profile PATH_TO_NCU_REPORT` to regenerate decoded views on a CPU
Modal container. This requires extracting the compressed profile archive first.

## Work still required

The architecture/training target has advanced to PR #360. Follow `FRONTIER.md`
for its mixed-width attention, 2816-wide MLP, E5M2 gradients, sampled loss,
ANVIL/Adam, sparse n-gram table and exact 1194-step schedule. The older NorMuon
and 3072-wide contract above is historical control evidence, not the new target.


- Finish the Hopper compute pipeline: validated WGMMA/TMA, independent load/compute/
  store roles, and further measured tile/worker choices.
- Reduce scheduler, register, and partial-accumulator overhead while preserving
  the ordered weight-gradient reductions.
- Move FP8 scale reduction/quantization and dynamic scale statistics into the DAG.
- Connect the native attention/QKV transforms to projections, gains and gates;
  replace scalar attention with validated tensor-core tasks. The new component
  checks and their explicit limits are in `FRONTIER.md`.
- Add BF16 matrix multiplication for O projections and ANVIL, then port the
  remaining head/loss, optimizer and full-model routing nodes.
- Port schedule/data ownership and device communication; CUDA graph launch of
  separate NCCL calls alone would not meet the full fusion goal.
- Replace the active Python/Triton training implementation only after the native
  path can execute the real model. Keep `train_gpt.py` as the sole Python wrapper.

There is no full-model CUDA training result or speedup claim yet.

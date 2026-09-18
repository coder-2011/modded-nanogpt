# CUDA/PTX megakernel research and implementation ledger

Research started 2026-09-18. The objective is the main NanoGPT speedrun: minimum
training time while preserving its token streams and reaching the validation
target. A component benchmark is not a speedrun result.

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
group publishes its input-gradient tasks. Weight-gradient tasks wait for all
their producer rows. There is no grid-wide barrier between these operations.

Each output-writing thread fences before its block signals completion. Dependency
counters use PTX `atom.acq_rel.gpu.global.add`; the last arrival therefore gathers
the prior producers' writes before it publishes children. Queue slots use release
stores and acquire loads. A reserved queue slot is always published without
waiting for a consumer, avoiding a circular queue dependency.

The revised queue allocates unique consumer tickets with atomic addition. A
consumer may wait for its slot to be published, but a producer never waits for a
consumer and only publishes children after finishing its own tile. Roots are
published before launch. Thus an empty queue with unfinished work has a producer
already executing, rather than depending on a block that has not obtained work.
Tickets past the known total task count exit without accessing the queue.
Tests include one worker, seven workers, 132 workers, and grids larger than the
occupancy limit. The final queue head must equal task count plus worker count,
the tail must equal task count, and every dependency count is checked.

The accepted compute instruction uses FP8 `mma.sync.m16n8k32` with FP32
accumulators and 64-by-64 output tiles. It does not yet have HazyResearch-style
independent loader/consumer/storer warps, TMA, or cross-instruction prefetch.
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

### Latest accepted-code validation

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

Promotion requires:

1. Native output/gradient correctness, including small/ragged/zero inputs and
   repeated launches, followed by direct comparison against the pinned upstream
   FP8 implementation (not only the independent cuBLAS reference).
2. Clean CUDA sanitizers and inspected resource/SASS output.
3. Faster representative shapes on the same GPU, including queue initialization,
   scaling and layout costs; compare against upstream and CUDA Graph replay.
4. A complete training step with matching state transitions, then unchanged
   FineWeb validation and same-node multi-run 8-H100 comparisons.

## Work still required

- Finish the Hopper compute pipeline: WGMMA/TMA, double buffering, load/compute/
  store overlap and measured tile/worker choices.
- Split weight-gradient reductions into independently schedulable token ranges
  without changing BF16 rounding/accumulation semantics accidentally.
- Move FP8 scale reduction/quantization and dynamic scale statistics into the DAG.
- Port and validate attention forward/backward, head/loss and optimizer nodes.
- Port schedule/data ownership and device communication; CUDA graph launch of
  separate NCCL calls alone would not meet the full fusion goal.
- Replace the active Python/Triton training implementation only after the native
  path can execute the real model. Keep `train_gpt.py` as the sole Python wrapper.

There is no full-model CUDA training result or speedup claim yet.

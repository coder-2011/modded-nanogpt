# Current model and training target

Checked 2026-09-19 UTC. The target is the fastest open main-track submission
found with an explicit maintainer legality ruling: [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360),
commit `c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7`. It reports 39.914 seconds over
18 runs on eight H100s. That result has not been reproduced here, and the PR is
still open. The latest merged reference remains `bc3a0c2`, record 91, listed at
1.126 minutes. These are different statuses and different model/training recipes.

[Keller Jordan's explicit ruling](https://github.com/KellerJordan/modded-nanogpt/pull/360#issuecomment-5565930470)
approves the described techniques as legitimate, while reserving record acceptance
until he reproduces the result. [ClassicLarry's latest comment](https://github.com/KellerJordan/modded-nanogpt/pull/360#issuecomment-5726720656)
says he intends to integrate and validate the changes piece by piece.

`frontier.lock.json` pins the exact trainer, five kernel modules, dependencies,
Dockerfile and launch script by SHA256. Their first sixteen hash digits match the
PR's published source hashes. The untouched reference checkout lives at
`../modded-nanogpt-frontier`. The native implementation remains CUDA/PTX, with
`train_gpt.py` as its only new Python glue. Upstream reference modules remain in
the separate checkout.

```sh
# Recreate the exact reference if this worktree is absent:
git fetch origin pull/360/head:refs/remotes/origin/pr-360
git worktree add --detach ../modded-nanogpt-frontier c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7
python train_gpt.py --check-frontier

# In the reference's GPU/dependency environment, launch its original eight-rank trainer:
python train_gpt.py --frontier-reference
```

The launcher verifies the pinned files and requires the default step count and
enabled assertions. It does not provision GPUs or install dependencies. Use the
reference Dockerfile and [reproduction notes](https://github.com/KellerJordan/modded-nanogpt/blob/c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7/records/track_1_short/2026-08-30_ANVIL2/README.md).
Its patched FA3 requires CUDA runtime 13, the pinned torch 2.10 cu128 environment
and the kernel revision recorded in the lock. The CUDA 12.8 image used for our
standalone MLP checks is not the full reference environment.

`bash run.sh` now selects the verified PR #360 reference on all eight ranks.
The wrapper preserves the workspace data directory (or an explicit `DATA_PATH`)
when entering the reference checkout. The original Python training body in this
branch remains the merged-record control, available by invoking `train_gpt.py`
directly under torchrun without `--frontier-reference`. The default `--cuda-check`
selects the frontier MLP component. These are separate full-reference and native
component execution paths; the reference does not use our native MLP yet.

## Architecture extracted from executable source

Read the pinned source rather than copying its older descriptive comments.
Several comments mention 1188 steps or old window settings, while the actual
constants and schedule constructor produce the values below.

| Surface | PR #360 contract | Native status |
| --- | --- | --- |
| Residual stream | Width 768, eleven numbered layers, six heads | Dimensions pinned |
| MLP | Hidden width 2816, bank order 0,1,2,3,4,5,6,8,11,9,10; layer 8 has two parallel MLPs sharing its normalized input | Individual FP8 training branch and BF16 evaluation forward implemented; full routing absent |
| Attention | Active at layers 0,1,2,3,5,8,10; QK width 128 at 3/10 and 64 elsewhere; value width 64 at 1/8 and 128 elsewhere | Connected FP8 training forward/backward and BF16 evaluation forward, including projections and gates; full routing remains; FA3/compiled-trainer parity unverified |
| Skipped layer | Layer 7 omits both sublayers but retains residual scaling/injection and saved state; layers 4/9 omit attention; layer 6 already lacks attention | Not implemented |
| MUDD and gates | Grouped post-loop mixing, learned gates, exact injection sites and saved-activation reuse | Not implemented |
| Embeddings | 84,602,880 × 768 logical learned n-gram table, split into bigram/trigram halves; eight shards, 8192 sign rows; four value-embedding planes | Not implemented |
| Matrix optimizer | ANVIL twin velocity rails, six spectral maps, per-bank grouping/annealing, terminal blends and norm restoration | Rank-local update body implemented in the persistent worker; training schedules, gradient exchange and model integration remain |
| Other optimizer state | Parameter-specific Adam rules; sparse touched-row exchange/update; value/n-gram cadence changes at step 336 | Not implemented |
| Loss | Sampled softcapped CE early, full vocabulary later; MTP and prefix auxiliary schedules; full-vocabulary validation | Not implemented |
| Timing | Reset warmup state, charge first data fetch, prefix-table construction, table updates, final weight blends and required validation row gathering | Not implemented |

A matching individual MLP does not establish matching architecture, initialization,
training trajectory or model quality. Full native training is not available yet.
The original frontier reference is the runnable complete model today.

## Exact schedule

The source defaults to 1122 base scheduled steps, adds 52 inside stage 2, then
runs 20 extension steps: **1194 total**, with embedding/head untie at step 1175.
Boundaries below are zero-based and end-exclusive. They were derived by executing
only the pinned schedule definitions with a CPU tensor placeholder, without
importing the trainer or changing its arithmetic.

| Step range | Tokens per GPU | Document cap | Short/long attention blocks | Stage LR multiplier |
| --- | ---: | ---: | --- | ---: |
| 0–319 | 16384 | 896 | 1 / 3 | 1.0 |
| 320–680 | 32768 | 2048 | 3 / 7 | 1.52 |
| 681–1106 | 49152 | 3072 | 5 / 11 | 1.73 |
| 1107–1173 | 40960 | 3072 | 5 / 11 | 1.579266707473229 |
| 1174–1193 | 16384 | 3072 | 6 / 13 | 1.0 |

Blocks contain 128 tokens. A separate virtual attention-segment cap is 2560.
The cooldown starts at step 234, uses fraction 0.80 and ends at LR floor 0.30.
Sampled candidates are 10240 in stages 0/1, then 14336/14336/24576 over the three
parts of stage 2. Sampling ends at step 1107, so stages 3/4 use full softmax despite
the unused 32768 entries in `SNS_CANDIDATES_PER_STAGE`. Prefix weight is 0.25 in
stage 0, then 0.20/0.15/0.10/0.05 over stage 1, then zero. MTP weights and stage
endpoints are recorded in the lock.

ANVIL uses the exact six `ANVIL_MAPS` coefficient triples from the pinned trainer,
not the older NorMuon/Polar Express coefficients. Preserve the fast-rail beta
schedule, slow beta 0.98, blend weight 0.4385 and engagement at step 514. Tail
averaging, final blending and norm restoration are part of the trained model,
not optional evaluation cleanup. Port them before claiming parity.

The validation contract remains the first 10,485,760 canonical FineWeb tokens
and mean CE ≤3.28. PR #360 uses full-vocabulary CE for validation and predates
merged canonical masking #350. First reproduce its exact evaluation, then test
any #350 combination as a separate change. Never substitute sampled validation
or change token IDs to improve the reported result.

## Native MLP transition

The default CUDA validator now consumes `cuda/frontier.cuh` for width and all
four distinct training batch shapes. It uses E4M3 inputs, weights and post
activations, with E5M2 incoming gradients and dpre. PTX mixed-format MMA handles
E5M2×E4M3 and E4M3×E5M2. Backward reconstructs the derivative in FP32 from stored
FP8 post activations, rounding to BF16 only after the product, as the pinned
`RECON_SQRT` path does. The independent cuBLAS reference decodes each operand
according to its descriptor. Gradient layout comparisons decode E5M2 as E5M2.

The benchmark still accepts prequantized inputs and supplied scales. It does not
yet implement scale refresh, the residual post-lambda fold and its scalar
gradient, parallel-branch accumulation, or producer/optimizer integration. The
source uses static input scale 1/16, incoming-gradient scale 1/64, post headroom
1.03, dpre headroom 1.25 and weight headroom 1.12 after sixteen exact-scale
bootstrap calls. Those are pinned for the port; generic random component fixtures
use deliberate non-unit stress scales and are not a training simulation.

`--variant=merged` retains the old width-3072, all-E4M3 benchmark. Previous timing
gains apply to that old workload only. `--kernel-compare` now requires the frontier
default/control pair, so it cannot accidentally compare different arithmetic.
The WGMMA and rejected direct-accumulation experiments remain legacy-only.

## Other open PRs

The current discussion snapshots are in
`experiments/frontier-pr-audit-20260919.json`. "No ruling found" does not mean a
technique is illegal, only that it fails the requested confirmed-legality filter.

| PR | Finding | Decision |
| --- | --- | --- |
| [360](https://github.com/KellerJordan/modded-nanogpt/pull/360) | Explicit maintainer approval of described techniques; fastest reported main-track result in this audit | Pinned target |
| [367](https://github.com/KellerJordan/modded-nanogpt/pull/367) | Longest exact-match retrieval, 47.2s claim; no maintainer ruling in current discussion | Do not adopt as confirmed |
| [366](https://github.com/KellerJordan/modded-nanogpt/pull/366) | Host-RAM n-gram table, 67.87s claim; no maintainer ruling in current discussion | Do not combine with #360 by assumption |
| [363](https://github.com/KellerJordan/modded-nanogpt/pull/363) | FP8 input-gradient/kernel tuning against an older architecture; no maintainer ruling in current discussion | #360 already specifies full FP8 gradients; old tile choices need new profiling |
| [347](https://github.com/KellerJordan/modded-nanogpt/pull/347) | Maintainer requests rebase/simplification of TailEMA | #360 already has terminal averaging; do not apply a second incompatible blend |
| [348](https://github.com/KellerJordan/modded-nanogpt/pull/348) | Layer dropout deferred for complexity versus 0.2s benefit | Preserve #360's fixed layer routing |
| [346](https://github.com/KellerJordan/modded-nanogpt/pull/346) | Ember deferred by maintainer, loss/speed tradeoff and stale base | Preserve ANVIL/Adam |
| [336](https://github.com/KellerJordan/modded-nanogpt/pull/336) | Maintainer reproduction was slower with worse loss | Not adopted |
| [254](https://github.com/KellerJordan/modded-nanogpt/pull/254) | Maintainer requires p<0.01 because normalization changes the learning algorithm | Do not replace ANVIL's normalization with it |
| [231](https://github.com/KellerJordan/modded-nanogpt/pull/231), [123](https://github.com/KellerJordan/modded-nanogpt/pull/123) | Older prefetch proposals have complexity or cross-provider reproducibility concerns | Port #360's pinned ownership/transfer ordering first |

Systems-equivalent optimizations may be tested without changing this recipe.
Any training-method combination needs its own full-run loss evidence and
same-hardware timing against the pinned frontier, counting all runs. Neither
maintainer legality approval nor component correctness substitutes for that test.

## Validation of this transition

`experiments/cuda-check-20260919T014333409640Z-195f9552.log` records the native
frontier component at M=16384/32768/49152/40960, C=768, H=2816 on H100 80GB HBM3
with CUDA 12.8.93. All outputs pass the existing relative-L2 limit of 0.0002
(maximum observed 0.000041781484). Ordered chunked weight gradients exactly match
the unsplit MMA path. Memcheck, initcheck, racecheck and synccheck all pass the
aligned/ragged quick cases. Production uses 124 registers, four resident CTAs
per SM and no reported stack or spills in this build.

| Tokens | Megakernel + queue reset | Same-tile CUDA Graph |
| ---: | ---: | ---: |
| 16384 | 3.839248 ms | 3.387024 ms |
| 32768 | 7.586960 ms | 6.644080 ms |
| 49152 | 11.347857 ms | 9.893856 ms |
| 40960 | 9.492640 ms | 8.278032 ms |

These are warm-buffer medians with five warmups and ten samples, not full-model
training times. The graph control still wins. Mixed-format MMA dispatch needs
further optimization; no speedup against the old all-E4M3 workload is claimed.
The archived source includes the exact code used for these numbers.

The final default development-shape check also passes at width 2816 in
`experiments/cuda-check-20260919T015054985793Z-b478cfa3.log`. Its archive matches
all current CUDA/build sources, Python glue and the frontier lock.

The retained merged-mode regression passes in
`experiments/cuda-check-20260919T014852573835Z-5413bf6f.log`. Source verification,
header/lock agreement, and intercepted reference-launch checks are recorded in
`experiments/frontier-contract-20260919.log`. Separate temporary-copy checks
verified that a changed kernel hash and a changed training step count are
rejected. Shell syntax and Python parsing pass. These checks did not run the
full eight-GPU reference or validate a complete native training trajectory.

The first attempted GPU build was launched before `frontier.cuh` generation
completed and failed on that missing dependency. Its log is retained as
`cuda-check-20260919T014229253914Z-fc65d91b.log`; it contains no correctness or
performance result. The successful run above supersedes it.

## Native attention port

`cuda/attention.cuh` implements the three live QK/V geometries (64/128, 64/64,
128/128), causal variable-length masking, inclusive left windows, online FP32
softmax, and backward with exclusive ownership of dQ, dK and dV. It accepts the
current YaRN softmax scale rather than assuming `1/sqrt(head_dim)`. Paired heads
use the reference's `[2*T, 3, D]` layout and doubled document offsets.

`cuda/qkv.cuh` ports the Q/K RMS normalization, BF16 rotary factors, paired-head
factor parity, shifted stationary key channels, auxiliary-value addition, and
the backward BF16-to-E4M3 conversion into both row and transposed layouts. The
key shift crosses document boundaries exactly as in the pinned implementation.
It does not replace that operation with a document-local shift.

These operations execute as device-callable tasks inside `nano::megakernel`,
using the same acquire/release dependency queue as the MLP. In the connected
validator, QKV transforms release attention forward, each forward tile releases
its dQ tasks, all dQ tasks release dK/dV, and completed attention gradients
release the QKV backward pack. A final dependent GEMM checks heterogeneous
dispatch. That GEMM is a scheduler sentinel, not an output projection.

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=attention --sanitize
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=attention --profile
```

The independent C++ reference materializes probabilities in FP64 and accumulates
backward by query, independently of the CUDA online recurrence and key-owned
gradient loops. It follows the saved-BF16-output delta convention. Numerical
checks require relative L2 <= 0.0002 or maximum absolute error <= 0.00002 for
near-zero results. BF16 casts, gradient transpose, task visits, poisoned-buffer
reuse and instrumented/production equivalence are checked separately. Cases
cover ragged and one-token documents, window zero, paired boundaries, uniform
and sharp logits, zero-norm Q/K rows, and one-worker/oversubscribed execution.
This establishes mathematical component correctness, not bitwise or training
parity with the patched FA3 binary.

The scalar attention baseline remains slow. On H100 80GB, CUDA 12.8.93, at
16384 tokens, six heads, window 384, and document lengths capped at 896:

| QK / V width | Three separate attention launches | Persistent attention + reset + sentinel |
| --- | ---: | ---: |
| 64 / 128 | 10.351936 ms | 19.927296 ms |
| 64 / 64 | 9.161296 ms | 18.126368 ms |
| 128 / 128 | 12.472480 ms | 24.969151 ms |

These are same-allocation warm-buffer medians, five warmups and ten retained
CUDA-event samples, from `cuda-check-20260919T032338370437Z-67a8b46a`. This timing
excludes QKV transforms/projections and post-attention gates. Both timed paths
use the new scalar arithmetic, not FA3. The persistent side additionally includes
queue initialization and one 64x64 GEMM. All four sanitizers pass that run.

Final attention/QKV validation is recorded in
`cuda-check-20260919T032848168561Z-c53ea42e`, including the live non-paired 64/64
geometry and paired 64/128 without auxiliary values. Its source archive captures
the attention checkpoint's native kernels, build file, Python glue and frontier lock. All four
sanitizers pass. The largest reported relative-L2 error is 0.000114196161 on an
adversarial sharp-logit attention gradient. No tolerance was relaxed.

`profile-20260919T032750660525Z-c300a7bd.tar.gz` contains PTX, SASS, the NCU report
and decoded views for the production worker. The profile reports 128 registers
per thread, 24.95% achieved occupancy, 65.05% cycles with no eligible warp,
and 6.33 long-scoreboard stall cycles per issued instruction. Tensor-pipe activity
rounds to 0.00% because attention is scalar; the static HMMA instructions belong
to the GEMM branch. This points to the attention compute/memory pipeline and
worker resource budget, not DRAM bandwidth (1.70% of peak). Profiler timings are
diagnostic, separate from the ordinary event measurements above. The profile
predates only the final host-side test additions, not a kernel change.

The initial dispatch change accidentally passed global GEMM descriptors by
reference, slowing the existing MLP. Restoring local descriptor copies returned
its four main shapes to 3.829072, 7.519824, 11.242960 and 9.399744 ms including
reset, with all numerical and sanitizer checks passing in
`cuda-check-20260919T032338666606Z-4f85ddb8`. The slower intermediate run is retained
as `cuda-check-20260919T031933744608Z-22b96716`. These separate-allocation timings
are regression evidence, not a controlled optimization speedup claim.

The first attention build failed because `math_constants.h` was not included.
Its source and error log remain in `cuda-check-20260919T031314253832Z-66abb5c3`.
Successful runs supersede it; it contributes no performance result.

Still missing: QKV/O projection integration and gain gradients, XSA/head-gate
backward, tensor-core attention, model routing/normalization, embeddings, loss,
optimizer, scale updates, data ownership and distributed execution. The native
path cannot train or rank the complete model yet. BF16 matrix multiplication for
the O projections and ANVIL's matrix maps is now implemented below, but those
operations still need to be connected to their full forward/backward and optimizer
state transitions.

## BF16 matrix primitive

`cuda/bf16_gemm.cuh` adds device-callable 64x64 tasks using PTX
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, two shared-memory operand
stages, and `cp.async` on aligned contiguous inputs. Strided operands use
coalesced reads along their contiguous dimension. Input/output addressing is
64-bit. The existing FP8 MLP remains a separate arithmetic path.

The descriptor supports pre-GEMM BF16 weight scaling for O projections, FP32
accumulation, an optional BF16 product rounding before addition for ANVIL's
split path, and a fused scaled-matrix addition for its other path. Symmetric
Gram tasks own one triangle and mirror their outputs without duplicate writes.
Output may alias the additive C operand, but not either multiplicative input.

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=bf16 --ablate --sanitize
```

The validator checks an independent unpack-to-FP32/cuBLAS pedantic reference,
BF16 rounding, exact symmetry, a dependent identity GEMM, poisoned-buffer reuse,
one-worker/oversubscribed scheduling, and in-place additive output. It covers
all four input-layout combinations, ragged tails, single-element matrices,
the 768/384-wide O projection geometries, and ANVIL-sized Gram/products. No
correctness threshold was loosened for the load-layout optimization.

The first same-GPU candidate/control comparison is
`cuda-check-20260919T034326458867Z-71addc3b`, on H100 80GB and CUDA 12.8.93:

| M / N / K, operation | Original persistent + reset | Coalesced persistent + reset |
| --- | ---: | ---: |
| 16384 / 768 / 768, scaled projection | 0.211952 ms | 0.209696 ms |
| 16384 / 768 / 384, scaled projection | 0.144512 ms | 0.143632 ms |
| 768 / 768 / 2816, transposed symmetric Gram | 0.688896 ms | 0.545808 ms |
| 2816 / 768 / 768, split product/add | 0.171120 ms | 0.169616 ms |

The Gram result is a 20.77% time reduction in this controlled component run.
The other persistent differences are small. The faster separate-launch controls
remain in the raw log, and this is not a speedup against cuBLAS or the complete
frontier trainer. Both variants pass the numerical checks, with maximum relative
L2 0.0000564413038. The coalesced candidate passes all four sanitizers. The
first BF16 implementation and its checks are retained in
`cuda-check-20260919T034022871321Z-2af75848`.

Final BF16 checks, including the worker compiled with attention and BF16 support
together, pass in `cuda-check-20260919T034853939196Z-9c0ebc97`. That source archive
matches the current native source/build files, Python glue and frontier lock.
Reloading the immutable completion signals only on the controller reduced this
combined candidate's spill loads/stores from eight to four bytes per thread.
The combined specialization still has one static LDL/STL pair, 128 registers and
four resident CTAs per SM. The dedicated BF16 specialization reports no spills.
This is a resource improvement, not an established full-model timing improvement.
An additional shared-slot reload did not remove the remaining spill and was
rejected. Its evidence is `cuda-check-20260919T035104616570Z-1c5c0ad8`.

The same final shared headers pass the full MLP main-shape checks and all four
sanitizers in `cuda-check-20260919T034854781208Z-48b4dc04`, and attention/QKV checks
and all four sanitizers in `cuda-check-20260919T034855970984Z-9866c50a`. The BF16
validator also checks that its outputs remain identical when attention support
is compiled into the worker. These still are separate component graphs, not a
complete native training step.

## Rank-local ANVIL update body

`cuda/anvil.cuh` and `cuda/anvil_graph.cuh` now run the complete rank-local
matrix-update body in one persistent launch. The graph starts with BF16 reduced
gradients and FP32 fast/slow velocity state, builds the unnormalized Gram,
normalizes from its trace, executes the six pinned matrix maps, updates lane
energy, restores the update norm, and applies cautious decay plus the parameter
update. Each matrix has independent dependencies, so different banks can progress
concurrently. This body is not yet connected to the model's backward pass,
reduce-scatter/all-gather, step schedule, Adam updates or final parameter blends.

There are 24 stages and 18 BF16 products per matrix. GEMMs remain 64x64 tensor-core
tasks. Elementwise work is partitioned into 1024-element tasks, and lane reductions
have one CTA per lane. The actual per-rank workload (six 256x768 QK matrices, two
768x768 VO matrices, three 2816x768 MLP matrices) has 198 matrix products, with
independently shuffled initial tasks and persistent per-matrix scalar addresses.
Changing the momentum, fast beta, blend, decay and learning-rate scalars needs no
graph rebuild.

The port preserves executable details that differ from loose descriptions:

- The first normalization uses the trace of the BF16-rounded unnormalized Gram.
  Matrices with more than 1024 rows round the product to BF16 before the additive
  map term. Square matrices use the left-multiply branch.
- The lane reduction runs over columns when rows >= columns, and over rows
  otherwise. It reduces the shorter dimension, despite the reference comment
  saying "LONGER". Energy state and the reciprocal-square-root gain are separate.
- The cautious-decay gate reads the slow velocity. The parameter and mantissa
  buffers store the high and low 16 bits of an FP32 shadow. The high half is
  truncated, not rounded to BF16. The decay scalar already contains a learning
  rate and is multiplied by the per-matrix learning rate again, as in the source.
- Lerp follows [PyTorch 2.10's compiled decomposition](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_refs/__init__.py#L4900-L4927),
  including its alternate base for weights at or above 0.5. The slow-rail and
  energy coefficients are 0.02f and 0.1f, preserving the reference's Python
  constant evaluation. Minimum clamps propagate NaNs instead of replacing them.

`cuda/anvil.cu` uses an independent CPU scalar implementation, FP64 reduction
sums and pedantic FP32 cuBLAS products with explicit BF16 rounding. The composed
cascade gate was set to 1% relative L2 before the first run, with a small absolute
fallback for near-zero outputs. The stricter existing BF16 GEMM gate is unchanged.
This approximate oracle does not establish bitwise parity with the pinned
PyTorch/Triton trainer or convergence parity. Velocity state and the FP32 shadow
update are checked bitwise. Staged kernels, serial and concurrent CUDA Graphs,
instrumented/production persistent workers and the combined worker must produce
identical outputs from the same state.

Checks cover three evolving updates, changed resident scalars and gradients,
poisoned work buffers, shuffled roots, task visits, one-worker execution, ragged
wide/tall/square inputs, zero gradients and both sides of the 1024-row split.
The validator also records an inherited numerical limitation: the independent
arithmetic reference and native cascade both diverge on the tested 1x1 input
and 17x33 rank-one input. These are explicit non-finite classification checks,
not successful finite training updates. The transposed 33x17 rank-one case remains
finite and passes the ordinary numerical gate. No stabilizing clamp or alternate
map was introduced. Whether this edge occurs in a real training trajectory is
unverified.

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=anvil --ablate --sanitize
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --experiment=anvil --profile
```

The same-GPU idle-delay experiment uses H100 80GB HBM3, CUDA 12.8.93, five warmups
and ten retained event samples. Each path starts from the same saved optimizer
state and runs repeated stateful updates with warm buffers. Persistent timing
includes queue reset. The concurrent CUDA Graph has independent matrix branches,
while the serial control intentionally orders them. All controls use the same
native tile math, not the reference trainer's faster library kernels.

| Workload, 528 workers | Default 1024 ns idle | Candidate 64 ns idle | Concurrent CUDA Graph, default build |
| --- | ---: | ---: | ---: |
| One 256x768 matrix | 1.081328 ms | 1.101184 ms | 0.507888 ms |
| One 768x768 matrix | 2.617216 ms | 2.623776 ms | 1.136880 ms |
| One 2816x768 matrix | 7.304304 ms | 7.307440 ms | 3.504624 ms |
| One matrix of each shape | 7.404928 ms | 7.489600 ms | 3.589104 ms |
| Full rank-local 11-matrix batch | 9.504608 ms | 9.170848 ms | 4.284352 ms |

The full-batch samples overlap substantially, and the shorter delay is worse on
the other 528-worker cases. It is not promoted. Reducing workers to 132 or 264 also
fails to improve the full batch. The serial Graph takes 15.818480 ms there, but
using it alone would overstate the persistent kernel's performance. The fairer
concurrent Graph remains more than twice as fast. This is an optimizer component
result, not a full-model or leaderboard result.

The profiler archive `profile-20260919T040954882457Z-c59d91be.tar.gz` contains
PTX/SASS and the full Nsight Compute report for one 2816x768 update at 528 workers.
The worker has 128 registers/thread, 32776 shared bytes/CTA, no reported spills,
24.85% achieved occupancy, 93.81% cycles with no eligible warp and 0.67% tensor-pipe
activity. Sleeping accounts for 41.42 of 63.73 warp cycles per issued instruction.
This led to the bounded idle-delay experiment, whose negative result shows that
sleep-stall share alone does not identify the critical path. Profiler timings
are diagnostic, not the event timings in the table.

Initial validation and all four sanitizers passed in
`cuda-check-20260919T040652485566Z-aaf6bc31`. The two scalar non-finite diagnostic
runs are preserved as `cuda-check-20260919T040921575094Z-6a73dbd6` and
`cuda-check-20260919T041201316461Z-e2ed8bc3`. Expanded validation and all four
sanitizers passed in `cuda-check-20260919T041342080683Z-22774a55` before the
concurrent Graph control was added. Shared-dispatch regression checks and all
four sanitizers passed for MLP in `cuda-check-20260919T040954882545Z-231da927`,
attention/QKV in `cuda-check-20260919T040954882509Z-0155f343`, and BF16 GEMM in
`cuda-check-20260919T040954882490Z-70af11a5`. Their non-ANVIL arithmetic remains
unchanged by the optimizer-specific idle-delay option and NaN clamp correction.

The complete idle comparison and all four sanitizers for **both** variants are
in `cuda-check-20260919T041538442378Z-d20c1784`. That comparison and the profiler
predate the NaN-preserving clamp correction. The corrected kernel passes all numerical,
state/replay and sanitizer checks in `cuda-check-20260919T041839305095Z-b481f800`.
Maximum relative-L2 errors are 0.00817729591 for
the equalized update, 0.00475073883 for lane energy and 0.000451144754 for the
parameter shadow, including the finite rank-one stress case. Velocity and the
shadow arithmetic on the native update remain bit-exact. The final full batch
contains 62158 tasks, 253 dependency groups and 198 GEMMs. Its persistent + reset
median is 9.546208 ms versus 4.276944 ms for the concurrent Graph. Dedicated and
combined workers still have 128 registers and no spill loads/stores. These final
timings confirm the performance gap and do not establish a frontier speedup.

Final planner review added a dimension ceiling that includes tile-padding
arithmetic, with rejection checks before any large task allocation. The first
host-test build failed because nvcc could not deduce the braced range type,
retained in `cuda-check-20260919T042455536240Z-0ac16fa2`. Giving the test array an
explicit type fixes the build. The final committed source passes the complete
numerical/replay suite and all four sanitizers in
`cuda-check-20260919T042600741821Z-9dabf0c4`. Every source-archive member matches
the final native source/build files, Python glue and frontier lock. Numerical
maxima are unchanged. On its H100, the full batch takes 9.319728 ms persistent
plus reset versus 4.327216 ms for the concurrent Graph, again showing no speedup.

## Connected training-attention layer

`cuda/attention_layer.cuh` now builds one persistent graph for the training
attention-call boundary. It connects cached E4M3 QKV projections, Q/K RMS
normalization and rotary transforms, paired or ordinary causal attention,
auxiliary values, optional XSA and head gates, the BF16 O projection, and all
backward products and gain gradients. The three supported head geometries are
QK64/V128 paired, QK64/V64, and QK128/V128 with shifted keys. Inactive V64 auxiliary
gradients are zeroed in the original 128-wide storage. Packed QKV weights and
returned gradients use compact logical rows. The enclosing model still needs
the actual bank packing/scatter and routing.

The O products round gain-scaled BF16 weights before multiplication. QKV gain
gradients dot unscaled BF16 weight gradients with the original BF16 weights,
not dequantized FP8 values. A root task reads resident scales and gains each
execution, including zero and negative gains, so graph reuse does not capture
stale scalar values. XSA uses a 1e-8 denominator floor and joins its FP32 value
side-gradient with attention's BF16 dV before the final BF16 cast.

The post-attention arithmetic uses FP32 fused intermediates with BF16 stores at
materialized boundaries. This follows the pinned compiler's default
[`emulate_precision_casts=False`](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/config.py#L745-L760),
but exact materialization and numerical parity with the compiled pinned trainer
remain unverified. The independent references check mathematical stages on their
actual validated input boundaries. They do not substitute for a full-model
training/convergence comparison or patched-FA3 binary parity.

The initial full validation checks five geometry/gating combinations across three graph
executions with changed gains/scales. It includes zero QKV/O gains, negative O
gains, poison/replay checks, exact BF16 and E4M3 stores, auxiliary gradient padding,
and finite-difference checks on the XSA/head-gate derivatives. Dedicated XSA tests
exercise zero V and magnitudes below/above the denominator floor, zero/negative
head gates and saturated alpha. GPU primitive results retain the existing
relative-L2 2e-4 or max-absolute 2e-5 gate. Audited, production, stage-by-stage and
CUDA-Graph execution must agree bitwise.

Initial compilation failed on a mixed-type `auto` declaration, retained in
`cuda-check-20260919T044720900064Z-75aa5048`. The corrected replay smoke test is
`cuda-check-20260919T044805814117Z-d5a16b12`. Initial independent stage checks and
all four sanitizers passed in `cuda-check-20260919T045354127362Z-40a41dc7`.
Expanded graph reuse, XSA stress, full-size replay, and all four sanitizers pass
in `cuda-check-20260919T050903781403Z-43d5b34b`. Shared-dispatch regressions and all
four sanitizers pass in `cuda-check-20260919T050944335319Z-f1278f2d` (MLP),
`cuda-check-20260919T050944335321Z-78539b10` (attention/QKV),
`cuda-check-20260919T050944335318Z-3211406a` (BF16), and
`cuda-check-20260919T050944335317Z-20f13048` (ANVIL).

The first layer profiler, `profile-20260919T050929722575Z-0552ee58.tar.gz`, records
128 registers/thread, 32776 shared bytes/block, no spills in the combined worker,
24.98% achieved occupancy and only 0.17 eligible warps per scheduler. Sleep stalls
account for 13.77 of 29.81 warp cycles per issued instruction, long-scoreboard
stalls for 6.94, and barrier stalls for 3.72. SASS samples locate large non-sleep
stalls at the scalar attention loads. The attention-only dispatcher specialization
has a four-byte spill, while the combined attention/BF16/ANVIL worker used for
these initial timings does not.

The graph contains 210339–217923 tile tasks at 16384 tokens, but currently uses
fifteen whole-stage dependencies. Publishing each stage's ready tasks from one
controller thread was expensive. A cooperative candidate lets the controller
acquire the final dependency counter and reserve the queue range, then uses a
block barrier to pass visibility to all 128 threads before they publish disjoint
queue slots with release stores. Task ownership and all numerical operations
remain unchanged. This is a scheduler optimization, not a full-model result.

Both publication variants pass the full numerical/replay suite and all four
sanitizers in `cuda-check-20260919T051140680180Z-50fa263a`. On the same H100 80GB
HBM3 with CUDA 12.8.93 and 528 worker blocks, cooperative publication gives:

| Synthetic attention workload | Single-thread publication | Cooperative publication | Matching CUDA Graph |
| --- | ---: | ---: | ---: |
| Paired QK64/V128, window 128 | 77.496704 ms | 16.485744 ms | 17.672112 ms |
| QK64/V64, window 128 | 76.201057 ms | 15.173632 ms | 16.025344 ms |
| QK128/V128, window 384 | 80.487984 ms | 33.357344 ms | 36.696527 ms |

These use 16384 tokens, nineteen documents capped at 896 original tokens,
five warmups and ten retained event samples. Paired document offsets are doubled.
Timings include resident scalar setup and persistent queue reset, but prepacked
activation/weight caches, host planning and allocation are outside the timed
region. The Graph control uses the same fifteen stages and tile primitives.
All stage and gradient outputs agree bitwise at these geometries. The independent
FP64 stage oracles run on the smaller stress cases, not on the 16384-token cases.

The cooperative worker uses 128 registers, 32800 shared bytes, an eight-byte
stack frame and four bytes each of spill loads/stores. It was faster despite
that small spill. Worker counts of 132 and 264 were slower for these workloads.
Cooperative publication is promoted only for the attention-layer target. The
existing MLP, BF16, attention-core and ANVIL targets retain their prior build flags.
The initial comparison archive calls the candidate `layer_cooperative` and the
original `layer`. After promotion, `layer` selects the candidate and
`layer_serial` retains the original publication control. No kernel arithmetic or
scheduler implementation changed during that target rename.

This is a 2.4–5.0x reduction in this native component's persistent time. Its
apparent 5–9% advantage over the initial CUDA Graph is superseded by the stronger
control below. It does not establish a full-model training speedup or a leaderboard result. BF16 evaluation QKV/MLP,
normalization and residual/MUDD routing, embedding/gate construction, loss, Adam,
optimizer communication/schedules, dynamic-scale updates and the distributed
training loop still need to be connected and validated against the pinned trainer.

The promoted default and renamed original control passed again in
`cuda-check-20260919T052247126211Z-98a4da54`, on an **H100 NVL** rather than the
80GB HBM3 model in the preceding table. Persistent times were
17.723007/15.966928/36.146175 ms versus single-thread-publication times of
69.519474/67.377693/74.267216 ms. Those comparisons are within that same NVL run,
not across GPU models. Its companion H100 80GB profile is
`profile-20260919T052247126211Z-2669b656.tar.gz`.

Review found an occupancy mismatch in the initial Graph control: its generic
stage dispatcher used 153 registers, while the persistent worker used 128.
The final harness measures both the unconstrained stage kernel and a stage
kernel with the same four-block launch-bounds target as the persistent worker.
The latter compiles to 126 registers without spills and is the faster control.
Both are checked against the audited graph bitwise before timing. The final
comparison on H100 80GB HBM3 is:

| Workload | Cooperative persistent + reset | Faster native CUDA Graph | Persistent overhead |
| --- | ---: | ---: | ---: |
| Paired QK64/V128, window 128 | 16.477280 ms | 15.048384 ms | 9.50% |
| QK64/V64, window 128 | 15.173520 ms | 13.605152 ms | 11.53% |
| QK128/V128, window 384 | 33.335663 ms | 31.855248 ms | 4.65% |

Thus cooperative publication fixes a large native scheduler regression, but the
persistent attention layer **does not beat the stronger CUDA-Graph control**.
Do not use the earlier weaker control to claim a fusion win. The remaining
scalar-attention cost also precludes a claim against the pinned FA3 trainer.

The final harness, including both occupancy controls, passes numerical checks,
bitwise replay and all four sanitizers in
`cuda-check-20260919T052415996137Z-21e63829`. The 24 members of its source archive
match the final CUDA/build files, Python glue and frontier lock exactly. No
numerical or scheduler changes were made after that snapshot. Maximum relative-L2
errors in the expanded suite are 8.14953908e-7 for BF16 products,
6.37381712e-7 for gain gradients, and 1.78119669e-7 for attention gradients.
The independent finite-difference check reaches at most 7.45278173e-8 relative L2.

The final H100 80GB profile, `profile-20260919T052435618777Z-fa4b6e9e.tar.gz`,
contains PTX, SASS, the complete NCU report, metrics and instruction samples.
The combined worker has 128 registers/thread, 32800 shared bytes/block and a
four-byte spill load/store. Achieved occupancy is 25.00%, with 0.38 eligible
warps per scheduler. Of 13.99 warp cycles per issued instruction, long-scoreboard
stalls account for 7.28, sleeping for 1.23 and barriers for 0.67. This confirms
that queue publication no longer dominates in the same way. Scalar attention
memory dependencies are now the principal measured stall source. The profiler's
33.45 ms duration is diagnostic and is not substituted for the event timings.

## Native BF16 evaluation components

`--experiment=evaluation` now runs BF16 attention forward and the BF16 MLP
forward inside the same typed persistent worker used for training components.
The attention graph has six stages and no FP8 products or backward tasks.
QKV/O weights are the original BF16 values, with gains multiplied and rounded
into BF16 weights before the products. Narrow V64 evaluation uses separate QK
and compact live-row V products. Other geometries use the reference's packed
QKV layout, including paired evaluation. Q/K normalization, supplied rotary
factors, shifted keys, auxiliary values, XSA and head gates retain the training
component's math and storage contracts.

The evaluation graph accepts normalized BF16 input. FP8 and backward pointers
are null, and the tests put NaNs in the unused FP8 scale slots. Reused graphs
must still produce finite, correct results as QKV/O gains change, including zero
and negative gains. The harness allocates no FP8 caches or backward storage for
evaluation and omits diagnostic buffers in the large timed cases. It does not
build the parent model's normalization, rotary schedule, gates or parameter-bank
views. Prepared scalar inputs and any scalar-gradient casts also belong to that
parent graph, so these components do not establish compiled-trainer parity.

`cuda/mlp_evaluation.cuh` implements the reference's 768 → 2816 → 768 evaluation
MLP. The BF16 up-product first rounds its pre-activation to BF16, then computes
squared ReLU and stores BF16 before the down-product. The down weight remains
in its original `[2816, 768]` bank layout. Evaluation does not fold the post-lambda
into its weight scale: the reference applies that gain at the residual site.
That enclosing site and layer 8's parallel-MLP combination remain to be wired.
The planner releases each 64-token row's down-product tiles after that row's
44 up-product tiles finish. It does not wait for every row of the batch.

The pinned reference uses **262144 tokens per GPU** for validation. At the last
step, `TrainingManager.apply_final_ws_ext()` sets the long window to 20×128 =
**2560**, while the short window stays at 6×128 = **768**. The extension changes
the window without another Yarn update. These constants are now explicit in
`cuda/frontier.cuh` and `frontier.lock.json`. They do not change the canonical
10485760-token held-out set, full-vocabulary CE, or timing convention. Synthetic
component checks at those shapes are not a held-out validation run.

Initial BF16 attention references, graph reuse/replay and all four sanitizers
pass in `cuda-check-20260919T053721181201Z-66d43978`. Adding the MLP's BF16
activation and per-row dependencies also passes the combined numerical suite
and all four sanitizers in `cuda-check-20260919T054303831948Z-fb34d4b8`. MLP tests
include changed inputs, a partial final row tile and zero down-projection
weights, using pedantic-cuBLAS products and exact BF16 activation/store checks.
The existing numerical gates remain unchanged. Shared changes pass the training
attention layer in `cuda-check-20260919T054325910154Z-14e31d71`, FP8 MLP in
`cuda-check-20260919T054325910257Z-cb1ae2d7`, BF16 primitives in
`cuda-check-20260919T054325910128Z-be04e727`, and ANVIL in
`cuda-check-20260919T054325910324Z-6086ffee`, each with all four sanitizers.

The evaluation profile `profile-20260919T054936290928Z-6bdf07ea.tar.gz` captures
the BF16 wide-attention graph at 16384 tokens and a 384-token window on H100
80GB HBM3. The combined worker uses 128 registers, 32800 shared bytes and a
four-byte spill. Achieved occupancy is 24.98%, with 0.36 eligible warps per
scheduler. Long-scoreboard stalls account for 8.81 of 14.20 warp cycles per
issued instruction, barriers for 0.54 and sleeping for 0.13. This remains a
scalar-attention memory-dependency bottleneck, not evidence that the native
attention approaches the pinned FA3 implementation.

Auditing the parent call sites exposed a coverage error in the earlier harness:
its geometry/gating stress cases were not all actual layer configurations.
The new manifest and tests use the effective roles below, after skipped sites
are removed. Layers 0 and 5 share an attention configuration but differ in parent
routing. Layer 10 receives auxiliary values including MUDD and uses a unit extra
O multiplier. Its enclosing graph must discard the derivative of that constant.

| Layer | Paired | Auxiliary V | XSA | Head gate | Extra O multiplier |
| --- | --- | --- | --- | --- | --- |
| 0, 5 | Yes | No | No | No | Learned |
| 1 | No | Yes | Yes | No | Learned |
| 2 | Yes | Yes | No | No | Learned |
| 3 | No | No | Yes | Yes | Learned |
| 8 | No | Yes | No | No | Learned |
| 10 | No | Yes | No | Yes | Unit |

The final role-specific training suite passes all four sanitizers in
`cuda-check-20260919T055852114767Z-11d426b4`; evaluation passes all four in
`cuda-check-20260919T055852114769Z-93c5b12f`. The evaluation suite's numerical
checks and bitwise replay include the 262144-token MLP and wide attention with
64 synthetic 4096-token documents and a 2560-token window. The following event
medians use H100 80GB HBM3, five warmups and ten retained samples; persistent
timing includes queue reset, and both Graph occupancy variants are measured.
These are native component comparisons with externally prepared inputs.

| Workload | Persistent, 528 blocks | Faster native CUDA Graph |
| --- | ---: | ---: |
| Training role 2, 16384 tokens, window 128 | 16.435280 ms | 14.994864 ms |
| Training role 1, 16384 tokens, window 128 | 15.071264 ms | 13.546336 ms |
| Training role 3, 16384 tokens, window 384 | 33.515039 ms | 31.973696 ms |
| Evaluation role 2, 16384 tokens, window 128 | 2.942368 ms | 2.500096 ms |
| Evaluation role 1, 16384 tokens, window 128 | 2.668912 ms | 2.207584 ms |
| Evaluation role 3, 16384 tokens, window 384 | 9.975536 ms | 9.481952 ms |
| Evaluation MLP, 16384 tokens | 4.265872 ms | 4.912352 ms |
| Evaluation MLP, 262144 tokens | 68.820545 ms | 76.747536 ms |
| Evaluation role 3, 262144 tokens, window 2560 | 580.132996 ms | 569.041992 ms |

The evaluation MLP benefits from row-local dependencies in this native comparison;
attention remains slower than its stronger Graph control. Evaluation component
timings do not establish a speedup in the benchmark's timed training loop.

The final role-3 profile is `profile-20260919T055852114767Z-37a4fdc7.tar.gz`.
Modal supplied **H100 NVL** for this profile, unlike the H100 80GB HBM3 event
measurements above. It retains PTX, SASS and the complete NCU report. Resources
remain 128 registers, 32800 shared bytes and a four-byte spill; achieved occupancy
is 24.98%, with 0.37 eligible warps per scheduler. Long-scoreboard stalls account
for 8.58 of 13.88 warp cycles per issued instruction, barriers for 0.50 and sleeping
for 0.12. Its 10.96 ms profiled duration is diagnostic, not a timing comparison
against a run on the other GPU model.

All 26 members of each final training, evaluation and profile source archive
match the CUDA/build files, Python glue and frontier lock exactly. The source
hash check and local Python/shell syntax checks pass. This checkpoint adds
evaluation components and faithful role coverage; the full parent routing,
loss, optimizer integration and distributed training remain unfinished.

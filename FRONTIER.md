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
| MLP | Hidden width 2816, bank order 0,1,2,3,4,5,6,8,11,9,10; layer 8 has two parallel MLPs sharing its normalized input | Individual six-GEMM branch implemented; full routing absent |
| Attention | Active at layers 0,1,2,3,5,8,10; QK width 128 at 3/10 and 64 elsewhere; value width 64 at 1/8 and 128 elsewhere | Not implemented |
| Skipped layer | Layer 7 omits both sublayers but retains residual scaling/injection and saved state; layers 4/9 omit attention; layer 6 already lacks attention | Not implemented |
| MUDD and gates | Grouped post-loop mixing, learned gates, exact injection sites and saved-activation reuse | Not implemented |
| Embeddings | 84,602,880 × 768 logical learned n-gram table, split into bigram/trigram halves; eight shards, 8192 sign rows; four value-embedding planes | Not implemented |
| Matrix optimizer | ANVIL twin velocity rails, six spectral maps, per-bank grouping/annealing, terminal blends and norm restoration | Not implemented |
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

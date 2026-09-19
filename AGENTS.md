# modded-nanogpt hillclimb

The target is the main NanoGPT speedrun, using Modal GPUs and a fully fused,
fine-grained CUDA/PTX training megakernel. `train_gpt.py` is the only Python entry
point for the new implementation. Write new computation, scheduling, numerical
references and tests in CUDA/C++. Do not add Python modules or Triton kernels.
Existing upstream Python is a temporary baseline, not the intended implementation.

Read `FRONTIER.md` and `frontier.lock.json` first: the current target is PR #360 at
`c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7`, whose techniques have explicit
maintainer approval. The exact reference is in `../modded-nanogpt-frontier`.
`python train_gpt.py --check-frontier` verifies its source hashes. `run.sh`
launches this full reference, while `--cuda-check` runs only our native component. The full native
model is incomplete; matching MLP dimensions is not proof of architecture or
training parity. Preserve PR #360's optimizer, loss, scales, routing and schedule
when porting, including its 40960-token taper batch. Do not silently substitute
the older merged architecture or combine unvalidated training changes.

`MEGAKERNEL_RESEARCH.md` records historical experiments and remaining work.
The unchanged merged-record control is in `../modded-nanogpt-upstream`.
HazyResearch's reference and its pinned ThunderKittens submodule are in
`../Megakernels`.

Work on `hillclimb/main-path`. Preserve the upstream baseline until the native
training path is validated. Do not replace the actual model with a toy transformer
and call that a main-path result.

Validate on Modal with:

```sh
uv run --no-project --python 3.12 --with modal==1.2.6 train_gpt.py --cuda-check --modal --sanitize
```

Use `--main-shapes --kernel-compare --sanitize` for the optimized default versus
the original control on one GPU. Use `--profile` to capture PTX, SASS and a full
Nsight Compute report. Profile timings are diagnostic; use the separate ordinary
CUDA-event run for speed comparisons.

The default native validator now targets PR #360 MLP arithmetic. Use
`--variant=merged` to reproduce the older 3072-wide E4M3 component. Timing these
different workloads against each other is not an optimization comparison.

`--experiment=attention --sanitize` checks the mixed-operation persistent worker:
Q/K RMS normalization and rotary transforms, causal variable-length attention,
attention backward, and FP8 QKV gradient packing. `--experiment=attention
--profile` captures its current scalar attention baseline. This older component
test does not include projection/gain gradients, XSA or head gating.
The dependent GEMM in this test is a scheduler sentinel, not an output projection.
FP64 mathematical checks do not establish parity with the pinned FA3 binary.

`--experiment=bf16 --ablate --sanitize` compares the native BF16 GEMM's coalesced,
original strided and independent asynchronous loaders on the same GPU. This primitive preserves the
scaled-weight and split-product rounding needed by O projections and ANVIL,
and the connected training-attention layer uses it for O and projection gradients.

`--experiment=layer --sanitize` checks the connected training-attention layer:
cached FP8 QKV projections, QK normalization/RoPE, attention, XSA/head gates,
O projection and backward through every component and gain. Independent CUDA/C++
references check each materialized boundary; graph reuse changes device gains
and scales, including zero gains. It still accepts normalized FP8 caches and an
external output gradient. The enclosing model graph and exact pinned
compiled-trainer/FA3 parity remain unimplemented or unverified.
Its task graph uses 64x64 matrix tiles and four-token/head tasks with whole-stage
dependencies. Cooperative queue publication is the layer default. Use
`--experiment=layer --ablate --sanitize` to compare it with the single-thread
publication control on the same GPU, including four sanitizers for each.
`--experiment=layer --profile` captures the combined worker at 16384 tokens.
The layer benchmark also checks a CUDA-Graph control with the same four-block
occupancy target. Use the faster control when assessing persistent performance.

`--experiment=evaluation --sanitize` checks BF16 attention forward and the
2816-wide BF16 MLP forward. Evaluation uses original BF16 weights and normalized
BF16 activations, with no FP8 caches or backward tasks. The MLP rounds the
pre-activation before squared ReLU and applies its post-lambda at the enclosing
residual site. That site is still absent. Tests include repeated graph execution,
zero/negative gains, zero down-projection weights, and 262144-token shapes.
The final validation long window is 2560, extended from the last training stage's
1664 without another Yarn update. Synthetic component checks do not constitute
full-vocabulary validation or a full-model result. The parent model still owns
bank views, normalization, prepared scalar inputs and scalar-gradient casts.

`--experiment=anvil --ablate --sanitize` checks the rank-local optimizer update
body, including velocity state, six matrix maps, lane-energy normalization and
the split FP32 parameter shadow. It compares idle delays of 1024 and 64 ns on
one GPU. The default stays at 1024 ns because the shorter delay has no consistent
benefit. Serial and concurrent CUDA-Graph controls preserve the same matrix
dependencies. Scalar/rank-one divergence is reported separately when both the
native implementation and independent arithmetic reference reproduce it.
This is not a full optimizer/training integration or a pinned-trainer parity test.
Use `--experiment=anvil --profile` for the optimizer worker's PTX/SASS/NCU report.

Record the GPU actually supplied, compiler resources, numerical errors and raw
timings. Modal may supply H200 for an H100 request. Compare candidates on the same
device, include initialization/quantization/layout costs for promotion, and keep
component timings separate from whole-model training time. Never loosen numerical
checks merely to admit a faster candidate.

A full-model result requires forward, loss, backward, optimizer, scale updates and
distributed communication. The six-GEMM MLP and attention graphs are components;
both still accept externally supplied output gradients.

`--experiment=loss --sanitize` checks tiled training cross entropy and E5M2
logit gradients against the pinned CUDA loss source. It starts from resident raw
E4M3 logit codes and valid target/candidate positions. It preserves FP16 sigmoid
rounding, MTP and prefix corrections, and the reference's full MTP weight sum at
the last rows. Its three task types have row-group dependencies and unique
gradient-byte ownership. Tests cover full vocabulary and the actual sampled
widths 10240, 14336 and 24576, graph reuse and zero loss scale.
`--experiment=loss --profile` captures its PTX/SASS/NCU report. This correctness
baseline is slower than the pinned standalone loss; no training speedup is
claimed. Head GEMMs are connected in `--experiment=head`; candidate selection,
gather/densification and full-model training integration remain. The numerical reference is CUDA/C++, and
`train_gpt.py` still only launches the new implementation.

`--experiment=head --sanitize` checks the connected FP8 training head, starting
from BF16 hidden states and resident FP8 weight caches. Input division/packing,
transposes, raw E4M3 logit GEMM, MTP/prefix loss, E5M2 logit gradients and both
BF16 gradient products execute in the persistent graph. Backward matrix products
accumulate fresh K=32 tensor products with explicit FP32 additions. This applies
only to head backward; existing component arithmetic is unchanged. Tests check
cuBLAS products, the pinned CUDA loss, all rounding/transposes and Graph replay.
The BF16 division boundary follows the explicit source; compiled-Torch fusion
and exact scaled-cuBLAS/FA3 parity remain unverified. `--experiment=head --profile`
profiles the 256-token, full-vocabulary fixture, not an end-to-end training step.
Both Graph controls run hidden-state and weight-gradient products concurrently;
compare against the faster control. The enclosing training body, candidate
transport/densification and optimizer/distributed integration remain incomplete.

`--experiment=tail --sanitize` extends the training head backward through final
RMS normalization and post-loop grouped MUDD, including its shared GELU network,
ten independent source adjoints and all coefficient-network parameter gradients.
It consumes externally supplied layer outputs and value planes. Shared-source
gradient accumulation into the model body remains missing. Matrix tasks use
64x64 tiles and pointwise tasks use four rows. The head retains row-group loss
dependencies; the added tail stages currently use whole-stage dependencies.
The CUDA/C++ numerical and replay checks are hard gates. The pinned Torch 2.10
C++ autograd comparison is diagnostic: eager BF16 coefficient/mixing boundaries
differ from the native FP32 fusion. Do not call this compiled-trainer parity.
`--experiment=tail --profile` captures the 256-token, full-vocabulary fixture.
Ordinary runs include the ATen diagnostic. Tail sanitizer runs pass
`--native-only`: the ATen linear/GELU/autograd reference independently fails
initcheck under both CUDA 12.8 and 13.1 tools. `--experiment=tail_reference_probe
--sanitize` retains the standalone reproducer, which launches no native kernels.
Do not claim the reference sanitizer failure has been fixed or all-library
sanitizer coverage has passed.

`--experiment=suffix --sanitize` connects the final layer's 2816-wide FP8 MLP
to that tail and loss, including RMS, activation/gradient packing, both layouts,
six products, per-token residual coefficients and both MLP parameter gradients.
The final mixing stage reuses the normalized MLP input. Its adjoint and the MLP's
input adjoint are summed before RMS backward. FP32 normalized values feed E4M3
packing without an extra BF16 cast, while the reused MUDD source stays BF16.
Post-activation and dpre maxima are collected before FP8 conversion. Delayed
scale refresh and weight-cache updates remain external. The last MLP uses
per-token coefficients, so it does not use the scalar post-lambda fold required
by earlier MLPs. Head and MLP weight-gradient work can run concurrently with
their input-gradient paths in both the persistent graph and Graph controls.
The input residual, last-layer coefficients and other source states remain
external. Earlier body backward, full-model integration and compiled-Torch
parity are still missing. Sanitizers use the same explicit `--native-only`
scope as the tail. `--experiment=suffix --profile` uses 256 tokens/full vocabulary.

`--experiment=body --sanitize` runs the connected eleven-layer BF16 evaluation
body: seven attention calls, eleven MLP calls, residual/skip routing, shared
layer-8/10 normalization, parallel layer-8 MLPs and post-loop MUDD mixing. It
starts from token IDs, resident token/value tables, resolved sparse-cache rows,
learned parameters and supplied rotary factors. Token/value gathers, signed
n-gram combination, smear, initial normalization, all four coefficient networks
and the full-vocabulary BF16 evaluation head/loss now run inside the graph.
The head reduces 64-column tiles without materializing the full logit tensor;
small cases also retain logits to check rounding and the reduction separately.
Sparse-cache transport, compiled-trainer parity, full backward and training
integration remain missing. Do not describe this as a complete native trainer.
Add `--main-shapes` for the 16384-token check and native
body timings; `--experiment=body --profile` captures PTX/SASS/NCU. The fixture uses
synthetic documents and diagnostic buffers, not canonical validation. Compare
against the faster of both Graph controls before claiming a fusion speedup.
The body enables `NANO_BF16_INDEPENDENT_LOAD`: contiguous aligned operands use
asynchronous copies even when their partner is transposed. `--experiment=body
--ablate --main-shapes --sanitize` compares this default with `body_control`,
which preserves the former coupled load decision, and sanitizes both builds.
Other component defaults retain their existing load policy.
`--experiment=routing --sanitize` checks weighted mixing and RMS normalization
forward/backward, including FP64 references and normalization finite differences.
Coefficient adjoints are unrounded per-token/group partials; their consumers own
fan-in reduction and final parameter casts. `--experiment=routing --profile`
profiles these operations in the combined worker. Its component timing includes
diagnostic outputs and is not a whole-model timing.

Use `--main-shapes --gradient-chunk=1024 --ablate --sanitize` with the command
above to check ordered gradient chunks and compare operand-stage sizes, FIFO,
unsplit gradients, and 4096-token chunks on the same GPU. Preserve the exact
chunked-versus-unsplit gradient check. Audit task visits outside timing and
validate the production specialization too.

Keep training data, canonical held-out evaluation, validation target and timing
accounting intact. No validation leakage, cached answers, selective reporting,
or relaxed correctness gates to improve a benchmark number.

`--experiment=last_layer --sanitize` connects layer 10's FP8 QKV projections,
attention, head gate and BF16 O projection to its 14-coefficient MUDD network,
final FP8 MLP and existing tail/loss/backward. It returns network/projection/MLP
parameter gradients, bigram and head-gate adjoints, and the summed contributions
to cache[0], cache[7], cache[9] and layer-10 values. The shared normalized
cache[7] joins attention and tail consumers before RMS backward. Earlier uses,
including layer 8, must still contribute when the preceding body is connected.
The fixture supplies cached states, bigram values, the earlier head-gate network's
output, rotary factors, weights and scales. It is not a full-model training run.
Its 83 phases retain 64x64 matrix tiles and explicit joins for shared adjoints;
many dependencies still wait for complete stages. Native cuBLAS/FP64, rounding,
poisoned-buffer replay and task audits are hard gates. ATen remains a post-loop
tail diagnostic only; sanitizers explicitly use the existing native-only scope.
Use `--experiment=last_layer --ablate --sanitize` to compare the default
1024 ns idle delay with `last_layer_idle64` on the same GPU. Shorter sleeping and
three-block occupancy did not improve the first experiments; preserve the
four-block, 1024 ns default unless a new controlled measurement supports a change.

`--experiment=layer_nine --sanitize` extends the training section through layer
9's skipped-attention residual/bigram site and folded FP8 MLP. Layer 9 has no
x0 injection. Its post-lambda is folded into the forward down-projection and
backward dpre scales; the saved post-activation scale stays unfolded. The
unscaled BF16 dW2 is dotted with the original BF16 W2 for the post-lambda gradient,
then multiplied by the BF16 post-lambda for the final weight gradient. Never
recover that scalar gradient by dividing by the post-lambda: zero is supported.
The two residual scalar gradients are FP32 reductions followed by BF16 casts.
Layer-9 and layer-10 bigram adjoints join, while the earlier bigram consumers and
head/bigram gate-network producers remain outside this section. Use
`--experiment=layer_nine --profile` for the 256-token/full-vocabulary fixture.
Native-only sanitizer scope and the diagnostic-only ATen tail comparison remain
unchanged. Earlier body backward, scale/cache updates and optimizer/distributed
integration are still required for a complete trainer.

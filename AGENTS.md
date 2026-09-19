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
claimed. Logit/gradient GEMMs, candidate selection/gather/densification and full
model training integration remain. The numerical reference is CUDA/C++, and
`train_gpt.py` still only launches the new implementation.

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

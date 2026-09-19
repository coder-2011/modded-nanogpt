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

`--experiment=bf16 --ablate --sanitize` compares the native BF16 GEMM's coalesced
and original strided loaders on the same GPU. This primitive preserves the
scaled-weight and split-product rounding needed by O projections and ANVIL,
and the connected training-attention layer uses it for O and projection gradients.

`--experiment=layer --sanitize` checks the connected training-attention layer:
cached FP8 QKV projections, QK normalization/RoPE, attention, XSA/head gates,
O projection and backward through every component and gain. Independent CUDA/C++
references check each materialized boundary; graph reuse changes device gains
and scales, including zero gains. It still accepts normalized FP8 caches and an
external output gradient. BF16 evaluation projections, the enclosing model graph,
and exact pinned compiled-trainer/FA3 parity remain unimplemented or unverified.
Its task graph uses 64x64 matrix tiles and four-token/head tasks with whole-stage
dependencies. Cooperative queue publication is the layer default. Use
`--experiment=layer --ablate --sanitize` to compare it with the single-thread
publication control on the same GPU, including four sanitizers for each.
`--experiment=layer --profile` captures the combined worker at 16384 tokens.
The layer benchmark also checks a CUDA-Graph control with the same four-block
occupancy target. Use the faster control when assessing persistent performance.

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

Use `--main-shapes --gradient-chunk=1024 --ablate --sanitize` with the command
above to check ordered gradient chunks and compare operand-stage sizes, FIFO,
unsplit gradients, and 4096-token chunks on the same GPU. Preserve the exact
chunked-versus-unsplit gradient check. Audit task visits outside timing and
validate the production specialization too.

Keep training data, canonical held-out evaluation, validation target and timing
accounting intact. No validation leakage, cached answers, selective reporting,
or relaxed correctness gates to improve a benchmark number.

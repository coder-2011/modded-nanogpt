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
--profile` captures its current scalar attention baseline. It is not a complete
attention layer: projection/gain gradients, XSA and head gating remain unwired.
The dependent GEMM in this test is a scheduler sentinel, not an output projection.
FP64 mathematical checks do not establish parity with the pinned FA3 binary.

`--experiment=bf16 --ablate --sanitize` compares the native BF16 GEMM's coalesced
and original strided loaders on the same GPU. This primitive preserves the
scaled-weight and split-product rounding needed by O projections and ANVIL,
but the complete projection/gain and optimizer paths are not wired yet.

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

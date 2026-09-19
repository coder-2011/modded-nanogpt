# modded-nanogpt hillclimb

The target is the main NanoGPT speedrun, using Modal GPUs and a fully fused,
fine-grained CUDA/PTX training megakernel. `train_gpt.py` is the only Python entry
point for the new implementation. Write new computation, scheduling, numerical
references and tests in CUDA/C++. Do not add Python modules or Triton kernels.
Existing upstream Python is a temporary baseline, not the intended implementation.

Read `MEGAKERNEL_RESEARCH.md` for pinned references, source contracts and remaining
work. The unchanged main-path reference is in `../modded-nanogpt-upstream`.
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

Record the GPU actually supplied, compiler resources, numerical errors and raw
timings. Modal may supply H200 for an H100 request. Compare candidates on the same
device, include initialization/quantization/layout costs for promotion, and keep
component timings separate from whole-model training time. Never loosen numerical
checks merely to admit a faster candidate.

A full-model result requires forward, loss, backward, optimizer, scale updates and
distributed communication. The initial six-GEMM MLP graph is a component only; it
accepts prequantized inputs/scales and an externally supplied output gradient.

Use `--main-shapes --gradient-chunk=1024 --ablate --sanitize` with the command
above to check ordered gradient chunks and compare operand-stage sizes, FIFO,
unsplit gradients, and 4096-token chunks on the same GPU. Preserve the exact
chunked-versus-unsplit gradient check. Audit task visits outside timing and
validate the production specialization too.

Keep training data, canonical held-out evaluation, validation target and timing
accounting intact. No validation leakage, cached answers, selective reporting,
or relaxed correctness gates to improve a benchmark number.

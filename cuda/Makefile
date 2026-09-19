NVCC ?= nvcc
NVFLAGS := -std=c++17 -O3 -arch=sm_90a -lineinfo -Xptxas=-v,-warn-spills
BIN ?= validate
.DEFAULT_GOAL := validate

ALIGNED_FLAGS := -DNANO_TRANSPOSE_PAD=4 -DNANO_REUSE_QUANT=1 -DNANO_MIN_BLOCKS=4 -DNANO_ALIGNED=1
OPTFLAGS := $(ALIGNED_FLAGS) -DNANO_K_LOOP=1
validate_FLAGS := $(OPTFLAGS) -DNANO_FRONTIER=1
attention_FLAGS := $(validate_FLAGS)
bf16_FLAGS := $(validate_FLAGS) -DNANO_BF16_COALESCED=1
bf16_control_FLAGS := $(validate_FLAGS)
validate_merged_FLAGS := $(OPTFLAGS)
validate_control_FLAGS := -DNANO_FRONTIER=1
validate_pad_FLAGS := -DNANO_TRANSPOSE_PAD=4 -DNANO_FRONTIER=1
validate_direct_FLAGS := -DNANO_DIRECT_ACCUM=1
validate_resident_FLAGS := -DNANO_TRANSPOSE_PAD=4 -DNANO_MIN_BLOCKS=4 -DNANO_FRONTIER=1
validate_reuse_FLAGS := -DNANO_TRANSPOSE_PAD=4 -DNANO_REUSE_QUANT=1 -DNANO_FRONTIER=1
validate_aligned_FLAGS := $(ALIGNED_FLAGS) -DNANO_FRONTIER=1
validate_aligned_k64_FLAGS := $(ALIGNED_FLAGS) -DNANO_K_TILE=64 -DNANO_FRONTIER=1
validate_aligned_loop_FLAGS := $(OPTFLAGS) -DNANO_FRONTIER=1
validate_wgmma_FLAGS := -DNANO_WGMMA=1
validate_fifo_FLAGS := $(OPTFLAGS) -DNANO_FIFO=1 -DNANO_FRONTIER=1
validate_serial_FLAGS := $(OPTFLAGS) -DNANO_SERIAL=1 -DNANO_FRONTIER=1
validate_baseline_FLAGS := $(OPTFLAGS) -DNANO_FIFO=1 -DNANO_SERIAL=1 -DNANO_FRONTIER=1
validate_k32_FLAGS := $(OPTFLAGS) -DNANO_K_TILE=32 -DNANO_FRONTIER=1
validate_k64_FLAGS := $(OPTFLAGS) -DNANO_K_TILE=64 -DNANO_FRONTIER=1
VARIANTS := validate validate_merged validate_control validate_pad validate_direct validate_resident validate_reuse \
            validate_aligned validate_aligned_k64 validate_aligned_loop validate_wgmma validate_fifo \
            validate_serial validate_baseline validate_k32 validate_k64

$(VARIANTS): validate.cu megakernel.cuh attention.cuh qkv.cuh bf16_gemm.cuh hopper.cuh frontier.cuh Makefile
	$(NVCC) $(NVFLAGS) $($@_FLAGS) validate.cu -lcublas -o $@

quantize: quantize.cu bench.cuh megakernel.cuh attention.cuh qkv.cuh bf16_gemm.cuh hopper.cuh Makefile
	$(NVCC) $(NVFLAGS) quantize.cu -lcublas -o $@

attention: attention.cu attention.cuh qkv.cuh bf16_gemm.cuh bench.cuh megakernel.cuh hopper.cuh frontier.cuh Makefile
	$(NVCC) $(NVFLAGS) $(attention_FLAGS) attention.cu -o $@

bf16 bf16_control: bf16.cu bf16_gemm.cuh megakernel.cuh attention.cuh qkv.cuh bench.cuh hopper.cuh Makefile
	$(NVCC) $(NVFLAGS) $($@_FLAGS) bf16.cu -lcublas -o $@

transport: transport.cu bench.cuh Makefile
	$(NVCC) $(NVFLAGS) transport.cu -lnccl -o $@

.PHONY: check sanitize clean print-flags
print-flags:
	@echo $(NVFLAGS) $($(BIN)_FLAGS)

check: validate
	./validate

sanitize: $(BIN)
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)
	compute-sanitizer --tool initcheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)
	compute-sanitizer --tool racecheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)
	compute-sanitizer --tool synccheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)

clean:
	rm -f $(VARIANTS) quantize transport tokenizer attention bf16 bf16_control

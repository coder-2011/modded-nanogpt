NVCC ?= nvcc
NVFLAGS := -std=c++17 -O3 -arch=sm_90a -lineinfo -Xptxas=-v,-warn-spills
BIN ?= validate
.DEFAULT_GOAL := validate

quantize: quantize.cu bench.cuh megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) quantize.cu -lcublas -o $@

transport: transport.cu bench.cuh
	$(NVCC) $(NVFLAGS) transport.cu -lnccl -o $@

validate: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) validate.cu -lcublas -o $@

validate_wgmma: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) -DNANO_WGMMA=1 validate.cu -lcublas -o $@

validate_fifo: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) -DNANO_FIFO=1 validate.cu -lcublas -o $@

validate_serial: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) -DNANO_SERIAL=1 validate.cu -lcublas -o $@

validate_baseline: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) -DNANO_FIFO=1 -DNANO_SERIAL=1 validate.cu -lcublas -o $@

validate_k32: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) -DNANO_K_TILE=32 validate.cu -lcublas -o $@

validate_k64: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) -DNANO_K_TILE=64 validate.cu -lcublas -o $@

.PHONY: check sanitize clean
check: validate
	./validate

sanitize: $(BIN)
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)
	compute-sanitizer --tool initcheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)
	compute-sanitizer --tool racecheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)
	compute-sanitizer --tool synccheck --error-exitcode 1 ./$(BIN) --quick $(CHECK_FLAGS)

clean:
	rm -f validate validate_wgmma validate_fifo validate_serial validate_baseline validate_k32 validate_k64 quantize transport tokenizer

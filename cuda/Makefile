NVCC ?= nvcc
NVFLAGS := -std=c++17 -O3 -arch=sm_90a -lineinfo -Xptxas=-v,-warn-spills
BIN ?= validate

validate: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) validate.cu -lcublas -o $@

validate_wgmma: validate.cu megakernel.cuh hopper.cuh
	$(NVCC) $(NVFLAGS) -DNANO_WGMMA=1 validate.cu -lcublas -o $@

.PHONY: check sanitize clean
check: validate
	./validate

sanitize: $(BIN)
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(BIN) --quick
	compute-sanitizer --tool initcheck --error-exitcode 1 ./$(BIN) --quick
	compute-sanitizer --tool racecheck --error-exitcode 1 ./$(BIN) --quick
	compute-sanitizer --tool synccheck --error-exitcode 1 ./$(BIN) --quick

clean:
	rm -f validate validate_wgmma

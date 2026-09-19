#include "bench.cuh"
#include "megakernel.cuh"
#include <cmath>
#include <cublas_v2.h>
#include <random>

using namespace nano;

void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}

__global__ void unpack_bf16(BF16Matmul op, float *a, float *b) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < int64_t(op.m) * op.k)
        a[i] = float(op.a[(i / op.k) * op.ar + (i % op.k) * op.ak]);
    if (i < int64_t(op.n) * op.k)
        b[i] = float(bf16(float(op.b[(i / op.k) * op.br + (i % op.k) * op.bk]) * op.b_scale));
}

__global__ void reference_bf16_post(BF16Matmul op, float *product) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(op.m) * op.n)
        return;
    float x = op.alpha * product[i];
    if (op.round_product)
        x = float(bf16(x));
    if (op.beta != 0.0f)
        x = fmaf(op.beta, float(op.c[i]), x);
    product[i] = x;
}

__global__ void staged_bf16(BF16Matmul op) {
    __shared__ __align__(1024) bf16 scratch[bf16_scratch_bytes / sizeof(bf16)];
    if (!op.symmetric || blockIdx.x >= blockIdx.y)
        bf16_matmul_tile(op, blockIdx.x, blockIdx.y, scratch);
}

__global__ void reset_bf16_queue(int *state, int *counters, int *queue, const int *initial,
                               int count, int roots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        queue[i] = initial[i];
    if (i == 0) {
        state[0] = roots; state[1] = roots; state[2] = count; state[3] = 0;
        counters[0] = 0;
    }
}

struct Schedule {
    int roots, count;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, initial, state, audit;
    DeviceBuffer<BF16Matmul> ops;
    struct Host {
        int roots;
        std::vector<Task> tasks;
        std::vector<BF16Matmul> ops;
        explicit Host(BF16Matmul op, const BF16Matmul *consumer) : ops{op} {
            for (int r = 0; r < (op.m + 63) / 64; ++r)
                for (int c = 0; c < (op.n + 63) / 64; ++c)
                    if (!op.symmetric || r >= c)
                        tasks.push_back({0, r, c, {consumer ? 0 : -1, -1}, 0, 0, TaskKind::bf16_matmul});
            roots = int(tasks.size());
            if (consumer) {
                ops.push_back(*consumer);
                for (int r = 0; r < (consumer->m + 63) / 64; ++r)
                    for (int c = 0; c < (consumer->n + 63) / 64; ++c)
                        tasks.push_back({1, r, c, {-1, -1}, 0, 0, TaskKind::bf16_matmul});
            }
        }
    };
    explicit Schedule(const Host &host)
        : roots(host.roots), count(int(host.tasks.size())), tasks(count), groups(1),
          counters(1), queue(count), initial(count), state(4), audit(count + 2), ops(host.ops.size()) {
        tasks.put(host.tasks); ops.put(host.ops); groups.put({{roots, roots, count}});
        std::vector<int> entries(count, 0);
        for (int i = 0; i < roots; ++i)
            entries[i] = i + 1;
        std::mt19937 rng(931);
        std::shuffle(entries.begin(), entries.begin() + roots, rng);
        initial.put(entries);
    }
    void reset() {
        reset_bf16_queue<<<(count + 127) / 128, 128>>>(state.p, counters.p, queue.p, initial.p, count, roots);
    }
    Graph graph(bool checked = false) {
        return {nullptr, tasks.p, groups.p, counters.p, queue.p, state.p, count, roots,
                checked ? audit.p : nullptr, nullptr, nullptr, ops.p};
    }
};

void compare(const std::vector<float> &actual, const std::vector<float> &reference) {
    double error = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]))
            throw std::runtime_error("non-finite BF16 GEMM output");
        double e = double(actual[i]) - reference[i];
        error += e * e;
        norm += double(reference[i]) * reference[i];
        maximum = std::max(maximum, std::abs(e));
    }
    double relative = std::sqrt(error / std::max(norm, 1e-30));
    printf("BF16 GEMM rel_l2=%.9g max_abs=%.9g\n", relative, maximum);
    if (relative > 0.0002 && maximum > 0.00002)
        throw std::runtime_error("BF16 GEMM reference mismatch");
}

void check(int m, int n, int k, bool trans_a, bool trans_b, float b_scale,
           float alpha, float beta, bool round_product, bool symmetric, int workers, bool timing,
           bool inplace = false, int offset = 0) {
    printf("BF16 shape M=%d N=%d K=%d trans_a=%d trans_b=%d b_scale=%g alpha=%g beta=%g split=%d symmetric=%d workers=%d offset=%d\n",
           m, n, k, trans_a, trans_b, b_scale, alpha, beta, round_product, symmetric, workers, offset);
    DeviceBuffer<bf16> a(size_t(m) * k + offset), b(size_t(n) * k + offset), c(size_t(m) * n), y(c.n), copied(c.n), identity(size_t(n) * n);
    DeviceBuffer<float> raw(c.n), packed_a(a.n), packed_b(b.n), reference(c.n);
    std::mt19937 rng(1009 + m + n + k);
    std::normal_distribution<float> normal;
    for (auto *buffer : {&a, &b, &c}) {
        std::vector<bf16> values(buffer->n);
        for (auto &x : values)
            x = bf16(normal(rng) * 0.2f);
        buffer->put(values);
    }
    std::vector<bf16> id(identity.n, bf16(0.0f));
    for (int i = 0; i < n; ++i)
        id[i * n + i] = bf16(1.0f);
    identity.put(id);
    BF16Matmul op{a.p + offset, (symmetric ? a.p : b.p) + offset, c.p, y.p, raw.p, m, n, k,
                  trans_a ? 1 : k, trans_a ? m : 1,
                  trans_b ? 1 : k, trans_b ? n : 1, alpha, beta, b_scale, round_product, symmetric};
    if (symmetric && (m != n || trans_a != trans_b || b_scale != 1.0f))
        throw std::runtime_error("invalid symmetric test");
    if (symmetric && beta != 0.0f) {
        auto values = c.get();
        for (int r = 0; r < m; ++r)
            for (int col = 0; col < r; ++col)
                values[col * n + r] = values[r * n + col];
        c.put(values);
    }
    BF16Matmul consumer{y.p, identity.p, nullptr, copied.p, nullptr, m, n, n, n, 1, n, 1};
    if (inplace)
        op.c = y.p;
    Schedule schedule(Schedule::Host(op, &consumer));
    auto poison = [&] {
        CHECK_CUDA(cudaMemset(y.p, 0xff, y.n * sizeof(bf16)));
        CHECK_CUDA(cudaMemset(copied.p, 0xff, copied.n * sizeof(bf16)));
        CHECK_CUDA(cudaMemset(raw.p, 0xff, raw.n * sizeof(float)));
        if (inplace)
            CHECK_CUDA(cudaMemcpy(y.p, c.p, y.n * sizeof(bf16), cudaMemcpyDeviceToDevice));
    };
    poison();
    CHECK_CUDA(cudaMemset(schedule.audit.p, 0, schedule.audit.n * sizeof(int)));
    schedule.reset();
    megakernel<true, false, false, true><<<workers, threads>>>(schedule.graph(true));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    auto visits = schedule.audit.get();
    for (int i = 0; i < schedule.count; ++i)
        if (visits[i] != 1)
            throw std::runtime_error("BF16 task visits mismatch");
    if (schedule.state.get()[2] != 0)
        throw std::runtime_error("unfinished BF16 tasks");
    auto actual = raw.get();
    auto output = y.get(), dependent = copied.get();
    for (size_t i = 0; i < y.n; ++i)
        if (float(output[i]) != float(bf16(actual[i])) || float(dependent[i]) != float(output[i]))
            throw std::runtime_error("BF16 rounding or dependent GEMM mismatch");
    if (symmetric)
        for (int r = 0; r < m; ++r)
            for (int col = 0; col < n; ++col)
                if (actual[r * n + col] != actual[col * n + r])
                    throw std::runtime_error("Gram symmetry mismatch");

    unpack_bf16<<<(std::max(a.n, b.n) + 255) / 256, 256>>>(op, packed_a.p, packed_b.p);
    cublasHandle_t handle;
    blas_check(cublasCreate(&handle));
    blas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    float one = 1, zero = 0;
    blas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one,
                          packed_b.p, k, packed_a.p, k, &zero, reference.p, n));
    BF16Matmul ref_op = op;
    ref_op.c = c.p;
    reference_bf16_post<<<(c.n + 255) / 256, 256>>>(ref_op, reference.p);
    CHECK_CUDA(cudaGetLastError());
    compare(actual, reference.get());
    blas_check(cublasDestroy(handle));
    for (int repeat = 0; repeat < 2; ++repeat) {
        poison(); schedule.reset();
        megakernel<false, false, false, true><<<workers, threads>>>(schedule.graph());
        CHECK_CUDA(cudaGetLastError());
        if (raw.get() != actual)
            throw std::runtime_error("BF16 production/reuse mismatch");
    }
    poison(); schedule.reset();
    megakernel<false, false, true, true><<<workers, threads>>>(schedule.graph());
    CHECK_CUDA(cudaGetLastError());
    if (raw.get() != actual)
        throw std::runtime_error("combined attention/BF16 worker mismatch");
    poison();
    staged_bf16<<<dim3((m + 63) / 64, (n + 63) / 64), threads>>>(op);
    CHECK_CUDA(cudaGetLastError());
    if (raw.get() != actual)
        throw std::runtime_error("BF16 staged/persistent mismatch");

    if (timing) {
        op.raw = nullptr;
        Schedule timed(Schedule::Host(op, nullptr));
        device_time("bf16_staged", [&] {
            staged_bf16<<<dim3((m + 63) / 64, (n + 63) / 64), threads>>>(op);
        });
        device_time("bf16_persistent_with_reset", [&] {
            timed.reset();
            megakernel<false, false, false, true><<<workers, threads>>>(timed.graph());
        });
        CHECK_CUDA(cudaGetLastError());
    }
}

int main(int argc, char **argv) {
    try {
        bool quick = argc > 1 && std::string(argv[1]) == "--quick";
        device_info();
        cudaDeviceProp p;
        CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
        int resident;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, megakernel<false, false, false, true>, threads, 0));
        printf("BF16 resident_ctas_per_sm=%d\n", resident);
        check(1, 1, 1, false, false, 0.7f, 1, 0, true, false, 1, false);
        check(7, 17, 33, false, true, 1, 0.9f, 0.3f, false, false, 7, false, true);
        for (int layout = 0; layout < 4; ++layout)
            check(7, 17, 33, layout & 1, layout & 2, -0.37f, 0.9f, 0.3f, layout & 1, false,
                  layout == 0 ? 1 : 7, false);
        check(65, 65, 71, false, false, 1, 3.9052346f, -6.0950269f, false, true, 7, false);
        check(64, 64, 128, true, true, 1, 1, 0, false, true, 1, false);
        check(129, 96, 72, false, false, 0.625f, 1, 0, false, false, 7, false);
        for (int offset : {0, 1}) {
            check(65, 80, 72, false, true, 0.625f, 1, 0, false, false, 7, false, false, offset);
            check(65, 80, 72, true, false, 0.625f, 1, 0, false, false, 7, false, false, offset);
        }
        if (!quick) {
            check(257, 129, 273, true, true, 1, 1, 0, true, false, p.multiProcessorCount * 8, false);
            check(16384, 768, 768, false, false, 0.875f, 1, 0, false, false, p.multiProcessorCount * resident, true);
            check(16384, 768, 384, false, false, 0.875f, 1, 0, false, false, p.multiProcessorCount * resident, true);
            check(768, 768, 2816, true, true, 1, 1, 0, false, true, p.multiProcessorCount * resident, true);
            check(2816, 768, 768, false, true, 1, 1, 3.923798f, true, false, p.multiProcessorCount * resident, true);
        }
        puts("PASS: native BF16 GEMM, scaled weights, split epilogues, symmetric Gram and dependent scheduling");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

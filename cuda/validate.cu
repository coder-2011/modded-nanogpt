#include "megakernel.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nano;

#define CUDA(call)                                                                                 \
    do {                                                                                           \
        auto e = (call);                                                                           \
        if (e != cudaSuccess)                                                                      \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e));           \
    } while (0)
#define BLAS(call)                                                                                 \
    do {                                                                                           \
        auto e = (call);                                                                           \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
            throw std::runtime_error(std::string(#call) + ": " + std::to_string(e));               \
    } while (0)

template <class T> struct Buffer {
    T *p = nullptr;
    size_t size;
    explicit Buffer(size_t n) : size(n) { CUDA(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void put(const std::vector<T> &values) {
        if (values.size() != size)
            throw std::runtime_error("buffer size mismatch");
        CUDA(cudaMemcpy(p, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> values(size);
        CUDA(cudaMemcpy(values.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

struct Schedule {
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<int> roots;
    int range(int op, int row, int columns, int g0, int g1) {
        int begin = int(tasks.size());
        for (int c = 0; c < columns; ++c)
            tasks.push_back({op, row, c, {g0, g1}});
        return begin;
    }
    Schedule(int m, int c, int h) {
        const int mr = (m + tile - 1) / tile, cr = (c + tile - 1) / tile;
        const int hr = (h + tile - 1) / tile;
        groups.resize(2 * mr + 2);
        const int all_up = 2 * mr, all_dpre = all_up + 1;
        for (int r = 0; r < mr; ++r) {
            int start = range(0, r, hr, 2 * r, all_up);
            for (int i = start; i < int(tasks.size()); ++i)
                roots.push_back(i);
            int begin = range(1, r, cr, -1, -1);
            range(2, r, hr, 2 * r + 1, all_dpre);
            groups[2 * r] = {hr, begin, int(tasks.size())};
            begin = range(3, r, cr, -1, -1);
            groups[2 * r + 1] = {hr, begin, int(tasks.size())};
        }
        int begin = int(tasks.size());
        for (int r = 0; r < hr; ++r)
            range(4, r, cr, -1, -1);
        groups[all_dpre] = {mr * hr, begin, int(tasks.size())};
        begin = int(tasks.size());
        for (int r = 0; r < hr; ++r)
            range(5, r, cr, -1, -1);
        groups[all_up] = {mr * hr, begin, int(tasks.size())};
    }
};

struct DeviceSchedule {
    Buffer<Task> tasks;
    Buffer<Group> groups;
    Buffer<int> counters, queue, state;
    Buffer<Matmul> ops;
    std::vector<int> initial_queue;
    int root_count;
    DeviceSchedule(const Schedule &schedule, const std::vector<Matmul> &descriptors)
        : tasks(schedule.tasks.size()), groups(schedule.groups.size()), counters(groups.size),
          queue(tasks.size), state(3), ops(descriptors.size()), initial_queue(tasks.size),
          root_count(int(schedule.roots.size())) {
        tasks.put(schedule.tasks);
        groups.put(schedule.groups);
        ops.put(descriptors);
        for (int i = 0; i < root_count; ++i)
            initial_queue[i] = schedule.roots[i] + 1;
    }
    void reset() {
        CUDA(cudaMemset(counters.p, 0, counters.size * sizeof(int)));
        queue.put(initial_queue);
        state.put({0, root_count, int(tasks.size)});
    }
    Graph graph() {
        return {ops.p, tasks.p, groups.p, counters.p, queue.p, state.p, int(tasks.size)};
    }
};

__global__ void unpack(const fp8 *x, float *y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        y[i] = float(x[i]);
}

__global__ void reference_epilogue(const float *x, Matmul op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.m * op.n)
        return;
    float value = x[i] * op.scale;
    if (op.raw)
        op.raw[i] = value;
    if (op.epilogue == Epilogue::relu_square) {
        value = fmaxf(float(bf16(value)), 0.0f);
        value = float(bf16(value * value));
    } else if (op.epilogue == Epilogue::relu_backward) {
        float relu = float(bf16(sqrtf(float(op.post[i]) * op.post_scale)));
        value = float(bf16(2.0f * float(bf16(value)) * relu));
    }
    if (op.output)
        op.output[i] = bf16(value);
    if (op.quantized)
        op.quantized[i] = fp8(value / op.output_scale);
    if (op.quantized_t)
        op.quantized_t[(i % op.n) * op.m + i / op.n] = fp8(value / op.output_scale);
}

void reference(cublasHandle_t handle, const Matmul &op) {
    Buffer<float> a(size_t(op.m) * op.k), b(size_t(op.n) * op.k), out(size_t(op.m) * op.n);
    unpack<<<(a.size + 255) / 256, 256>>>(op.a, a.p, int(a.size));
    unpack<<<(b.size + 255) / 256, 256>>>(op.b, b.p, int(b.size));
    const float alpha = 1, beta = 0;
    const auto ta = op.ak == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto tb = op.bk == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
    BLAS(cublasSgemm(handle, tb, ta, op.n, op.m, op.k, &alpha, b.p, op.bk == 1 ? op.br : op.bk, a.p,
                     op.ak == 1 ? op.ar : op.ak, &beta, out.p, op.n));
    reference_epilogue<<<(out.size + 255) / 256, 256>>>(out.p, op);
    CUDA(cudaGetLastError());
}

template <class T>
void compare(const char *name, const Buffer<T> &actual, const Buffer<T> &expected,
             float relative_limit) {
    const auto a = actual.get(), e = expected.get();
    double square_error = 0, square_reference = 0;
    float max_abs = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float av = float(a[i]), ev = float(e[i]);
        if (!std::isfinite(av) || !std::isfinite(ev))
            throw std::runtime_error("nonfinite output");
        float error = std::abs(av - ev);
        square_error += double(error) * error;
        square_reference += double(ev) * ev;
        max_abs = std::max(max_abs, error);
        mismatches += av != ev;
    }
    const double relative = std::sqrt(square_error / std::max(square_reference, 1e-30));
    printf("  %-6s rel_l2=%.8g max_abs=%.8g different=%zu/%zu\n", name, relative, max_abs,
           mismatches, a.size());
    if (relative > relative_limit)
        throw std::runtime_error(std::string(name) + " numerical mismatch");
}

__global__ void staged_kernel(Matmul op) {
    __shared__ __align__(1024) fp8 scratch[8192];
    Task task{0, int(blockIdx.y), int(blockIdx.x), {-1, -1}};
    execute_tile(op, task, scratch);
}

void validate(int m, int c, int h, int workers, bool timing, unsigned seed = 42,
              bool zeros = false) {
    printf("shape M=%d C=%d H=%d workers=%d seed=%u zeros=%d\n", m, c, h, workers, seed, zeros);
    Buffer<fp8> x(size_t(m) * c), w1(size_t(h) * c), w2(size_t(h) * c), dy(size_t(m) * c);
    Buffer<fp8> xt(x.size), w1t(w1.size), w2t(w2.size), dyt(dy.size);
    Buffer<fp8> post(size_t(m) * h), dpre(size_t(m) * h), rpost(post.size), rdpre(dpre.size);
    Buffer<fp8> postt(post.size), dpret(dpre.size), rpostt(post.size), rdpret(dpre.size);
    Buffer<float> raw(m < 256 ? post.size : 1), rraw(raw.size);
    Buffer<bf16> y(x.size), dx(x.size), dw1(w1.size), dw2(w2.size);
    Buffer<bf16> ry(y.size), rdx(dx.size), rdw1(dw1.size), rdw2(dw2.size);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0, 1);
    for (auto *buffer : {&x, &w1, &w2, &dy}) {
        std::vector<fp8> input(buffer->size);
        for (auto &v : input)
            v = fp8(zeros ? 0.0f : normal(rng));
        buffer->put(input);
    }
    auto transpose = [](const Buffer<fp8> &src, Buffer<fp8> &dst, int rows, int cols) {
        auto data = src.get();
        std::vector<fp8> transposed(data.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                transposed[c * rows + r] = data[r * cols + c];
        dst.put(transposed);
    };
    transpose(x, xt, m, c);
    transpose(dy, dyt, m, c);
    transpose(w1, w1t, h, c);
    transpose(w2, w2t, h, c);
    // Deliberately non-unit scales exercise the FP8 dequantization boundaries.
    const float xs = 0.5f, ws = 0.03125f, gs = 0.25f, ps = 0.03125f, ds = 0.03125f;
    std::vector<Matmul> ops = {{x.p, w1.p, nullptr, post.p, nullptr, m, h, c, c, 1, c, 1, xs * ws,
                                ps, 0, Epilogue::relu_square},
                               {post.p, w2t.p, nullptr, nullptr, y.p, m, c, h, h, 1, h, 1, ps * ws,
                                0, 0, Epilogue::linear},
                               {dy.p, w2.p, post.p, dpre.p, nullptr, m, h, c, c, 1, c, 1, gs * ws,
                                ds, ps, Epilogue::relu_backward},
                               {dpre.p, w1t.p, nullptr, nullptr, dx.p, m, c, h, h, 1, h, 1, ds * ws,
                                0, 0, Epilogue::linear},
                               {dpret.p, xt.p, nullptr, nullptr, dw1.p, h, c, m, m, 1, m, 1,
                                ds * xs, 0, 0, Epilogue::linear},
                               {postt.p, dyt.p, nullptr, nullptr, dw2.p, h, c, m, m, 1, m, 1,
                                ps * gs, 0, 0, Epilogue::linear}};
    ops[0].quantized_t = postt.p;
    ops[2].quantized_t = dpret.p;
    if (m < 256)
        ops[0].raw = raw.p;
    auto ref = ops;
    ref[0].quantized = rpost.p;
    ref[0].quantized_t = rpostt.p;
    if (m < 256)
        ref[0].raw = rraw.p;
    ref[1].a = rpost.p;
    ref[1].output = ry.p;
    ref[2].post = rpost.p;
    ref[2].quantized = rdpre.p;
    ref[2].quantized_t = rdpret.p;
    ref[3].a = rdpre.p;
    ref[3].output = rdx.p;
    ref[4].a = rdpret.p;
    ref[4].output = rdw1.p;
    ref[5].a = rpostt.p;
    ref[5].output = rdw2.p;
    cublasHandle_t handle;
    BLAS(cublasCreate(&handle));
    BLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    for (const auto &op : ref)
        reference(handle, op);
    BLAS(cublasDestroy(handle));
    Schedule schedule(m, c, h);
    DeviceSchedule device(schedule, ops);
    for (auto *buffer : {&post, &dpre, &postt, &dpret})
        CUDA(cudaMemset(buffer->p, 0xff, buffer->size * sizeof(fp8)));
    for (auto *buffer : {&y, &dx, &dw1, &dw2})
        CUDA(cudaMemset(buffer->p, 0xff, buffer->size * sizeof(bf16)));
    device.reset();
    megakernel<<<workers, threads>>>(device.graph());
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    auto state = device.state.get();
    if (state[0] != int(schedule.tasks.size()) + workers ||
        state[1] != int(schedule.tasks.size()) || state[2] != 0)
        throw std::runtime_error("scheduler did not execute every task exactly once");
    auto counters = device.counters.get();
    for (size_t i = 0; i < counters.size(); ++i)
        if (counters[i] != schedule.groups[i].expected)
            throw std::runtime_error("bad dependency count");
    if (m < 256) {
        compare("raw", raw, rraw, 0.0002f);
        auto av = raw.get(), rv = rraw.get();
        int count = 0;
        for (size_t i = 0; i < av.size() && count < 5; ++i)
            if (av[i] != rv[i]) {
                printf("raw diff [%zu] got=%.10g reference=%.10g\n", i, av[i], rv[i]);
                ++count;
            }
    }
    compare("post", post, rpost, 0.0002f);
    compare("dpre", dpre, rdpre, 0.0002f);
    compare("post.T", postt, rpostt, 0.0002f);
    compare("dpre.T", dpret, rdpret, 0.0002f);
    compare("y", y, ry, 0.0002f);
    compare("dx", dx, rdx, 0.0002f);
    compare("dw1", dw1, rdw1, 0.0002f);
    compare("dw2", dw2, rdw2, 0.0002f);
    if (!timing)
        return;
    cudaEvent_t begin, end;
    CUDA(cudaEventCreate(&begin));
    CUDA(cudaEventCreate(&end));
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (auto op : ops)
        staged_kernel<<<dim3((op.n + tile - 1) / tile, (op.m + tile - 1) / tile), threads, 0,
                        stream>>>(op);
    CUDA(cudaStreamEndCapture(stream, &graph));
    CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    std::vector<float> fused_times, staged_times, graph_times, reset_times;
    for (int i = 0; i < 15; ++i) {
        device.reset();
        CUDA(cudaEventRecord(begin, stream));
        megakernel<<<workers, threads, 0, stream>>>(device.graph());
        CUDA(cudaEventRecord(end, stream));
        CUDA(cudaEventSynchronize(end));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, begin, end));
        if (i >= 5)
            fused_times.push_back(ms);
        CUDA(cudaEventRecord(begin, stream));
        for (auto op : ops)
            staged_kernel<<<dim3((op.n + tile - 1) / tile, (op.m + tile - 1) / tile), threads, 0,
                            stream>>>(op);
        CUDA(cudaEventRecord(end, stream));
        CUDA(cudaEventSynchronize(end));
        CUDA(cudaEventElapsedTime(&ms, begin, end));
        if (i >= 5)
            staged_times.push_back(ms);
        CUDA(cudaEventRecord(begin, stream));
        CUDA(cudaGraphLaunch(graph_exec, stream));
        CUDA(cudaEventRecord(end, stream));
        CUDA(cudaEventSynchronize(end));
        CUDA(cudaEventElapsedTime(&ms, begin, end));
        if (i >= 5)
            graph_times.push_back(ms);
        CUDA(cudaEventRecord(begin, stream));
        device.reset();
        megakernel<<<workers, threads, 0, stream>>>(device.graph());
        CUDA(cudaEventRecord(end, stream));
        CUDA(cudaEventSynchronize(end));
        CUDA(cudaEventElapsedTime(&ms, begin, end));
        if (i >= 5)
            reset_times.push_back(ms);
    }
    auto median = [](std::vector<float> values) {
        std::sort(values.begin(), values.end());
        return (values[4] + values[5]) * 0.5f;
    };
    printf("timing median_ms fused=%.6f staged_same_tiles=%.6f graph_same_tiles=%.6f "
           "fused_with_reset=%.6f tasks=%zu (warm cache)\n",
           median(fused_times), median(staged_times), median(graph_times), median(reset_times),
           schedule.tasks.size());
    for (const auto *samples : {&fused_times, &staged_times, &graph_times, &reset_times}) {
        printf("samples_ms:");
        for (float value : *samples)
            printf(" %.6f", value);
        puts("");
    }
    CUDA(cudaGraphExecDestroy(graph_exec));
    CUDA(cudaGraphDestroy(graph));
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaEventDestroy(begin));
    CUDA(cudaEventDestroy(end));
}

int main(int argc, char **argv) {
    try {
        cudaDeviceProp prop;
        CUDA(cudaGetDeviceProperties(&prop, 0));
        if (prop.major != 9)
            throw std::runtime_error("this build targets Hopper sm_90a");
        printf("GPU=%s SMs=%d CUDA_runtime=%d\n", prop.name, prop.multiProcessorCount,
               CUDART_VERSION);
        cudaFuncAttributes attr;
        CUDA(cudaFuncGetAttributes(&attr, megakernel));
        int resident_blocks;
        CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks, megakernel, threads,
                                                           0));
        printf("megakernel registers=%d static_shared=%zu local_bytes=%zu\n", attr.numRegs,
               attr.sharedSizeBytes, attr.localSizeBytes);
        printf("resident_blocks_per_SM=%d\n", resident_blocks);
        const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
        const bool main_shapes = argc > 1 && std::string(argv[1]) == "--main-shapes";
        for (int workers :
             {1, 7, prop.multiProcessorCount, (resident_blocks + 1) * prop.multiProcessorCount}) {
            validate(7, 17, 33, workers, false);
            validate(129, 96, 160, workers, false, 1337);
            validate(65, 32, 64, workers, false, 2, true);
        }
        if (main_shapes) {
            for (int m : {16384, 32768, 49152})
                validate(m, 768, 3072, resident_blocks * prop.multiProcessorCount, true);
        } else if (!quick) {
            for (int blocks_per_sm : {1, 2, 4, resident_blocks}) {
                validate(256, 768, 3072, blocks_per_sm * prop.multiProcessorCount, true);
                validate(8192, 768, 3072, blocks_per_sm * prop.multiProcessorCount, true);
            }
        }
        puts("PASS: FP8 six-GEMM MLP graph, all outputs and gradients; not full-model training.");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

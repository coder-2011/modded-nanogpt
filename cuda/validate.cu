#include "megakernel.cuh"
#include "frontier.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
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
    Schedule(int m, int c, int h, int chunk) {
        const int mr = (m + tile - 1) / tile, cr = (c + tile - 1) / tile;
        const int hr = (h + tile - 1) / tile;
        const int chunks = chunk ? (m + chunk - 1) / chunk : 0;
        groups.resize(2 * mr + (chunk ? 2 * chunks * hr : 2));
        const int all_up = 2 * mr, all_dpre = all_up + 1;
        auto gradient_group = [=](int op, int part, int row) {
            return 2 * mr + ((op - 4) * chunks + part) * hr + row;
        };
        for (int r = 0; r < mr; ++r) {
            int start = range(0, r, hr, 2 * r, chunk ? -1 : all_up);
            for (int i = start; i < int(tasks.size()); ++i) {
                roots.push_back(i);
                if (chunk)
                    tasks[i].signal[1] = gradient_group(5, r * tile / chunk, i - start);
            }
            int begin = range(1, r, cr, -1, -1);
            start = range(2, r, hr, 2 * r + 1, chunk ? -1 : all_dpre);
            if (chunk)
                for (int i = start; i < int(tasks.size()); ++i)
                    tasks[i].signal[1] = gradient_group(4, r * tile / chunk, i - start);
            groups[2 * r] = {hr, begin, int(tasks.size())};
            begin = range(3, r, cr, -1, -1);
            groups[2 * r + 1] = {hr, begin, int(tasks.size())};
        }
        if (chunk) {
            for (int op : {4, 5})
                for (int part = 0; part < chunks; ++part)
                    for (int r = 0; r < hr; ++r) {
                        int next = part + 1 < chunks ? gradient_group(op, part + 1, r) : -1;
                        int begin = range(op, r, cr, next, -1);
                        const int k_begin = part * chunk, k_end = std::min(k_begin + chunk, m);
                        for (int i = begin; i < int(tasks.size()); ++i) {
                            tasks[i].k_begin = k_begin;
                            tasks[i].k_end = k_end;
                        }
                        // Input producers and the preceding accumulator owners must all finish.
                        groups[gradient_group(op, part, r)] = {(k_end - k_begin + tile - 1) / tile +
                                                                   (part ? cr : 0),
                                                               begin, int(tasks.size())};
                    }
            return;
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
          queue(tasks.size), state(4), ops(descriptors.size()), initial_queue(tasks.size),
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
#ifdef NANO_FIFO
        state.put({0, root_count, int(tasks.size), 0});
#else
        state.put({root_count, root_count, int(tasks.size), 0});
#endif
    }
    Graph graph() {
        return {ops.p,   tasks.p, groups.p,        counters.p,
                queue.p, state.p, int(tasks.size), root_count};
    }
};

__global__ void unpack(const fp8 *x, float *y, int size, bool e5) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        y[i] = decode_fp8(x[i], e5);
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
        float relu = sqrtf(float(op.post[i]) * op.post_scale);
        value = op.quantized_e5 ? float(bf16(2.0f * value * relu))
                                : float(bf16(2.0f * float(bf16(value)) * float(bf16(relu))));
    }
    if (op.output)
        op.output[i] = bf16(value);
    if (op.quantized)
        op.quantized[i] = encode_fp8(value / op.output_scale, op.quantized_e5);
    if (op.quantized_t)
        op.quantized_t[(i % op.n) * op.m + i / op.n] = encode_fp8(value / op.output_scale, op.quantized_e5);
}

void reference(cublasHandle_t handle, const Matmul &op) {
    Buffer<float> a(size_t(op.m) * op.k), b(size_t(op.n) * op.k), out(size_t(op.m) * op.n);
    unpack<<<(a.size + 255) / 256, 256>>>(op.a, a.p, int(a.size), op.a_e5);
    unpack<<<(b.size + 255) / 256, 256>>>(op.b, b.p, int(b.size), op.b_e5);
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
             float relative_limit, bool e5 = false) {
    const auto a = actual.get(), e = expected.get();
    double square_error = 0, square_reference = 0;
    float max_abs = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float av, ev;
        if constexpr (std::is_same_v<T, fp8>) {
            av = decode_fp8(a[i], e5);
            ev = decode_fp8(e[i], e5);
        } else {
            av = float(a[i]);
            ev = float(e[i]);
        }
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

template <bool Full = false> __global__ void staged_kernel(Matmul op) {
    __shared__ __align__(1024) fp8 scratch[scratch_bytes];
    Task task{0, int(blockIdx.y), int(blockIdx.x), {-1, -1}};
    execute_tile<Full>(op, task, scratch);
}

bool aligned(const Matmul &op) {
    return op.m % tile == 0 && op.n % tile == 0 && op.k % k_tile == 0 && op.ak == 1 && op.bk == 1 &&
           op.ar % 16 == 0 && op.br % 16 == 0;
}

template <bool Audited = false>
void launch_megakernel(Graph graph, int workers, bool full, cudaStream_t stream = nullptr) {
#ifdef NANO_ALIGNED
    if (full) {
        megakernel<Audited, true><<<workers, threads, 0, stream>>>(graph);
        return;
    }
#endif
    megakernel<Audited><<<workers, threads, 0, stream>>>(graph);
}

void launch_staged(Matmul op, cudaStream_t stream = nullptr) {
    dim3 grid((op.n + tile - 1) / tile, (op.m + tile - 1) / tile);
#ifdef NANO_ALIGNED
    if (aligned(op)) {
        staged_kernel<true><<<grid, threads, 0, stream>>>(op);
        return;
    }
#endif
    staged_kernel<><<<grid, threads, 0, stream>>>(op);
}

void validate(int m, int c, int h, int workers, bool timing, unsigned seed = 42, bool zeros = false,
              int gradient_chunk = 0, bool profile = false) {
    const int chunk = gradient_chunk ? (m < 256 ? 64 : gradient_chunk) : 0;
    printf("shape M=%d C=%d H=%d workers=%d seed=%u zeros=%d gradient_chunk=%d\n", m, c, h, workers,
           seed, zeros, chunk);
    Buffer<fp8> x(size_t(m) * c), w1(size_t(h) * c), w2(size_t(h) * c), dy(size_t(m) * c);
    Buffer<fp8> xt(x.size), w1t(w1.size), w2t(w2.size), dyt(dy.size);
    Buffer<fp8> post(size_t(m) * h), dpre(size_t(m) * h), rpost(post.size), rdpre(dpre.size);
    Buffer<fp8> postt(post.size), dpret(dpre.size), rpostt(post.size), rdpret(dpre.size);
    Buffer<float> raw(m < 256 ? post.size : 1), rraw(raw.size);
    Buffer<bf16> y(x.size), dx(x.size), dw1(w1.size), dw2(w2.size);
    Buffer<bf16> ry(y.size), rdx(dx.size), rdw1(dw1.size), rdw2(dw2.size);
    Buffer<float> accum1(chunk ? dw1.size : 1), accum2(chunk ? dw2.size : 1);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0, 1);
    for (auto *buffer : {&x, &w1, &w2, &dy}) {
        std::vector<fp8> input(buffer->size);
#ifdef NANO_FRONTIER
        const bool e5 = buffer == &dy;
#else
        const bool e5 = false;
#endif
        for (auto &v : input)
            v = encode_fp8(zeros ? 0.0f : normal(rng), e5);
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
#ifdef NANO_FRONTIER
    ops[2].a_e5 = ops[2].quantized_e5 = true;
    ops[3].a_e5 = ops[4].a_e5 = ops[5].b_e5 = true;
#endif
    ops[0].quantized_t = postt.p;
    ops[2].quantized_t = dpret.p;
    if (chunk) {
        ops[4].accumulation = accum1.p;
        ops[5].accumulation = accum2.p;
    }
    if (m < 256)
        ops[0].raw = raw.p;
    const bool full = chunk % k_tile == 0 && std::all_of(ops.begin(), ops.end(), aligned);
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
    Schedule schedule(m, c, h, chunk);
    if (m < 256)
        std::shuffle(schedule.roots.begin(), schedule.roots.end(), rng);
    DeviceSchedule device(schedule, ops);
    Buffer<int> audit(schedule.tasks.size() + 2);
    CUDA(cudaMemset(audit.p, 0, audit.size * sizeof(int)));
    auto poison = [&] {
        for (auto *buffer : {&post, &dpre, &postt, &dpret})
            CUDA(cudaMemset(buffer->p, 0xff, buffer->size * sizeof(fp8)));
        for (auto *buffer : {&y, &dx, &dw1, &dw2})
            CUDA(cudaMemset(buffer->p, 0xff, buffer->size * sizeof(bf16)));
        for (auto *buffer : {&raw, &accum1, &accum2})
            CUDA(cudaMemset(buffer->p, 0xff, buffer->size * sizeof(float)));
    };
    poison();
    device.reset();
    auto checked_graph = device.graph();
    checked_graph.audit = audit.p;
    launch_megakernel<true>(checked_graph, workers, full);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    auto visits = audit.get();
    for (size_t i = 0; i < schedule.tasks.size(); ++i)
        if (visits[i] != 1)
            throw std::runtime_error("task visit count is not exactly one");
    if (visits[schedule.tasks.size()] != int(schedule.roots.size()))
        throw std::runtime_error("bad completed root count");
    printf("audit: every task executed once; gradient_tasks_before_all_roots=%d\n", visits.back());
    auto state = device.state.get();
    const int expected_head = int(schedule.tasks.size()) + workers;
#ifndef NANO_FIFO
    if (state[3] < int(schedule.roots.size()) || state[3] >= int(schedule.roots.size()) + workers)
        throw std::runtime_error("bad root ticket count");
#endif
    if (state[0] != expected_head || state[1] != int(schedule.tasks.size()) || state[2] != 0)
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
    auto compare_outputs = [&] {
        if (m < 256)
            compare("raw", raw, rraw, 0.0002f);
        compare("post", post, rpost, 0.0002f);
        compare("dpre", dpre, rdpre, 0.0002f, ops[2].quantized_e5);
        compare("post.T", postt, rpostt, 0.0002f);
        compare("dpre.T", dpret, rdpret, 0.0002f, ops[2].quantized_e5);
        compare("y", y, ry, 0.0002f);
        compare("dx", dx, rdx, 0.0002f);
        compare("dw1", dw1, rdw1, 0.0002f);
        compare("dw2", dw2, rdw2, 0.0002f);
    };
    compare_outputs();
    if (chunk) {
        Buffer<bf16> saved1(dw1.size), saved2(dw2.size);
        CUDA(cudaMemcpy(saved1.p, dw1.p, dw1.size * sizeof(bf16), cudaMemcpyDeviceToDevice));
        CUDA(cudaMemcpy(saved2.p, dw2.p, dw2.size * sizeof(bf16), cudaMemcpyDeviceToDevice));
        for (int op : {4, 5})
            launch_staged(ops[op]);
        CUDA(cudaGetLastError());
        compare("dw1.chunk", saved1, dw1, 0.0f);
        compare("dw2.chunk", saved2, dw2, 0.0f);
    }
    // The measured specialization omits audit atomics; validate it independently after reuse.
    poison();
    device.reset();
    launch_megakernel(device.graph(), workers, full);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    puts("production specialization after reset:");
    compare_outputs();
    if (profile) {
        for (int i = 0; i < 5; ++i) {
            device.reset();
            launch_megakernel(device.graph(), workers, full);
        }
        CUDA(cudaDeviceSynchronize());
        device.reset();
        CUDA(cudaProfilerStart());
        launch_megakernel(device.graph(), workers, full);
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaProfilerStop());
        compare_outputs();
    }
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
        launch_staged(op, stream);
    CUDA(cudaStreamEndCapture(stream, &graph));
    CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    std::vector<float> fused_times, staged_times, graph_times, reset_times;
    for (int i = 0; i < 15; ++i) {
        device.reset();
        CUDA(cudaEventRecord(begin, stream));
        launch_megakernel(device.graph(), workers, full, stream);
        CUDA(cudaEventRecord(end, stream));
        CUDA(cudaEventSynchronize(end));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, begin, end));
        if (i >= 5)
            fused_times.push_back(ms);
        CUDA(cudaEventRecord(begin, stream));
        for (auto op : ops)
            launch_staged(op, stream);
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
        launch_megakernel(device.graph(), workers, full, stream);
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
#ifdef NANO_MIN_BLOCKS
        CUDA(cudaFuncSetAttribute(megakernel<>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));
        CUDA(cudaFuncSetAttribute(megakernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));
#ifdef NANO_ALIGNED
        CUDA(cudaFuncSetAttribute(megakernel<false, true>,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));
        CUDA(cudaFuncSetAttribute(megakernel<true, true>,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));
#endif
#endif
        cudaFuncAttributes attr;
#ifdef NANO_ALIGNED
        const auto measured_kernel = megakernel<false, true>;
#else
        const auto measured_kernel = megakernel<>;
#endif
        CUDA(cudaFuncGetAttributes(&attr, measured_kernel));
        int resident_blocks;
        CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks, measured_kernel,
                                                           threads, 0));
        printf("megakernel registers=%d static_shared=%zu local_bytes=%zu\n", attr.numRegs,
               attr.sharedSizeBytes, attr.localSizeBytes);
        printf("resident_blocks_per_SM=%d\n", resident_blocks);
        printf("operand_K=%d\n", k_tile);
#ifdef NANO_FIFO
        puts("scheduler=FIFO");
#else
        puts("scheduler=ready_first");
#endif
#ifdef NANO_SERIAL
        puts("operand_stages=1");
#else
        puts("operand_stages=2");
#endif
    #ifdef NANO_FRONTIER
        constexpr int main_mlp_width = frontier::mlp_width;
        constexpr auto main_token_counts = frontier::unique_local_tokens;
        printf("target=PR360 commit=%s MLP=E4M3/E5M2 width=%d; component only\n",
               frontier::commit, main_mlp_width);
#else
        constexpr int main_mlp_width = 3072;
        constexpr std::array<int, 3> main_token_counts{16384, 32768, 49152};
        puts("target=merged-record MLP=E4M3 width=3072; component only");
#endif
        bool quick = false, main_shapes = false, profile = false;
#ifdef NANO_WGMMA
        int gradient_chunk = 0;
#else
        int gradient_chunk = 4096;
#endif
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--quick")
                quick = true;
            else if (arg == "--main-shapes")
                main_shapes = true;
            else if (arg == "--profile")
                profile = true;
            else if (arg == "--stream-gradients")
                gradient_chunk = 1024;
            else if (arg.rfind("--gradient-chunk=", 0) == 0)
                gradient_chunk = std::stoi(arg.substr(17));
            else
                throw std::runtime_error("unknown argument: " + arg);
        }
#ifdef NANO_WGMMA
        if (gradient_chunk)
            throw std::runtime_error("chunked accumulation is implemented for MMA only");
#endif
        if (gradient_chunk < 0 || gradient_chunk % tile != 0)
            throw std::runtime_error("gradient chunk must be a nonnegative multiple of 64");
        if (profile) {
            validate(16384, frontier::model_dim, main_mlp_width, resident_blocks * prop.multiProcessorCount, true, 42, false,
                     gradient_chunk, true);
            puts("PASS: profile-region output validation");
            return 0;
        }
        for (int workers :
             {1, 7, prop.multiProcessorCount, (resident_blocks + 1) * prop.multiProcessorCount}) {
            validate(7, 17, 33, workers, false, 42, false, gradient_chunk);
            validate(129, 96, 160, workers, false, 1337, false, gradient_chunk);
            validate(65, 32, 64, workers, false, 2, true, gradient_chunk);
#ifdef NANO_ALIGNED
            validate(256, 128, 256, workers, false, 1337, false, gradient_chunk);
#endif
        }
        if (gradient_chunk)
            validate(2 * gradient_chunk + 17, 96, 160, 7, false, 1337, false, gradient_chunk);
#ifdef NANO_ALIGNED
        if (gradient_chunk)
            validate(2 * gradient_chunk, 128, 256, 7, false, 1337, false, gradient_chunk);
#endif
        if (main_shapes) {
            for (int m : main_token_counts)
                validate(m, frontier::model_dim, main_mlp_width, resident_blocks * prop.multiProcessorCount, true, 42, false,
                         gradient_chunk);
        } else if (!quick) {
            for (int blocks_per_sm : {1, 2, 4, resident_blocks}) {
                validate(256, frontier::model_dim, main_mlp_width, blocks_per_sm * prop.multiProcessorCount, true, 42, false,
                         gradient_chunk);
                validate(8192, frontier::model_dim, main_mlp_width, blocks_per_sm * prop.multiProcessorCount, true, 42, false,
                         gradient_chunk);
            }
        }
        puts("PASS: FP8 six-GEMM MLP graph, all outputs and gradients; not full-model training.");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

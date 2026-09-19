#include "bench.cuh"
#include "anvil_graph.cuh"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <memory>
#include <random>

using namespace nano;

uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
float from_bits(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }
float lerp_reference(float a, float b, float w) {
    return std::fma(std::abs(w) >= 0.5f ? w - 1.0f : w, b - a, std::abs(w) >= 0.5f ? b : a);
}

struct Matrix {
    int m, n, d, lanes;
    DeviceBuffer<bf16> gradient, x, work, gram, polynomial;
    DeviceBuffer<float> fast, slow, energy, power, gain, norm;
    DeviceBuffer<uint16_t> parameter, mantissa;
    DeviceBuffer<AnvilScalars> scalars;
    Matrix(int rows, int cols) : m(rows), n(cols), d(std::min(m, n)), lanes(std::max(m, n)),
        gradient(size_t(m) * n), x(gradient.n), work(gradient.n), gram(size_t(d) * d), polynomial(gram.n),
        fast(gradient.n), slow(gradient.n), energy(lanes), power(lanes), gain(lanes), norm(2),
        parameter(gradient.n), mantissa(gradient.n), scalars(1) {}
    Anvil op() {
        return {gradient.p, fast.p, slow.p, x.p, work.p, gram.p, polynomial.p, energy.p, power.p,
                gain.p, norm.p, parameter.p, mantissa.p, scalars.p, m, n};
    }
    void init(int seed, int pattern) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal;
        std::vector<bf16> g(gradient.n);
        std::vector<uint16_t> high(g.size()), low(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            g[i] = bf16(pattern == 1 ? 0.0f : normal(rng) * 0.03f);
            if (pattern == 2)
                g[i] = bf16((i / n % 2 ? -1.0f : 1.0f) * (int(i % n % 8) - 4) * 0.00390625f);
            uint32_t shadow = bits(normal(rng) * 0.05f);
            high[i] = shadow >> 16; low[i] = shadow & 0xffff;
        }
        gradient.put(g); parameter.put(high); mantissa.put(low);
        fast.put(std::vector<float>(fast.n)); slow.put(std::vector<float>(slow.n));
        energy.put(std::vector<float>(energy.n));
        scalars.put({{0.95f, 2.25f * 0.023f, 0.95f, 1.0f, 0.023f}});
    }
    void poison() {
        for (auto *buffer : {&x, &work, &gram, &polynomial})
            CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(bf16)));
        for (auto *buffer : {&power, &gain, &norm})
            CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(float)));
    }
};

struct Snapshot {
    std::vector<bf16> grad;
    std::vector<float> fast, slow, energy;
    std::vector<uint16_t> parameter, mantissa;
    AnvilScalars scalar;
    explicit Snapshot(const Matrix &m) : grad(m.gradient.get()), fast(m.fast.get()), slow(m.slow.get()),
        energy(m.energy.get()), parameter(m.parameter.get()), mantissa(m.mantissa.get()), scalar(m.scalars.get()[0]) {}
    void restore(Matrix &m) const {
        m.gradient.put(grad); m.fast.put(fast); m.slow.put(slow); m.energy.put(energy);
        m.parameter.put(parameter); m.mantissa.put(mantissa); m.scalars.put({scalar});
    }
};

__global__ void reset_anvil(Graph graph, const int *initial, int group_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < graph.task_count)
        graph.queue[i] = initial[i];
    if (i < group_count)
        graph.counters[i] = 0;
    if (i < 4)
        graph.state[i] = i == 2 ? graph.task_count : i == 3 ? 0 : graph.root_count;
}

__global__ void staged_anvil(Graph graph, int begin) {
    __shared__ __align__(1024) bf16 scratch[bf16_scratch_bytes / sizeof(bf16)];
    Task task = graph.tasks[begin + blockIdx.x];
    if (task.kind == TaskKind::bf16_matmul) {
        BF16Matmul op = graph.bf16_ops[task.op];
        bf16_matmul_tile(op, task.row, task.col, scratch);
    } else {
        Anvil op = graph.anvil[task.op];
        anvil_tile(op, AnvilStage(task.col), task.row, reinterpret_cast<float *>(scratch));
    }
}

struct Schedule {
    AnvilPlan plan;
    DeviceBuffer<Anvil> matrices;
    DeviceBuffer<BF16Matmul> products;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, initial, state, audit;
    explicit Schedule(const std::vector<Anvil> &ops) : plan(ops), matrices(ops.size()),
        products(plan.products.size()), tasks(plan.tasks.size()), groups(plan.groups.size()),
        counters(groups.n), queue(tasks.n), initial(tasks.n), state(4), audit(tasks.n + 2) {
        matrices.put(ops); products.put(plan.products); tasks.put(plan.tasks); groups.put(plan.groups);
        std::vector<int> entries(tasks.n, 0);
        std::mt19937 rng(9981);
        auto roots = plan.roots;
        std::shuffle(roots.begin(), roots.end(), rng);
        for (size_t i = 0; i < roots.size(); ++i)
            entries[i] = roots[i] + 1;
        initial.put(entries);
    }
    Graph graph(bool checked = false) {
        return {nullptr, tasks.p, groups.p, counters.p, queue.p, state.p, int(tasks.n), int(plan.roots.size()),
                checked ? audit.p : nullptr, nullptr, nullptr, products.p, matrices.p};
    }
    void reset() {
        reset_anvil<<<(std::max(tasks.n, groups.n) + 127) / 128, 128>>>(graph(), initial.p, int(groups.n));
    }
    void staged() {
        for (const auto &stage : plan.stages)
            staged_anvil<<<stage.second - stage.first, 128>>>(graph(), stage.first);
    }
    void run(int workers, bool checked, bool all_families = false) {
        reset();
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, false, true, true><<<workers, 128>>>(graph(true));
        } else if (all_families) {
            megakernel<false, false, true, true, true><<<workers, 128>>>(graph());
        } else {
            megakernel<false, false, false, true, true><<<workers, 128>>>(graph());
        }
        CHECK_CUDA(cudaGetLastError());
    }
    void check_audit() {
        if (state.get()[2] != 0)
            throw std::runtime_error("unfinished ANVIL tasks");
        for (int count : counters.get())
            if (count <= 0)
                throw std::runtime_error("unreached ANVIL dependency");
        auto counts = audit.get();
        for (size_t i = 0; i < tasks.n; ++i)
            if (counts[i] != 1)
                throw std::runtime_error("ANVIL task did not execute exactly once");
    }
};

struct GraphControl {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    GraphControl(Schedule &schedule, bool independent) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t previous = nullptr;
        for (const auto &stage : schedule.plan.stages) {
            const Task &first = schedule.plan.tasks[stage.first];
            if (independent && first.kind == TaskKind::anvil && first.col == int(AnvilStage::velocity))
                previous = nullptr;
            Graph descriptor = schedule.graph();
            int begin = stage.first;
            void *args[] = {&descriptor, &begin};
            cudaKernelNodeParams params{};
            params.func = reinterpret_cast<void *>(staged_anvil);
            params.gridDim = dim3(stage.second - stage.first);
            params.blockDim = dim3(128);
            params.kernelParams = args;
            cudaGraphNode_t node;
            CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, previous ? &previous : nullptr, previous ? 1 : 0, &params));
            previous = node;
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~GraphControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}

// Independent oracle: unpack BF16 on the CPU, pedantic FP32 cuBLAS, then CPU rounding.
struct Reference {
    cublasHandle_t handle;
    Reference() {
        blas_check(cublasCreate(&handle));
        blas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    }
    ~Reference() { cublasDestroy(handle); }
    std::vector<bf16> product(const std::vector<bf16> &a, const std::vector<bf16> &b,
                             const std::vector<bf16> *c, int m, int n, int k,
                             int ar, int ak, int br, int bk, float alpha = 1, float beta = 0,
                             bool split = false, bool symmetric = false) {
        std::vector<float> fa(size_t(m) * k), fb(size_t(n) * k);
        for (int r = 0; r < m; ++r)
            for (int j = 0; j < k; ++j)
                fa[r * k + j] = float(a[r * ar + j * ak]);
        for (int r = 0; r < n; ++r)
            for (int j = 0; j < k; ++j)
                fb[r * k + j] = float(b[r * br + j * bk]);
        DeviceBuffer<float> da(fa.size()), db(fb.size()), out(size_t(m) * n);
        da.put(fa); db.put(fb);
        float one = 1, zero = 0;
        blas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                              &one, db.p, k, da.p, k, &zero, out.p, n));
        auto value = out.get();
        std::vector<bf16> result(value.size());
        for (int r = 0; r < m; ++r)
            for (int col = 0; col < n; ++col) {
                if (symmetric && r < col)
                    continue;
                int i = r * n + col;
                float x = alpha * value[i];
                if (split)
                    x = float(bf16(x));
                if (c)
                    x = std::fma(beta, float((*c)[i]), x);
                result[i] = bf16(x);
                if (symmetric)
                    result[col * n + r] = result[i];
            }
        return result;
    }

    struct Result {
        Snapshot state;
        std::vector<bf16> x;
        Result(const Snapshot &s) : state(s), x(s.grad.size()) {}
    };
    Result run(const Matrix &matrix, const Snapshot &before) {
        Result out(before);
        auto &state = out.state;
        const auto s = state.scalar;
        for (size_t i = 0; i < out.x.size(); ++i) {
            float g = float(state.grad[i]);
            state.fast[i] = lerp_reference(state.fast[i], g, 1.0f - s.fast_beta);
            state.slow[i] = lerp_reference(state.slow[i], g, 0.02f);
            float mix = std::fma(s.fast_weight, state.fast[i], (1.0f - s.fast_weight) * state.slow[i]);
            out.x[i] = bf16(lerp_reference(g, mix, s.momentum));
        }
        int m = matrix.m, n = matrix.n, d = matrix.d;
        auto gram = [&] {
            int stride = m > n ? 1 : n, inner = m > n ? n : 1;
            return product(out.x, out.x, nullptr, d, d, std::max(m, n), stride, inner,
                           stride, inner, 1, 0, false, true);
        };
        auto a = gram();
        // FP64 sums avoid sharing the CUDA worker's reduction implementation.
        double trace = 0;
        for (int i = 0; i < d; ++i)
            trace += float(a[i * d + i]);
        float divisor = std::fma(std::sqrt(float(trace)), 1.05f, 1e-6f);
        for (auto &x : out.x) x = bf16(float(x) / divisor);
        for (auto &x : a) x = bf16(float(x) / (divisor * divisor));
        for (int i = 0; i < 6; ++i) {
            if (i) a = gram();
            auto b = product(a, a, &a, d, d, d, d, 1, d, 1,
                             float(anvil_maps[i][2]), float(anvil_maps[i][1]), false, true);
            if (m > n)
                out.x = product(out.x, b, &out.x, m, n, d, n, 1, d, 1, 1, float(anvil_maps[i][0]), m > 1024);
            else
                out.x = product(b, out.x, &out.x, m, n, d, d, 1, 1, n, 1, float(anvil_maps[i][0]), m > 1024);
        }
        std::vector<float> power(matrix.lanes), gain(matrix.lanes);
        double pre = 0, post = 0;
        for (int lane = 0; lane < matrix.lanes; ++lane) {
            double sum = 0;
            for (int j = 0; j < d; ++j) {
                float x = float(out.x[m >= n ? lane * n + j : j * n + lane]);
                sum += double(x) * x;
            }
            power[lane] = float(sum) / float(d);
            state.energy[lane] = lerp_reference(state.energy[lane], power[lane], 0.1f);
            gain[lane] = 1.0f / std::sqrt(std::max(state.energy[lane], 1e-10f));
            pre += power[lane];
            post += (power[lane] * d) * (gain[lane] * gain[lane]);
        }
        float correction = std::sqrt(float(pre) * d) / std::max(std::sqrt(float(post)), 1e-10f);
        for (size_t i = 0; i < out.x.size(); ++i) {
            int lane = m >= n ? int(i) / n : int(i) % n;
            float scale = float(bf16(gain[lane] * correction));
            out.x[i] = bf16(float(out.x[i]) * scale);
            float g = float(out.x[i]);
            float shadow = from_bits((uint32_t(state.parameter[i]) << 16) | state.mantissa[i]);
            float decay = (shadow * (state.slow[i] * shadow >= 0.0f ? 1.0f : 0.0f)) * s.decay;
            shadow = std::fma(-g, s.lr, std::fma(-decay, s.lr, shadow));
            state.parameter[i] = bits(shadow) >> 16;
            state.mantissa[i] = bits(shadow) & 0xffff;
        }
        return out;
    }
};

template <class T> void exact(const char *name, const std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)))
        throw std::runtime_error(std::string("bitwise mismatch: ") + name);
}
template <class A, class B>
void near(const char *name, const std::vector<A> &a, const std::vector<B> &b, double tolerance, double absolute) {
    double err = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = float(a[i]), y = float(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            fprintf(stderr, "%s non-finite index=%zu actual=%g reference=%g\n", name, i, x, y);
            throw std::runtime_error(std::string("non-finite ") + name);
        }
        err += (x - y) * (x - y); norm += y * y; maximum = std::max(maximum, std::abs(x - y));
    }
    double relative = std::sqrt(err / std::max(norm, 1e-30));
    printf("ANVIL %s rel_l2=%.9g max_abs=%.9g\n", name, relative, maximum);
    if (relative > tolerance && maximum > absolute)
        throw std::runtime_error(std::string("independent reference mismatch: ") + name);
}

void check_state(const Matrix &m, const Snapshot &before, const Reference::Result &reference) {
    Snapshot actual(m);
    exact("fast velocity", actual.fast, reference.state.fast);
    exact("slow velocity", actual.slow, reference.state.slow);
    // Six BF16 polynomial maps amplify rounding differences between MMA and SGEMM.
    // These are component gates, not bitwise parity with the pinned trainer.
    near("equalized update", m.x.get(), reference.x, 0.01, 2e-5);
    near("lane energy", actual.energy, reference.state.energy, 0.01, 1e-7);
    auto x = m.x.get();
    std::vector<float> shadow_actual(x.size()), shadow_reference(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float shadow = from_bits((uint32_t(before.parameter[i]) << 16) | before.mantissa[i]);
        float decay = (shadow * (actual.slow[i] * shadow >= 0 ? 1.0f : 0.0f)) * before.scalar.decay;
        float expected = std::fma(-float(x[i]), before.scalar.lr, std::fma(-decay, before.scalar.lr, shadow));
        if (((uint32_t(actual.parameter[i]) << 16) | actual.mantissa[i]) != bits(expected))
            throw std::runtime_error("FP32 shadow / cautious-decay bit mismatch");
        shadow_actual[i] = expected;
        shadow_reference[i] = from_bits((uint32_t(reference.state.parameter[i]) << 16) | reference.state.mantissa[i]);
    }
    near("parameter shadow", shadow_actual, shadow_reference, 0.01, 2e-6);
}

void check(const std::vector<std::pair<int, int>> &shapes, int workers, bool quick, int pattern = 0) {
    printf("ANVIL batch=%zu workers=%d pattern=%d\n", shapes.size(), workers, pattern);
    std::vector<std::unique_ptr<Matrix>> matrices;
    std::vector<Anvil> ops;
    for (const auto &shape : shapes) {
        printf("ANVIL matrix rows=%d cols=%d split=%d\n", shape.first, shape.second, shape.first > 1024);
        matrices.push_back(std::make_unique<Matrix>(shape.first, shape.second));
        matrices.back()->init(883 + int(matrices.size()), pattern);
        ops.push_back(matrices.back()->op());
    }
    Schedule schedule(ops);
    GraphControl serial_graph(schedule, false), concurrent_graph(schedule, true);
    Reference oracle;
    printf("ANVIL tasks=%zu dependencies=%zu roots=%zu products=%zu\n", schedule.tasks.n,
           schedule.groups.n, schedule.plan.roots.size(), schedule.products.n);
    for (int step = 0; step < 3; ++step) {
        printf("ANVIL step=%d\n", step);
        std::vector<Snapshot> before;
        std::vector<Reference::Result> expected;
        for (size_t i = 0; i < matrices.size(); ++i) {
            auto &m = *matrices[i];
            if (step == 1)
                m.scalars.put({{0.95f, 0.03f, 0.85f, 0.4385f, 0.009f * float(i + 1)}});
            if (step == 2) {
                m.scalars.put({{0.5f, 0.02f, 0.7f, 0.4385f, 0.004f * float(i + 1)}});
                auto gradient = m.gradient.get();
                for (size_t j = 0; j < gradient.size(); ++j)
                    gradient[j] = bf16(float(gradient[j]) * (j % 3 == 0 ? -0.5f : 1.25f));
                m.gradient.put(gradient);
            }
            before.emplace_back(m);
            expected.push_back(oracle.run(m, before.back()));
            m.poison();
        }
        schedule.run(workers, true);
        CHECK_CUDA(cudaDeviceSynchronize());
        schedule.check_audit();
        std::vector<Snapshot> actual;
        std::vector<std::vector<bf16>> result;
        for (size_t i = 0; i < matrices.size(); ++i) {
            check_state(*matrices[i], before[i], expected[i]);
            actual.emplace_back(*matrices[i]); result.push_back(matrices[i]->x.get());
        }
        for (int mode = 0; mode < 7; ++mode) {
            for (size_t i = 0; i < matrices.size(); ++i) {
                before[i].restore(*matrices[i]); matrices[i]->poison();
            }
            if (mode == 0)
                schedule.staged();
            else if (mode == 5)
                serial_graph.run();
            else if (mode == 6)
                concurrent_graph.run();
            else
                schedule.run(mode > 2 ? std::max(1, workers / (mode == 3 ? 2 : 4)) : workers, false, mode == 2);
            CHECK_CUDA(cudaDeviceSynchronize());
            for (size_t i = 0; i < matrices.size(); ++i) {
                Snapshot replay(*matrices[i]);
                exact("replay fast", replay.fast, actual[i].fast); exact("replay slow", replay.slow, actual[i].slow);
                exact("replay energy", replay.energy, actual[i].energy);
                exact("replay parameter", replay.parameter, actual[i].parameter);
                exact("replay mantissa", replay.mantissa, actual[i].mantissa);
                exact("replay X", matrices[i]->x.get(), result[i]);
            }
        }
    }
    if (!quick && !pattern) {
        // Timing repeats actual stateful steps. Both paths start from identical states.
        std::vector<Snapshot> initial;
        for (const auto &m : matrices) initial.emplace_back(*m);
        auto restore = [&] { for (size_t i = 0; i < matrices.size(); ++i) initial[i].restore(*matrices[i]); };
        for (int divisor : {4, 2, 1}) {
            restore();
            printf("ANVIL timing_workers=%d\n", workers / divisor);
            device_time("ANVIL persistent+reset repeated_steps", [&] { schedule.run(workers / divisor, false); });
        }
        restore();
        device_time("ANVIL separate repeated_steps", [&] { schedule.staged(); });
        restore();
        device_time("ANVIL serial CUDA-graph repeated_steps", [&] { serial_graph.run(); });
        restore();
        device_time("ANVIL concurrent CUDA-graph repeated_steps", [&] { concurrent_graph.run(); });
    }
    puts("ANVIL PASS: independent numerical oracle, exact state update, task visits, staged/production/all-family replay");
}

void check_degenerate(int rows, int cols, int pattern) {
    Matrix matrix(rows, cols);
    matrix.init(884, pattern);
    Snapshot before(matrix);
    Reference oracle;
    auto expected = oracle.run(matrix, before);
    Schedule schedule({matrix.op()});
    matrix.poison();
    schedule.run(3, true);
    CHECK_CUDA(cudaDeviceSynchronize());
    schedule.check_audit();
    auto actual = matrix.x.get();
    size_t nonfinite = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        bool reference_finite = std::isfinite(float(expected.x[i]));
        if (reference_finite != bool(std::isfinite(float(actual[i]))))
            throw std::runtime_error("degenerate ANVIL finite/non-finite reference mismatch");
        if (std::isnan(float(actual[i])) != std::isnan(float(expected.x[i])) ||
            (std::isinf(float(actual[i])) && std::signbit(float(actual[i])) != std::signbit(float(expected.x[i]))))
            throw std::runtime_error("degenerate ANVIL NaN/infinity reference mismatch");
        nonfinite += !reference_finite;
    }
    if (!nonfinite) {
        check_state(matrix, before, expected);
    } else {
        exact("degenerate fast", matrix.fast.get(), expected.state.fast);
        exact("degenerate slow", matrix.slow.get(), expected.state.slow);
        printf("ANVIL RECIPE LIMITATION rows=%d cols=%d pattern=%d: native and independent reference both produce %zu/%zu non-finite updates; no clamp applied\n",
               rows, cols, pattern, nonfinite, actual.size());
    }
}

int main(int argc, char **argv) {
    try {
        const std::pair<int, int> invalid_shapes[] = {{0, 1}, {-1, 1}, {INT32_MAX - 1, 1}, {46341, 46341}};
        for (const auto &shape : invalid_shapes) {
            Anvil invalid{};
            invalid.rows = shape.first;
            invalid.cols = shape.second;
            bool rejected = false;
            try { AnvilPlan plan({invalid}); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("ANVIL planner accepted overflowing/invalid dimensions");
        }
        bool quick = false, profile = false;
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--quick") quick = true;
            else if (std::string(argv[i]) == "--profile") profile = true;
            else if (std::string(argv[i]).find("--gradient-chunk=") == 0) {}
            else throw std::runtime_error("unknown ANVIL argument");
        device_info();
        printf("ANVIL idle_ns=%d\n", NANO_ANVIL_IDLE_NS);
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        int workers = prop.multiProcessorCount * 4;
        if (profile) {
            Matrix matrix(2816, 768);
            matrix.init(884, 0);
            Schedule schedule({matrix.op()});
            for (int i = 0; i < 5; ++i) schedule.run(workers, false);
            CHECK_CUDA(cudaDeviceSynchronize());
            schedule.reset();
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, false, true, true><<<workers, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaProfilerStop());
            return 0;
        }
        check({{17, 33}, {33, 17}, {17, 17}}, 3, true);
        check({{31, 65}}, 1, true, 1);
        check_degenerate(1, 1, 0);
        check_degenerate(17, 33, 2);
        check_degenerate(33, 17, 2);
        check({{1024, 17}, {1025, 17}}, 17, true);
        if (!quick) {
            check({{256, 768}}, workers, false);
            check({{768, 768}}, workers, false);
            check({{2816, 768}}, workers, false);
            check({{256, 768}, {768, 768}, {2816, 768}}, workers, false);
            check({{256, 768}, {256, 768}, {256, 768}, {256, 768}, {256, 768}, {256, 768},
                   {768, 768}, {768, 768}, {2816, 768}, {2816, 768}, {2816, 768}}, workers, false);
        }
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}

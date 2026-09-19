#define NANO_TRAINING_HEAD
#include "loss.cu"
#include <cublas_v2.h>

struct HeadStorage {
    LossStorage loss;
    DeviceBuffer<bf16> input, dx, dw;
    DeviceBuffer<fp8> x, xt, weight, weight_t;
    DeviceBuffer<__nv_fp8_e5m2> gradient_t;
    DeviceBuffer<float> scales, raw_logits, raw_dx, raw_dw;
    HeadStorage(int t, int v, int predictions) : loss(t, v, predictions), input(size_t(t) * 768), dx(input.n), dw(size_t(v) * 768),
        x(input.n), xt(input.n), weight(dw.n), weight_t(dw.n), gradient_t(size_t(t) * v),
        scales(2), raw_logits(size_t(t) * v), raw_dx(input.n), raw_dw(dw.n) {}
    void change(int step) {
        loss.change(step);
        loss.settings[0] = (0.0625f * (0.75f / 8) / 448) * (step == 2 ? 2.0f : 1.0f);
        loss.settings[1] = 0.0625f;
        loss.parameters.put({loss.settings[0], loss.settings[1], loss.settings[2]});
        float xs = step == 1 ? 0.25f : 100.0f / 448, ws = step == 1 ? 0.005f : 2.0f / 448;
        scales.put({xs, ws});
        std::mt19937 rng(7349 + step); std::normal_distribution<float> normal;
        std::vector<bf16> values(input.n);
        for (auto &v : values) v = bf16(step == 2 ? 0.0f : normal(rng));
        input.put(values);
        std::vector<fp8> w(weight.n), wt(weight.n);
        for (int row = 0; row < loss.vocabulary; ++row)
            for (int col = 0; col < 768; ++col) {
                fp8 value(step == 1 ? 0.0f : normal(rng) * 0.04f / ws);
                w[size_t(row) * 768 + col] = value;
                wt[size_t(col) * loss.vocabulary + row] = value;
            }
        weight.put(w); weight_t.put(wt);
    }
    TrainingLoss loss_descriptor(bool diagnostics = true) {
        auto op = loss.descriptor(false, diagnostics); op.gradient_transposed = gradient_t.p; return op;
    }
    std::vector<Matmul> products(bool diagnostics = true) {
        int t = loss.tokens, v = loss.vocabulary;
        std::vector<Matmul> ops{
            {x.p, weight.p, nullptr, loss.logits.p, nullptr, t, v, 768, 768, 1, 768, 1, 0, 1, 1, Epilogue::linear},
            {reinterpret_cast<fp8 *>(loss.gradient.p), weight_t.p, nullptr, nullptr, dx.p, t, 768, v, v, 1, v, 1, 0, 1, 1, Epilogue::linear},
            {xt.p, reinterpret_cast<fp8 *>(gradient_t.p), nullptr, nullptr, dw.p, 768, v, t, t, 1, t, 1, 0, 1, 1, Epilogue::linear}};
        ops[1].a_e5 = true; ops[2].b_e5 = true;
        ops[1].precise = ops[2].precise = true;
        if (diagnostics) { ops[0].raw = raw_logits.p; ops[1].raw = raw_dx.p; ops[2].raw = raw_dw.p; }
        return ops;
    }
    void poison() {
        loss.poison();
        for (auto *b : {&x, &xt, &loss.logits}) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n));
        CHECK_CUDA(cudaMemset(gradient_t.p, 0xff, gradient_t.n));
        for (auto *b : {&dx, &dw}) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : {&raw_logits, &raw_dx, &raw_dw}) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
    }
};

struct HeadPlan {
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    explicit HeadPlan(const HeadStorage &b) {
        int t = b.loss.tokens, v = b.loss.vocabulary, rows = (t + 3) / 4, columns = (v + loss_tile - 1) / loss_tile;
        auto stage = [&](TaskKind kind, int op, int nr, int nc, int signal) {
            int begin = int(tasks.size());
            for (int row = 0; row < nr; ++row)
                for (int col = 0; col < nc; ++col) tasks.push_back({op, row, col, {signal, -1}, 0, 0, kind});
            stages.push_back({begin, int(tasks.size())});
        };
        stage(TaskKind::head_setup, 0, 1, 1, 0);
        stage(TaskKind::head_input, 0, (t + 63) / 64, 12, 1);
        stage(TaskKind::matmul, 0, (t + 63) / 64, (v + 63) / 64, 2);
        stage(TaskKind::loss_partial, 0, rows, columns, -1);
        stage(TaskKind::loss_reduce, 0, rows, 1, -1);
        stage(TaskKind::loss_gradient, 0, rows, columns, 3 + rows * 2);
        stage(TaskKind::matmul, 1, (t + 63) / 64, 12, -1);
        stage(TaskKind::matmul, 2, 12, (v + 63) / 64, -1);
        for (int i = 0; i < 3; ++i)
            groups.push_back({stages[i].second - stages[i].first, stages[i + 1].first, stages[i + 1].second});
        for (int row = 0; row < rows; ++row) {
            int begin = stages[3].first + row * columns, reduce = stages[4].first + row, gradient = stages[5].first + row * columns;
            for (int task = begin; task < begin + columns; ++task) tasks[task].signal[0] = int(groups.size());
            groups.push_back({columns, reduce, reduce + 1});
            tasks[reduce].signal[0] = int(groups.size());
            groups.push_back({1, gradient, gradient + columns});
        }
        groups.push_back({rows * columns, stages[6].first, stages[7].second});
    }
};

struct HeadSchedule {
    HeadPlan plan;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    DeviceBuffer<Matmul> products;
    DeviceBuffer<TrainingLoss> losses;
    DeviceBuffer<HeadInput> inputs;
    DeviceBuffer<HeadSetup> setups;
    explicit HeadSchedule(HeadStorage &b) : plan(b), tasks(plan.tasks.size()), groups(plan.groups.size()),
        counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2), products(3), losses(1), inputs(1), setups(1) {
        tasks.put(plan.tasks); groups.put(plan.groups); products.put(b.products()); losses.put({b.loss_descriptor()});
        inputs.put({{b.input.p, b.x.p, b.xt.p, b.scales.p, b.loss.tokens}});
        setups.put({{products.p, b.scales.p, b.loss.parameters.p}});
    }
    Graph graph(bool checked = false) {
        Graph g{};
        g.ops = products.p; g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p; g.queue = queue.p;
        g.state = state.p; g.task_count = int(tasks.n); g.root_count = 1; g.audit = checked ? audit.p : nullptr;
        g.training_losses = losses.p; g.head_inputs = inputs.p; g.head_setups = setups.p;
        return g;
    }
    void run(int workers, bool checked = false) {
        reset_loss<<<(tasks.n + 127) / 128, 128>>>(graph(), int(groups.n));
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, false, true, false, false, false, true><<<workers, 128>>>(graph(true));
        } else megakernel<false, false, false, true, false, false, false, true><<<workers, 128>>>(graph());
        CHECK_CUDA(cudaGetLastError());
    }
    void verify() {
        auto visits = audit.get(), completed = counters.get();
        if (state.get()[2]) throw std::runtime_error("unfinished head tasks");
        for (size_t i = 0; i < tasks.n; ++i) if (visits[i] != 1) throw std::runtime_error("head task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i)
            if (completed[i] != plan.groups[i].expected) throw std::runtime_error("head dependency mismatch");
    }
};

template <int MinimumBlocks>
__global__ __launch_bounds__(128, MinimumBlocks) void head_stage(Graph g, int begin) {
    __shared__ __align__(1024) fp8 scratch[bf16_scratch_bytes];
    Task task = g.tasks[begin + blockIdx.x];
    if (task.kind == TaskKind::head_setup) head_setup(g.head_setups[task.op]);
    else if (task.kind == TaskKind::head_input) head_input(g.head_inputs[task.op], task.row, task.col, scratch);
    else if (task.kind == TaskKind::loss_partial) training_loss_partial(g.training_losses[task.op], task.row, task.col);
    else if (task.kind == TaskKind::loss_reduce) training_loss_reduce(g.training_losses[task.op], task.row);
    else if (task.kind == TaskKind::loss_gradient) training_loss_gradient(g.training_losses[task.op], task.row, task.col);
    else execute_tile<false, true>(g.ops[task.op], task, scratch);
}

struct HeadControl {
    cudaGraph_t graph; cudaGraphExec_t exec;
    explicit HeadControl(HeadSchedule &schedule, bool bounded = false) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0)); cudaGraphNode_t previous = nullptr, gradients = nullptr;
        int stage = 0;
        for (auto [begin, end] : schedule.plan.stages) {
            Graph g = schedule.graph(); void *args[] = {&g, &begin}; cudaKernelNodeParams p{};
            p.func = bounded ? reinterpret_cast<void *>(head_stage<4>) : reinterpret_cast<void *>(head_stage<1>);
            p.gridDim = dim3(end - begin); p.blockDim = dim3(128); p.kernelParams = args;
            cudaGraphNode_t node;
            cudaGraphNode_t dependency = stage == 7 ? gradients : previous;
            CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr, dependency ? 1 : 0, &p)); previous = node;
            if (stage++ == 5) gradients = node;
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~HeadControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

void check_head_math(HeadStorage &b, HeadSchedule &schedule) {
    auto input = b.input.get(); auto x = b.x.get(), xt = b.xt.get(); auto scales = b.scales.get();
    for (int row = 0; row < b.loss.tokens; ++row)
        for (int col = 0; col < 768; ++col) {
            fp8 expected(float(bf16(float(input[size_t(row) * 768 + col]) / scales[0])));
            if (x[size_t(row) * 768 + col].__x != expected.__x || xt[size_t(col) * b.loss.tokens + row].__x != expected.__x)
                throw std::runtime_error("head input BF16 division/FP8 packing mismatch");
        }
    check_loss_math(b.loss, true);
    auto gradient = b.loss.gradient.get(), gradient_t = b.gradient_t.get();
    for (int row = 0; row < b.loss.tokens; ++row)
        for (int col = 0; col < b.loss.vocabulary; ++col)
            if (gradient[size_t(row) * b.loss.vocabulary + col].__x != gradient_t[size_t(col) * b.loss.tokens + row].__x)
                throw std::runtime_error("head gradient transpose mismatch");
    cublasHandle_t handle;
    auto blas = [](cublasStatus_t status) { if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("head cuBLAS failure"); };
    blas(cublasCreate(&handle)); blas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    auto ops = schedule.products.get();
    for (int product = 0; product < 3; ++product) {
        const auto &op = ops[product];
        auto decode = [](const fp8 *pointer, size_t count, bool e5) {
            std::vector<fp8> packed(count); CHECK_CUDA(cudaMemcpy(packed.data(), pointer, count, cudaMemcpyDeviceToHost));
            std::vector<float> result(count); for (size_t i = 0; i < count; ++i) result[i] = decode_fp8(packed[i], e5); return result;
        };
        DeviceBuffer<float> a(size_t(op.m) * op.k), d(size_t(op.n) * op.k), expected(size_t(op.m) * op.n);
        a.put(decode(op.a, a.n, op.a_e5)); d.put(decode(op.b, d.n, op.b_e5));
        float one = 1, zero = 0;
        blas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, op.n, op.m, op.k, &one, d.p, op.k, a.p, op.k, &zero, expected.p, op.n));
        auto ref = expected.get(); std::vector<float> raw(ref.size());
        CHECK_CUDA(cudaMemcpy(raw.data(), op.raw, raw.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double error = 0, norm = 0, maximum = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            ref[i] *= op.scale;
            if (!std::isfinite(raw[i])) throw std::runtime_error("non-finite head product");
            double e = double(raw[i]) - ref[i]; error += e * e; norm += double(ref[i]) * ref[i]; maximum = std::max(maximum, std::abs(e));
        }
        double relative = std::sqrt(error / std::max(norm, 1e-30));
        printf("HEAD product=%d M=%d N=%d K=%d rel_l2=%.9g max_abs=%.9g\n", product, op.m, op.n, op.k, relative, maximum);
        if (relative > 2e-4 && maximum > 2e-5) throw std::runtime_error("head product arithmetic gate failed");
        if (product == 0) {
            auto logits = b.loss.logits.get();
            for (size_t i = 0; i < raw.size(); ++i)
                if (logits[i].__x != fp8(raw[i]).__x) throw std::runtime_error("head raw E4M3 logit rounding mismatch");
        } else {
            auto rounded = product == 1 ? b.dx.get() : b.dw.get();
            for (size_t i = 0; i < raw.size(); ++i)
                if (float(rounded[i]) != float(bf16(raw[i]))) throw std::runtime_error("head BF16 gradient rounding mismatch");
        }
    }
    blas(cublasDestroy(handle));
}

void check_head(int tokens, int vocabulary, int predictions, int workers, int steps, bool timing = false) {
    HeadStorage b(tokens, vocabulary, predictions); HeadSchedule schedule(b); HeadControl control(schedule), bounded(schedule, true);
    for (int step = 0; step < steps; ++step) {
        b.change(step); b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify(); check_head_math(b, schedule);
        auto dx = b.dx.get(), dw = b.dw.get(); auto logits = b.loss.logits.get(), x = b.x.get(), xt = b.xt.get();
        auto gradient = b.loss.gradient.get(), gradient_t = b.gradient_t.get(); auto loss = b.loss.loss.get();
        for (int mode = 0; mode < 4; ++mode) {
            b.poison(); schedule.products.put(b.products(mode != 3)); schedule.losses.put({b.loss_descriptor(mode != 3)});
            if (mode == 0) control.run(); else if (mode == 1) bounded.run(); else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto same = [](const auto &a, const auto &b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])); };
            if (!same(dx, b.dx.get()) || !same(dw, b.dw.get()) || !same(logits, b.loss.logits.get()) || !same(x, b.x.get()) ||
                !same(xt, b.xt.get()) || !same(gradient, b.loss.gradient.get()) || !same(gradient_t, b.gradient_t.get()) || !same(loss, b.loss.loss.get()))
                throw std::runtime_error("head bitwise replay failed");
        }
        schedule.products.put(b.products()); schedule.losses.put({b.loss_descriptor()});
        printf("HEAD PASS tokens=%d vocabulary=%d predictions=%d workers=%d step=%d tasks=%zu\n", tokens, vocabulary, predictions, workers, step, schedule.tasks.n);
    }
    if (timing) {
        schedule.products.put(b.products(false)); schedule.losses.put({b.loss_descriptor(false)});
        device_time("HEAD CUDA Graph", [&] { control.run(); });
        device_time("HEAD CUDA Graph four-block occupancy", [&] { bounded.run(); });
        device_time("HEAD persistent+reset", [&] { schedule.run(workers); });
    }
}

int main(int argc, char **argv) {
    try {
        device_info(); bool quick = false, profile = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--quick") quick = true;
            else if (std::string(argv[i]) == "--profile") profile = true;
            else if (std::string(argv[i]).rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown head argument");
        }
        if (profile) {
            HeadStorage b(256, 50304, 3); b.change(0); HeadSchedule schedule(b);
            schedule.products.put(b.products(false)); schedule.losses.put({b.loss_descriptor(false)});
            cudaDeviceProp properties{}; CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
            for (int i = 0; i < 5; ++i) schedule.run(properties.multiProcessorCount * 4);
            reset_loss<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, false, true, false, false, false, true><<<properties.multiProcessorCount * 4, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop()); return 0;
        }
        check_head(16, 2048, 3, 1, 3); check_head(65, 2048, 2, 17, 1);
        if (!quick) {
            check_head(16, 50304, 3, 17, 3);
            check_head(80, 10240, 3, 17, 1); check_head(80, 14336, 2, 17, 1); check_head(80, 24576, 1, 17, 1);
            cudaDeviceProp properties{}; CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
            check_head(256, 50304, 3, properties.multiProcessorCount * 4, 1, true);
        }
        puts("PASS: connected FP8 training head; compiled-trainer parity, candidate transport and full-model integration remain"); return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

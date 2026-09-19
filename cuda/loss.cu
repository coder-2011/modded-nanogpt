#include "bench.cuh"
#include "megakernel.cuh"
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <cuda_profiler_api.h>

using namespace nano;
void launch_reference_loss(const TrainingLoss &, const float *);

struct LossStorage {
    int tokens, vocabulary, predictions;
    DeviceBuffer<fp8> logits;
    DeviceBuffer<int64_t> targets, prefixes;
    DeviceBuffer<float> weights, parameters, partial, lse, loss, raw, reference_loss;
    DeviceBuffer<__nv_fp8_e5m2> gradient, reference_gradient;
    std::array<float, 3> settings;
    LossStorage(int t, int v, int p) : tokens(t), vocabulary(v), predictions(p), logits(size_t(t) * v),
        targets(t), prefixes(t), weights(p), parameters(3), partial(size_t(t) * ((v + loss_tile - 1) / loss_tile)),
        lse(t), loss(t), raw(logits.n), reference_loss(t), gradient(logits.n), reference_gradient(logits.n) {
        if (t <= 0 || v <= 0 || v % 2 || p < 1 || p > 3) throw std::runtime_error("invalid training loss geometry");
    }
    void change(int step) {
        std::mt19937 rng(6011 + step * 17 + vocabulary);
        std::normal_distribution<float> normal(0, 5);
        std::vector<fp8> x(logits.n);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = fp8(step == 1 ? 0.0f : normal(rng));
            if (step == 2) { x[i].__x = uint8_t(i % 256); if ((x[i].__x & 127) == 127) x[i] = fp8(0.0f); }
        }
        logits.put(x);
        std::vector<int64_t> y(tokens), prefix(tokens);
        for (int i = 0; i < tokens; ++i) {
            y[i] = i < 3 ? vocabulary - 1 : i % 4 == 0 ? 0 : (i * 509) % vocabulary;
            prefix[i] = i % 3 == 0 ? -1 : i % 3 == 1 ? y[i] : (y[i] + 1) % vocabulary;
        }
        targets.put(y); prefixes.put(prefix);
        std::vector<float> w(predictions);
        for (int i = 0; i < predictions; ++i) w[i] = i == 0 ? 1.0f : step == 1 ? 0.0f : i == 1 ? 0.2f : 0.07f;
        weights.put(w);
        settings = {step == 2 ? 0.0078125f : 0.03125f, step == 1 ? 0.0f : 0.0625f, step == 0 ? 0.15f : step == 1 ? 0.0f : 0.025f};
        parameters.put({settings[0], settings[1], settings[2]});
    }
    TrainingLoss descriptor(bool reference = false, bool diagnostics = true) {
        return {logits.p, targets.p, prefixes.p, weights.p, parameters.p, partial.p, lse.p,
                reference ? reference_loss.p : loss.p, reference ? reference_gradient.p : gradient.p,
                tokens, vocabulary, predictions, diagnostics ? raw.p : nullptr};
    }
    void poison() {
        for (auto *buffer : {&partial, &lse, &loss, &raw}) CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(float)));
        CHECK_CUDA(cudaMemset(gradient.p, 0xff, gradient.n));
    }
};

__global__ void reset_loss(Graph g, int groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g.task_count) g.queue[i] = i < g.root_count ? i + 1 : 0;
    if (i < groups) g.counters[i] = 0;
    if (i < 4) g.state[i] = i == 2 ? g.task_count : i == 3 ? 0 : g.root_count;
}

__global__ void loss_stage(Graph g, int begin) {
    Task task = g.tasks[begin + blockIdx.x];
    const auto &op = g.training_losses[task.op];
    if (task.kind == TaskKind::loss_partial) training_loss_partial(op, task.row, task.col);
    else if (task.kind == TaskKind::loss_reduce) training_loss_reduce(op, task.row);
    else training_loss_gradient(op, task.row, task.col);
}

struct LossSchedule {
    std::vector<Task> host_tasks;
    std::vector<Group> host_groups;
    DeviceBuffer<TrainingLoss> descriptor;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    int rows, columns, roots;
    LossSchedule(LossStorage &b) : descriptor(1),
        tasks(size_t((b.tokens + 3) / 4) * (2 * ((b.vocabulary + loss_tile - 1) / loss_tile) + 1)),
        groups(2 * ((b.tokens + 3) / 4)), counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2),
        rows((b.tokens + 3) / 4), columns((b.vocabulary + loss_tile - 1) / loss_tile), roots(rows * columns) {
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < columns; ++col)
                host_tasks.push_back({0, row, col, {row * 2, -1}, 0, 0, TaskKind::loss_partial});
        for (int row = 0; row < rows; ++row)
            host_tasks.push_back({0, row, 0, {row * 2 + 1, -1}, 0, 0, TaskKind::loss_reduce});
        for (int row = 0; row < rows; ++row) {
            int begin = int(host_tasks.size());
            for (int col = 0; col < columns; ++col)
                host_tasks.push_back({0, row, col, {-1, -1}, 0, 0, TaskKind::loss_gradient});
            host_groups.push_back({columns, roots + row, roots + row + 1});
            host_groups.push_back({1, begin, int(host_tasks.size())});
        }
        tasks.put(host_tasks); groups.put(host_groups); descriptor.put({b.descriptor()});
    }
    Graph graph(bool checked = false) {
        Graph g{};
        g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p; g.queue = queue.p;
        g.state = state.p; g.task_count = int(tasks.n); g.root_count = roots;
        g.audit = checked ? audit.p : nullptr; g.training_losses = descriptor.p;
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
        if (state.get()[2]) throw std::runtime_error("unfinished loss tasks");
        for (size_t i = 0; i < tasks.n; ++i) if (visits[i] != 1) throw std::runtime_error("loss task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i)
            if (completed[i] != host_groups[i].expected) throw std::runtime_error("loss dependency mismatch");
    }
};

struct LossControl {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    explicit LossControl(LossSchedule &schedule) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t previous = nullptr;
        for (auto [begin, end] : std::array<std::pair<int, int>, 3>{{{0, schedule.roots},
                {schedule.roots, schedule.roots + schedule.rows}, {schedule.roots + schedule.rows, int(schedule.tasks.n)}}}) {
            Graph g = schedule.graph();
            void *args[] = {&g, &begin};
            cudaKernelNodeParams params{};
            params.func = reinterpret_cast<void *>(loss_stage); params.gridDim = dim3(end - begin);
            params.blockDim = dim3(128); params.kernelParams = args;
            cudaGraphNode_t node;
            CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, previous ? &previous : nullptr, previous ? 1 : 0, &params));
            previous = node;
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~LossControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

void check_loss_math(LossStorage &b) {
    auto logits = b.logits.get(); auto targets = b.targets.get(), prefix = b.prefixes.get();
    auto weights = b.weights.get(), loss = b.loss.get(), raw = b.raw.get();
    auto gradient = b.gradient.get();
    double maximum_loss = 0, maximum_gradient = 0;
    for (int row : {0, b.tokens - 1}) {
        std::vector<double> sigmoid(b.vocabulary);
        double sum = 0, total_weight = 0;
        for (int col = 0; col < b.vocabulary; ++col) {
            double s = 1.0 / (1.0 + std::exp(-(double(float(logits[size_t(row) * b.vocabulary + col])) + 5.0) / 7.5));
            sigmoid[col] = float(__double2half(s));
            sum += std::exp(23.0 * sigmoid[col] - 23.0);
        }
        double lse = 23.0 + std::log(sum), expected_loss = 0;
        for (int k = 0; k < b.predictions; ++k) {
            total_weight += weights[k];
            if (row + k < b.tokens) expected_loss += weights[k] * (lse - 23.0 * sigmoid[targets[row + k]]);
        }
        if (prefix[row] >= 0) {
            total_weight += b.settings[2];
            expected_loss += b.settings[2] * (lse - 23.0 * sigmoid[prefix[row]]);
        }
        maximum_loss = std::max(maximum_loss, std::abs(loss[row] - expected_loss));
        for (int col = 0; col < b.vocabulary; ++col) {
            double correction = 0;
            for (int k = 0; k < b.predictions && row + k < b.tokens; ++k)
                if (targets[row + k] == col) correction += weights[k];
            if (prefix[row] == col) correction += b.settings[2];
            double expected = b.settings[1] * (23.0 / 7.5) / b.settings[0] *
                (total_weight * std::exp(23.0 * sigmoid[col] - lse) - correction) * sigmoid[col] * (1 - sigmoid[col]);
            maximum_gradient = std::max(maximum_gradient, std::abs(raw[size_t(row) * b.vocabulary + col] - expected));
        }
    }
    for (size_t i = 0; i < raw.size(); ++i) {
        if (!std::isfinite(raw[i]) || gradient[i].__x != __nv_fp8_e5m2(raw[i]).__x)
            throw std::runtime_error("training loss E5M2 boundary mismatch");
    }
    printf("LOSS FP64 max_loss=%.9g max_raw_gradient=%.9g\n", maximum_loss, maximum_gradient);
    if (maximum_loss > 2e-5 || maximum_gradient > 2e-5) throw std::runtime_error("training loss mathematical gate failed");

    launch_reference_loss(b.descriptor(true), b.settings.data());
    CHECK_CUDA(cudaDeviceSynchronize());
    auto expected_loss = b.reference_loss.get(); auto expected_gradient = b.reference_gradient.get();
    double squared = 0, norm = 0, max_loss = 0; size_t changed = 0;
    for (int row = 0; row < b.tokens; ++row) max_loss = std::max(max_loss, double(std::abs(loss[row] - expected_loss[row])));
    for (size_t i = 0; i < gradient.size(); ++i) {
        double a = float(gradient[i]), e = float(expected_gradient[i]);
        if (!std::isfinite(a) || !std::isfinite(e)) throw std::runtime_error("non-finite pinned loss comparison");
        squared += (a - e) * (a - e); norm += e * e;
        if (gradient[i].__x != expected_gradient[i].__x) {
            ++changed;
            if (std::abs(int(gradient[i].__x) - int(expected_gradient[i].__x)) > 1)
                throw std::runtime_error("pinned loss gradient differs by more than one E5M2 code");
        }
    }
    double relative = std::sqrt(squared / std::max(norm, 1e-30));
    printf("LOSS pinned CUDA max_loss=%.9g gradient_rel_l2=%.9g changed_codes=%zu/%zu\n", max_loss, relative, changed, gradient.size());
    if (max_loss > 2e-5 || relative > 2e-3 || changed > gradient.size() / 1000)
        throw std::runtime_error("pinned CUDA training loss gate failed");
}

void check_loss(int tokens, int vocabulary, int predictions, int workers, int steps, bool timed = false) {
    LossStorage b(tokens, vocabulary, predictions);
    LossSchedule schedule(b); LossControl control(schedule);
    for (int step = 0; step < steps; ++step) {
        b.change(step); b.poison(); schedule.run(workers, true);
        CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify(); check_loss_math(b);
        auto loss = b.loss.get(), raw = b.raw.get(), partial = b.partial.get(), lse = b.lse.get();
        auto gradient = b.gradient.get();
        for (int mode = 0; mode < 3; ++mode) {
            b.poison();
            schedule.descriptor.put({b.descriptor(false, mode != 2)});
            if (mode == 0) control.run(); else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto equal = [](const auto &a, const auto &b) { return a.size() == b.size() &&
                std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0; };
            if (!equal(loss, b.loss.get()) || !equal(partial, b.partial.get()) || !equal(lse, b.lse.get()) ||
                !equal(gradient, b.gradient.get()) || (mode != 2 && !equal(raw, b.raw.get())))
                throw std::runtime_error("training loss replay mismatch");
        }
        schedule.descriptor.put({b.descriptor()});
        printf("LOSS PASS tokens=%d vocabulary=%d predictions=%d workers=%d step=%d tasks=%zu\n",
               tokens, vocabulary, predictions, workers, step, schedule.tasks.n);
    }
    if (timed) {
        schedule.descriptor.put({b.descriptor(false, false)});
        device_time("LOSS pinned CUDA", [&] { launch_reference_loss(b.descriptor(true, false), b.settings.data()); });
        device_time("LOSS CUDA Graph", [&] { control.run(); });
        device_time("LOSS persistent+reset", [&] { schedule.run(workers); });
    }
}

int main(int argc, char **argv) {
    try {
        device_info(); bool quick = false, profile = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--quick") quick = true;
            else if (std::string(argv[i]) == "--profile") profile = true;
            else if (std::string(argv[i]).rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown loss argument");
        }
        cudaDeviceProp properties{}; CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        if (profile) {
            LossStorage b(1024, 50304, 3); b.change(0); LossSchedule schedule(b);
            schedule.descriptor.put({b.descriptor(false, false)});
            for (int i = 0; i < 5; ++i) schedule.run(properties.multiProcessorCount * 4);
            reset_loss<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, false, true, false, false, false, true><<<properties.multiProcessorCount * 4, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop()); return 0;
        }
        check_loss(5, 2048, 3, 1, 3);
        check_loss(5, 50304, 3, 17, 3);
        if (!quick) {
            check_loss(17, 10240, 3, 7, 3); check_loss(17, 14336, 2, 17, 3);
            check_loss(17, 24576, 1, 17, 3);
            check_loss(1024, 50304, 3, properties.multiProcessorCount * 4, 1, true);
        }
        puts("PASS: tiled FP8 training loss and logit gradients; head GEMMs and full training integration remain");
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

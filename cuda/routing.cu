#include "bench.cuh"
#include "megakernel.cuh"
#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <cstring>
#include <cuda_profiler_api.h>

using namespace nano;

struct RoutingStorage {
    int t, count, groups;
    DeviceBuffer<bf16> input, normalized, mixed, output, dy, dmixed, dnormalized, dx, coefficient, delta;
    DeviceBuffer<float> raw_norm, raw_mix, raw_output, raw_dmixed, raw_dx, rstd;
    std::vector<std::unique_ptr<DeviceBuffer<bf16>>> sources, gradients;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> coefficient_gradients;
    RoutingStorage(int tokens, int terms, int channel_groups)
        : t(tokens), count(terms), groups(channel_groups), input(size_t(t) * 768), normalized(input.n),
          mixed(input.n), output(input.n), dy(input.n), dmixed(input.n), dnormalized(input.n), dx(input.n),
          coefficient(size_t(t) * count), delta(size_t(t) * count * groups),
          raw_norm(input.n), raw_mix(input.n), raw_output(input.n), raw_dmixed(input.n), raw_dx(input.n), rstd(t) {
        if (tokens <= 0 || terms <= 0 || terms > 11 || channel_groups <= 0 || 768 % channel_groups)
            throw std::runtime_error("unsupported residual routing geometry");
        for (int i = 0; i < count; ++i) {
            sources.push_back(std::make_unique<DeviceBuffer<bf16>>(input.n));
            gradients.push_back(std::make_unique<DeviceBuffer<bf16>>(input.n));
            coefficient_gradients.push_back(std::make_unique<DeviceBuffer<float>>(size_t(t) * groups));
        }
        change(0);
    }
    void change(int step) {
        std::mt19937 rng(3029 + step * 97 + t);
        std::normal_distribution<float> normal;
        for (auto *buffer : {&input, &dy, &coefficient, &delta}) {
            std::vector<bf16> data(buffer->n);
            for (auto &value : data) value = bf16(normal(rng) * (buffer == &delta ? 0.05f : 0.2f));
            if (buffer == &input) {
                for (int c = 0; c < 768; ++c) data[c] = bf16(step == 0 ? 0.0f : step == 1 ? 1e-7f : 10.0f);
            }
            if (buffer == &coefficient && step == 2) std::fill(data.begin(), data.end(), bf16(0.0f));
            buffer->put(data);
        }
        for (auto &buffer : sources) {
            std::vector<bf16> data(buffer->n);
            for (auto &value : data) value = bf16(normal(rng) * 0.3f);
            buffer->put(data);
        }
    }
    ResidualMix mix() {
        ResidualMix op{};
        op.count = count; op.tokens = t; op.output = mixed.p; op.dy = dmixed.p; op.raw = raw_mix.p;
        for (int i = 0; i < count; ++i) {
            auto &term = op.terms[i];
            term.value = i == 0 ? normalized.p : sources[i]->p;
            term.coefficient = coefficient.p + i;
            term.coefficient_row_stride = count;
            term.group_delta = delta.p + i * groups;
            term.delta_row_stride = count * groups;
            term.groups = groups;
            term.constant = i == 0 ? 1.0f : 0.0f;
            term.dvalue = i == 0 ? dnormalized.p : gradients[i]->p;
            term.dcoefficient_rows = coefficient_gradients[i]->p;
        }
        return op;
    }
    std::vector<ResidualNorm> norms() {
        return {{input.p, dnormalized.p, normalized.p, dx.p, t, rstd.p, raw_norm.p, raw_dx.p},
                {mixed.p, dy.p, output.p, dmixed.p, t, nullptr, raw_output.p, raw_dmixed.p}};
    }
    std::vector<DeviceBuffer<bf16> *> outputs() {
        std::vector<DeviceBuffer<bf16> *> result{&normalized, &mixed, &output, &dmixed, &dnormalized, &dx};
        for (int i = 1; i < count; ++i) result.push_back(gradients[i].get());
        return result;
    }
    void poison() {
        for (auto *b : outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : {&raw_norm, &raw_mix, &raw_output, &raw_dmixed, &raw_dx, &rstd})
            CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto &b : coefficient_gradients) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
    }
};

struct RoutingPlan {
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    int roots;
    explicit RoutingPlan(const RoutingStorage &b) {
        int rows = (b.t + 3) / 4;
        roots = rows;
        auto stage = [&](TaskKind kind, int op, bool backward, int columns) {
            int begin = int(tasks.size());
            for (int col = 0; col < columns; ++col)
                for (int row = 0; row < rows; ++row)
                    tasks.push_back({op, row, col, {-1, -1}, int(backward), 0, kind});
            stages.push_back({begin, int(tasks.size())});
        };
        stage(TaskKind::residual_norm, 0, false, 1);
        stage(TaskKind::residual_mix, 0, false, 1);
        stage(TaskKind::residual_norm, 1, false, 1);
        stage(TaskKind::residual_norm, 1, true, 1);
        stage(TaskKind::residual_mix, 0, true, b.count);
        stage(TaskKind::residual_norm, 0, true, 1);
        for (int i = 0; i + 1 < int(stages.size()); ++i) {
            auto [begin, end] = stages[i];
            groups.push_back({end - begin, stages[i + 1].first, stages[i + 1].second});
            for (int task = begin; task < end; ++task) tasks[task].signal[0] = i;
        }
    }
};

__global__ void reset_routing(Graph g, int groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g.task_count) g.queue[i] = i < g.root_count ? i + 1 : 0;
    if (i < groups) g.counters[i] = 0;
    if (i < 4) g.state[i] = i == 2 ? g.task_count : i == 3 ? 0 : g.root_count;
}

template <int MinimumBlocks>
__global__ __launch_bounds__(128, MinimumBlocks) void routing_stage(Graph g, int begin) {
    const Task task = g.tasks[begin + blockIdx.x];
    if (task.kind == TaskKind::residual_norm) residual_norm(g.residual_norm[task.op], task.row, task.k_begin != 0);
    else if (task.k_begin) residual_mix_backward(g.residual_mix[task.op], task.row, task.col);
    else residual_mix_forward(g.residual_mix[task.op], task.row);
}

struct RoutingSchedule {
    RoutingPlan plan;
    DeviceBuffer<ResidualMix> mix;
    DeviceBuffer<ResidualNorm> norms;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    RoutingSchedule(RoutingStorage &b) : plan(b), mix(1), norms(2), tasks(plan.tasks.size()), groups(plan.groups.size()),
        counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2) {
        mix.put({b.mix()}); norms.put(b.norms()); tasks.put(plan.tasks); groups.put(plan.groups);
    }
    Graph graph(bool checked = false) {
        Graph g{};
        g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p; g.queue = queue.p;
        g.state = state.p; g.task_count = int(tasks.n); g.root_count = plan.roots;
        g.audit = checked ? audit.p : nullptr; g.residual_mix = mix.p; g.residual_norm = norms.p;
        return g;
    }
    void run(int workers, bool checked = false) {
        reset_routing<<<(tasks.n + 127) / 128, 128>>>(graph(), int(groups.n));
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, true, true, true, true, true><<<workers, 128>>>(graph(true));
        } else megakernel<false, false, true, true, true, true, true><<<workers, 128>>>(graph());
        CHECK_CUDA(cudaGetLastError());
    }
    void verify() {
        auto visits = audit.get(), completed = counters.get();
        if (state.get()[2]) throw std::runtime_error("unfinished routing tasks");
        for (size_t i = 0; i < tasks.n; ++i)
            if (visits[i] != 1) throw std::runtime_error("routing task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i)
            if (completed[i] != plan.groups[i].expected) throw std::runtime_error("routing dependency mismatch");
    }
};

struct RoutingControl {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    RoutingControl(RoutingSchedule &schedule, bool bounded) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t previous = nullptr;
        for (auto [begin, end] : schedule.plan.stages) {
            Graph g = schedule.graph();
            void *args[] = {&g, &begin};
            cudaKernelNodeParams params{};
            params.func = bounded ? reinterpret_cast<void *>(routing_stage<4>) : reinterpret_cast<void *>(routing_stage<1>);
            params.gridDim = dim3(end - begin); params.blockDim = dim3(128); params.kernelParams = args;
            cudaGraphNode_t node;
            CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, previous ? &previous : nullptr, previous ? 1 : 0, &params));
            previous = node;
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~RoutingControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

void routing_near(const char *name, const std::vector<float> &actual, const std::vector<double> &expected) {
    double squared = 0, reference = 0, maximum = 0;
    if (actual.size() != expected.size()) throw std::runtime_error("routing reference shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite routing result");
        double error = actual[i] - expected[i];
        squared += error * error; reference += expected[i] * expected[i]; maximum = std::max(maximum, std::abs(error));
    }
    double relative = sqrt(squared / std::max(reference, 1e-30));
    printf("ROUTING %s rel_l2=%.9g max_abs=%.9g\n", name, relative, maximum);
    if (!(relative <= 2e-4 || maximum <= 2e-5)) throw std::runtime_error("routing numerical gate failed");
}

void routing_rounding(const std::vector<bf16> &actual, const std::vector<float> &raw) {
    for (size_t i = 0; i < actual.size(); ++i)
        if (float(actual[i]) != float(bf16(raw[i]))) throw std::runtime_error("routing BF16 boundary mismatch");
}

// Independent FP64 row reductions, using the actual BF16 materialized inputs at
// each boundary. This checks arithmetic, not the pinned compiled Torch binary.
void check_routing_math(RoutingStorage &b) {
    auto norm_check = [&](const std::vector<bf16> &x, const std::vector<bf16> &dy,
                          DeviceBuffer<float> &raw, DeviceBuffer<float> &raw_dx,
                          DeviceBuffer<bf16> &out, DeviceBuffer<bf16> &dx) {
        std::vector<double> y(x.size()), gradient(x.size());
        for (int row = 0; row < b.t; ++row) {
            double squares = 0, dot = 0;
            for (int c = 0; c < 768; ++c) {
                int i = row * 768 + c;
                squares += double(float(x[i])) * float(x[i]);
                dot += double(float(x[i])) * float(dy[i]);
            }
            double inverse = 1.0 / sqrt(squares / 768.0 + 0x1p-23);
            for (int c = 0; c < 768; ++c) {
                int i = row * 768 + c;
                y[i] = float(x[i]) * inverse;
                gradient[i] = inverse * (float(dy[i]) - float(x[i]) * dot / 768.0 * inverse * inverse);
            }
        }
        routing_near("RMS forward", raw.get(), y); routing_near("RMS backward", raw_dx.get(), gradient);
        routing_rounding(out.get(), raw.get()); routing_rounding(dx.get(), raw_dx.get());
        auto native_gradient = raw_dx.get();
        for (int row : {0, b.t - 1}) {
            for (int channel : {0, 33, 767}) {
                double epsilon = row == 0 ? 1e-8 : 1e-5;
                auto objective = [&](double change) {
                    double squares = 0, dot = 0;
                    for (int c = 0; c < 768; ++c) {
                        double value = float(x[row * 768 + c]) + (c == channel ? change : 0);
                        squares += value * value; dot += value * float(dy[row * 768 + c]);
                    }
                    return dot / sqrt(squares / 768.0 + 0x1p-23);
                };
                double difference = (objective(epsilon) - objective(-epsilon)) / (2 * epsilon);
                double error = std::abs(native_gradient[row * 768 + channel] - difference);
                if (!(error <= 2e-5 || error <= 2e-4 * std::abs(difference)))
                    throw std::runtime_error("RMS finite-difference check failed");
            }
        }
    };
    norm_check(b.input.get(), b.dnormalized.get(), b.raw_norm, b.raw_dx, b.normalized, b.dx);
    norm_check(b.mixed.get(), b.dy.get(), b.raw_output, b.raw_dmixed, b.output, b.dmixed);
    auto coefficient = b.coefficient.get(), delta = b.delta.get(), dy = b.dmixed.get();
    std::vector<double> output(b.input.n, 0.0);
    for (int term = 0; term < b.count; ++term) {
        auto values = term == 0 ? b.normalized.get() : b.sources[term]->get();
        auto gradient = term == 0 ? b.dnormalized.get() : b.gradients[term]->get();
        std::vector<double> dg(size_t(b.t) * b.groups, 0.0);
        for (int row = 0; row < b.t; ++row) {
            for (int c = 0; c < 768; ++c) {
                int group = c / (768 / b.groups), i = row * 768 + c;
                double weight = (term == 0 ? 1.0 : 0.0) + float(coefficient[row * b.count + term]) +
                    float(delta[(row * b.count + term) * b.groups + group]);
                output[i] += weight * float(values[i]);
                // Products of these BF16 coefficients are exactly representable in FP32.
                if (float(gradient[i]) != float(bf16(float(weight) * float(dy[i]))))
                    throw std::runtime_error("routing source gradient mismatch");
                dg[row * b.groups + group] += double(float(dy[i])) * float(values[i]);
            }
        }
        routing_near("coefficient row/group adjoints", b.coefficient_gradients[term]->get(), dg);
    }
    routing_near("weighted mix", b.raw_mix.get(), output);
    routing_rounding(b.mixed.get(), b.raw_mix.get());
}

struct RoutingResult {
    std::vector<std::vector<bf16>> outputs;
    std::vector<std::vector<float>> coefficients;
    explicit RoutingResult(RoutingStorage &b) {
        for (auto *buffer : b.outputs()) {
            outputs.push_back(buffer->get());
            for (auto value : outputs.back())
                if (!std::isfinite(float(value))) throw std::runtime_error("non-finite routing output");
        }
        for (auto &buffer : b.coefficient_gradients) coefficients.push_back(buffer->get());
    }
    void compare(RoutingStorage &b) {
        size_t i = 0;
        for (auto *buffer : b.outputs()) {
            auto values = buffer->get();
            if (std::memcmp(values.data(), outputs[i++].data(), values.size() * sizeof(bf16)))
                throw std::runtime_error("routing output replay mismatch");
        }
        i = 0;
        for (auto &buffer : b.coefficient_gradients) {
            auto values = buffer->get();
            if (std::memcmp(values.data(), coefficients[i++].data(), values.size() * sizeof(float)))
                throw std::runtime_error("routing coefficient replay mismatch");
        }
    }
};

void check_routing(int tokens, int terms, int groups, int workers, int steps, bool timed = false) {
    RoutingStorage b(tokens, terms, groups);
    RoutingSchedule schedule(b);
    RoutingControl control(schedule, false), bounded(schedule, true);
    for (int step = 0; step < steps; ++step) {
        b.change(step); b.poison(); schedule.run(workers, true);
        CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify();
        if (!timed) check_routing_math(b);
        RoutingResult expected(b);
        for (int mode = 0; mode < 3; ++mode) {
            b.poison();
            if (mode == 0) control.run();
            else if (mode == 1) bounded.run();
            else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize()); expected.compare(b);
        }
        printf("ROUTING PASS tokens=%d terms=%d groups=%d workers=%d step=%d tasks=%zu\n",
               tokens, terms, groups, workers, step, schedule.tasks.n);
    }
    if (timed) {
        device_time("ROUTING CUDA Graph", [&] { control.run(); });
        device_time("ROUTING CUDA Graph four-block occupancy", [&] { bounded.run(); });
        device_time("ROUTING persistent+reset", [&] { schedule.run(workers); });
    }
}

int main(int argc, char **argv) {
    try {
        device_info();
        bool quick = false, profile = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--quick") quick = true;
            else if (std::string(argv[i]) == "--profile") profile = true;
            else if (std::string(argv[i]).rfind("--gradient-chunk=", 0) == 0) continue;
            else throw std::runtime_error("unknown routing argument");
        }
        if (profile) {
            RoutingStorage b(16384, 11, 12);
            RoutingSchedule schedule(b);
            cudaDeviceProp properties{};
            CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
            for (int i = 0; i < 5; ++i) schedule.run(properties.multiProcessorCount * 4);
            reset_routing<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, true, true, true, true, true><<<properties.multiProcessorCount * 4, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop());
            return 0;
        }
        check_routing(17, 2, 1, 1, quick ? 2 : 3);
        check_routing(32, 4, 2, 7, quick ? 2 : 3);
        check_routing(80, 11, 12, 17, quick ? 2 : 3);
        if (!quick) {
            check_routing(16, 4, 6, 7, 3);
            cudaDeviceProp properties{};
            CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
            check_routing(16384, 11, 12, properties.multiProcessorCount * 4, 1, true);
        }
        puts("PASS: residual mixing and RMS normalization forward/backward; model integration remains");
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

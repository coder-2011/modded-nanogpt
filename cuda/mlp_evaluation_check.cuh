#pragma once
#include "mlp_evaluation.cuh"

struct MLPEvaluationStorage {
    int t;
    DeviceBuffer<bf16> input, up, down, pre, post, output;
    DeviceBuffer<float> raw_up, raw_down;
    MLPEvaluationStorage(int tokens, bool diagnostics)
        : t(tokens), input(size_t(t) * frontier::model_dim), up(size_t(frontier::mlp_width) * frontier::model_dim),
          down(up.n), pre(diagnostics ? size_t(t) * frontier::mlp_width : 0),
          post(size_t(t) * frontier::mlp_width), output(input.n),
          raw_up(diagnostics ? post.n : 0), raw_down(diagnostics ? output.n : 0) {
        std::mt19937 rng(1013 + t);
        std::normal_distribution<float> normal;
        for (auto *buffer : {&input, &up, &down}) {
            std::vector<bf16> values(buffer->n);
            float scale = buffer == &input ? 0.4f : 0.04f;
            for (auto &value : values) value = bf16(normal(rng) * scale);
            buffer->put(values);
        }
    }
    MLPEvaluationBuffers buffers() {
        return {input.p, up.p, down.p, pre.p, post.p, output.p, raw_up.p, raw_down.p, t};
    }
    void poison() {
        for (auto *buffer : {&pre, &post, &output})
            if (buffer->n) CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(bf16)));
        for (auto *buffer : {&raw_up, &raw_down})
            if (buffer->n) CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(float)));
    }
    std::vector<std::vector<bf16>> result() {
        std::vector<std::vector<bf16>> values{pre.get(), post.get(), output.get()};
        for (const auto &buffer : values)
            for (auto value : buffer)
                if (!std::isfinite(float(value))) throw std::runtime_error("non-finite evaluation MLP output");
        return values;
    }
    void compare(const std::vector<std::vector<bf16>> &expected) {
        int i = 0;
        for (auto *buffer : {&pre, &post, &output}) {
            auto actual = buffer->get();
            if (buffer->n && std::memcmp(actual.data(), expected[i].data(), buffer->n * sizeof(bf16)))
                throw std::runtime_error("evaluation MLP replay mismatch");
            ++i;
        }
    }
};

struct MLPEvaluationSchedule {
    MLPEvaluationPlan plan;
    DeviceBuffer<BF16Matmul> ops;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    explicit MLPEvaluationSchedule(MLPEvaluationStorage &b)
        : plan(b.buffers()), ops(plan.ops.size()), tasks(plan.tasks.size()), groups(plan.groups.size()),
          counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2) {
        ops.put(plan.ops); tasks.put(plan.tasks); groups.put(plan.groups);
    }
    Graph graph(bool checked = false) {
        Graph g{};
        g.bf16_ops = ops.p; g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p;
        g.queue = queue.p; g.state = state.p; g.task_count = int(tasks.n); g.root_count = plan.roots;
        g.audit = checked ? audit.p : nullptr;
        return g;
    }
    void run(int workers, bool checked = false) {
        reset_layer<<<(tasks.n + 127) / 128, 128>>>(graph(), int(groups.n));
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, true, true, true, true><<<workers, 128>>>(graph(true));
        } else {
            megakernel<false, false, true, true, true, true><<<workers, 128>>>(graph());
        }
        CHECK_CUDA(cudaGetLastError());
    }
    void verify() {
        if (state.get()[2]) throw std::runtime_error("unfinished evaluation MLP tasks");
        auto visits = audit.get(), complete = counters.get();
        for (size_t i = 0; i < tasks.n; ++i)
            if (visits[i] != 1) throw std::runtime_error("evaluation MLP task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i)
            if (complete[i] != plan.groups[i].expected) throw std::runtime_error("evaluation MLP dependency mismatch");
    }
};

void check_mlp_evaluation(int tokens, int workers, int steps) {
    MLPEvaluationStorage b(tokens, true);
    MLPEvaluationSchedule schedule(b);
    LayerGraphControl control(schedule), bounded(schedule, true);
    LayerBlasReference reference;
    for (int step = 0; step < steps; ++step) {
        if (step == 1) {
            auto x = b.input.get();
            for (size_t i = 0; i < x.size(); i += 17) x[i] = bf16(-float(x[i]) * 1.25f);
            b.input.put(x);
        }
        if (step == 2) CHECK_CUDA(cudaMemset(b.down.p, 0, b.down.n * sizeof(bf16)));
        b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify();
        for (const auto &op : schedule.plan.ops) reference.check_product(op);
        auto expected = b.result();
        for (int mode = 0; mode < 3; ++mode) {
            b.poison();
            if (mode == 0) control.run();
            else if (mode == 1) bounded.run();
            else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize()); b.compare(expected);
        }
        printf("EVAL MLP PASS tokens=%d step=%d tasks=%zu roots=%d row_dependencies=%zu\n",
               tokens, step, schedule.tasks.n, schedule.plan.roots, schedule.groups.n);
    }
}

void benchmark_mlp_evaluation(int tokens) {
    MLPEvaluationStorage b(tokens, false);
    MLPEvaluationSchedule schedule(b);
    LayerGraphControl control(schedule), bounded(schedule, true);
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    int workers = properties.multiProcessorCount * 4;
    b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify();
    auto expected = b.result();
    for (int mode = 0; mode < 3; ++mode) {
        b.poison();
        if (mode == 0) control.run();
        else if (mode == 1) bounded.run();
        else schedule.run(workers);
        CHECK_CUDA(cudaDeviceSynchronize()); b.compare(expected);
    }
    printf("EVAL MLP BENCH tokens=%d hidden=%d tasks=%zu roots=%d row_dependencies=%zu\n",
           tokens, frontier::mlp_width, schedule.tasks.n, schedule.plan.roots, schedule.groups.n);
    device_time("EVAL MLP CUDA Graph", [&] { control.run(); });
    device_time("EVAL MLP CUDA Graph four-block occupancy", [&] { bounded.run(); });
    for (int multiple : {1, 2, 4}) {
        std::string name = "EVAL MLP persistent+reset workers=" + std::to_string(multiple * properties.multiProcessorCount);
        device_time(name.c_str(), [&] { schedule.run(multiple * properties.multiProcessorCount); });
    }
}

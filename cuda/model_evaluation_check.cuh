#pragma once
#include "model_evaluation.cuh"

struct EvaluationBodyStorage {
    int t;
    DeviceBuffer<bf16> x0, bigram, pre_gate, post_gate, skip_gate, residual_gains, mlp_gains, mu_last, mu_post, mu_groups;
    DeviceBuffer<bf16> initial, last_pre, last_aux, parallel, post_mix, output;
    std::array<std::unique_ptr<DeviceBuffer<bf16>>, 4> values;
    std::array<std::unique_ptr<DeviceBuffer<bf16>>, 11> after_attention, residual;
    std::array<std::unique_ptr<LayerStorage>, 11> attention;
    std::array<std::unique_ptr<MLPEvaluationStorage>, 12> mlp;
    explicit EvaluationBodyStorage(int tokens)
        : t(tokens), x0(size_t(t) * 768), bigram(x0.n), pre_gate(size_t(t) * 26), post_gate(size_t(t) * 41),
          skip_gate(t), residual_gains(22), mlp_gains(11), mu_last(size_t(t) * 14), mu_post(size_t(t) * 10),
          mu_groups(size_t(t) * 120), initial(x0.n), last_pre(x0.n), last_aux(x0.n), parallel(x0.n), post_mix(x0.n), output(x0.n) {
        for (auto &v : values) v = std::make_unique<DeviceBuffer<bf16>>(x0.n);
        for (int i = 0; i < 11; ++i) {
            after_attention[i] = std::make_unique<DeviceBuffer<bf16>>(x0.n);
            residual[i] = std::make_unique<DeviceBuffer<bf16>>(x0.n);
            if (frontier::qk_width[i]) {
                auto role = frontier::attention_role[i];
                attention[i] = std::make_unique<LayerStorage>(t, frontier::qk_width[i], frontier::value_width[i],
                    role.paired, frontier::qk_width[i] == 128, role.auxiliary, role.xsa, role.head_gate, 0, true);
                if (!role.extra_output_gain) {
                    auto scalars = attention[i]->scalars.get(); scalars[5] = 1; attention[i]->scalars.put(scalars);
                }
            }
            if (i != 7) mlp[i] = std::make_unique<MLPEvaluationStorage>(t, true);
        }
        mlp[11] = std::make_unique<MLPEvaluationStorage>(t, true);
        change(0);
    }
    void change(int step) {
        std::mt19937 rng(8173 + t + step * 31);
        std::normal_distribution<float> normal;
        for (auto *buffer : {&x0, &bigram, &pre_gate, &post_gate, &skip_gate, &residual_gains,
                              &mlp_gains, &mu_last, &mu_post, &mu_groups}) {
            std::vector<bf16> data(buffer->n);
            float scale = buffer == &x0 ? 0.5f : 0.05f;
            for (auto &value : data) value = bf16(scale * normal(rng));
            if (buffer == &residual_gains) for (auto &value : data) value = bf16(1.0f + float(value));
            if (buffer == &mlp_gains) for (auto &value : data) value = bf16(0.5f + float(value));
            if (buffer == &mu_last) {
                for (int row = 0; row < t; ++row)
                    for (int c : {8, 9, 12, 13}) data[row * 14 + c] = bf16(1.0f + float(data[row * 14 + c]));
            }
            if (step == 2 && (buffer == &mu_groups || buffer == &pre_gate || buffer == &post_gate))
                std::fill(data.begin(), data.end(), bf16(0.0f));
            buffer->put(data);
        }
        for (auto &buffer : values) {
            std::vector<bf16> data(buffer->n);
            for (auto &value : data) value = bf16(0.2f * normal(rng));
            buffer->put(data);
        }
        // Gate-network producers are not in the body yet. Supply matching packed
        // and compact views here, including live updates between graph replays.
        auto pre = pre_gate.get(), post = post_gate.get();
        for (int i : {1, 3, 10}) {
            std::vector<bf16> alpha(size_t(t) * 6), gate(alpha.size());
            for (int row = 0; row < t; ++row)
                for (int h = 0; h < 6; ++h) {
                    if (i != 10) alpha[row * 6 + h] = pre[row * 26 + (i == 1 ? 0 : 6) + h];
                    gate[row * 6 + h] = i == 10 ? post[row * 41 + 12 + h] : pre[row * 26 + 12 + h];
                }
            attention[i]->alpha.put(alpha); attention[i]->gate.put(gate);
        }
    }
    EvaluationBodyBuffers buffers() {
        EvaluationBodyBuffers b{};
        b.tokens = t; b.x0 = x0.p; b.bigram = bigram.p; b.pre_gate = pre_gate.p; b.post_gate = post_gate.p;
        b.skip_gate = skip_gate.p; b.residual_gains = residual_gains.p; b.mlp_gains = mlp_gains.p;
        b.mu_last = mu_last.p; b.mu_post = mu_post.p; b.mu_groups = mu_groups.p;
        b.initial = initial.p; b.last_pre = last_pre.p; b.last_aux = last_aux.p; b.parallel = parallel.p;
        b.post_mix = post_mix.p; b.output = output.p;
        for (int i = 0; i < 4; ++i) b.values[i] = values[i]->p;
        for (int i = 0; i < 11; ++i) {
            b.after_attention[i] = after_attention[i]->p; b.residual[i] = residual[i]->p;
            if (attention[i]) {
                b.attention[i] = attention[i]->buffers(frontier::qk_width[i] == 128 ? 384 : 128);
                b.attention_input[i] = attention[i]->input.p;
            }
        }
        for (int i = 0; i < 12; ++i) if (mlp[i]) { b.mlp[i] = mlp[i]->buffers(); b.mlp_input[i] = mlp[i]->input.p; }
        return b;
    }
};

struct EvaluationBodySchedule {
    EvaluationBodyPlan plan;
    DeviceBuffer<BF16Matmul> matrices;
    DeviceBuffer<QKVTransform> qkv;
    DeviceBuffer<Attention> attention;
    DeviceBuffer<AttentionPost> post;
    DeviceBuffer<AttentionLayerSetup> setup;
    DeviceBuffer<ResidualMix> mixes;
    DeviceBuffer<ResidualNorm> norms;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> raw_norms, raw_mixes;
    explicit EvaluationBodySchedule(EvaluationBodyStorage &b) : plan(b.buffers()), matrices(plan.matrices.size()),
        qkv(plan.qkv.size()), attention(plan.attention.size()), post(plan.post.size()), setup(plan.setup.size()),
        mixes(plan.mixes.size()), norms(plan.norms.size()), tasks(plan.tasks.size()), groups(plan.groups.size()),
        counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2) {
        for (size_t i = 0; i < plan.setup.size(); ++i) {
            auto [matrix, transform] = plan.setup_offsets[i];
            plan.setup[i].bf16_ops = matrices.p + matrix; plan.setup[i].qkv = qkv.p + transform;
        }
        for (auto &op : plan.norms) {
            raw_norms.push_back(std::make_unique<DeviceBuffer<float>>(b.x0.n)); op.raw = raw_norms.back()->p;
        }
        for (auto &op : plan.mixes) {
            raw_mixes.push_back(std::make_unique<DeviceBuffer<float>>(b.x0.n)); op.raw = raw_mixes.back()->p;
        }
        matrices.put(plan.matrices); qkv.put(plan.qkv); attention.put(plan.attention); post.put(plan.post);
        setup.put(plan.setup); mixes.put(plan.mixes); norms.put(plan.norms); tasks.put(plan.tasks); groups.put(plan.groups);
        if (plan.attention.size() != 7 || plan.matrices.size() != 38 || plan.norms.size() != 17 || plan.mixes.size() != 27)
            throw std::runtime_error("evaluation body topology count mismatch");
        // Attention slots 5/6 correspond to layers 8/10. Both projection inputs
        // must be the same allocation, and the extra MLP must share bank-8 input.
        int a8 = plan.setup_offsets[5].first, a10 = plan.setup_offsets[6].first;
        if (plan.matrices[a8 + 1].a != plan.matrices[a10 + 1].a)
            throw std::runtime_error("layer-8/10 normalization was not shared");
        std::vector<const bf16 *> mlp_inputs;
        for (const auto &op : plan.matrices) if (op.relu_square) mlp_inputs.push_back(op.a);
        if (mlp_inputs.size() != 11 || mlp_inputs[7] != mlp_inputs[8])
            throw std::runtime_error("parallel layer-8 MLP input was not shared");
    }
    Graph graph(bool checked = false) {
        Graph g{}; g.bf16_ops = matrices.p; g.qkv = qkv.p; g.attention = attention.p;
        g.attention_post = post.p; g.layer_setup = setup.p; g.residual_mix = mixes.p; g.residual_norm = norms.p;
        g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p; g.queue = queue.p; g.state = state.p;
        g.task_count = int(tasks.n); g.root_count = plan.roots; g.audit = checked ? audit.p : nullptr; return g;
    }
    void run(int workers, bool checked = false) {
        reset_layer<<<(tasks.n + 127) / 128, 128>>>(graph(), int(groups.n));
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, true, true, true, true, true><<<workers, 128>>>(graph(true));
        } else megakernel<false, false, true, true, true, true, true><<<workers, 128>>>(graph());
        CHECK_CUDA(cudaGetLastError());
    }
    void verify() {
        if (state.get()[2]) throw std::runtime_error("unfinished body tasks");
        auto visits = audit.get(), completed = counters.get();
        for (size_t i = 0; i < tasks.n; ++i) if (visits[i] != 1) throw std::runtime_error("body task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i)
            if (completed[i] != plan.groups[i].expected) throw std::runtime_error("body dependency mismatch");
    }
    std::vector<std::pair<bf16 *, size_t>> boundaries() {
        std::vector<std::pair<bf16 *, size_t>> result;
        for (const auto &op : plan.norms) result.push_back({op.output, size_t(op.tokens) * 768});
        for (const auto &op : plan.mixes) result.push_back({op.output, size_t(op.tokens) * 768});
        for (const auto &op : plan.matrices) result.push_back({op.output, size_t(op.m) * op.n});
        for (const auto &op : plan.qkv) {
            result.push_back({op.q, size_t(op.tokens) * op.heads * op.qk_dim});
            result.push_back({op.k, size_t(op.tokens) * op.heads * op.qk_dim});
            result.push_back({op.v, size_t(op.tokens) * op.heads * op.v_dim});
        }
        for (const auto &op : plan.attention) result.push_back({op.y, size_t(op.tokens) * op.heads * op.v_dim});
        for (const auto &op : plan.post) result.push_back({op.output, size_t(op.tokens) * op.heads * op.value_dim});
        return result;
    }
    void poison() {
        for (auto [ptr, count] : boundaries()) CHECK_CUDA(cudaMemset(ptr, 0xff, count * sizeof(bf16)));
    }
};

void check_evaluation_body_math(EvaluationBodySchedule &schedule) {
    LayerBlasReference reference;
    for (const auto &op : schedule.matrices.get()) reference.check_product(op);
    for (const auto &op : schedule.plan.norms) {
        auto x = layer_read(op.input, size_t(op.tokens) * 768);
        std::vector<double> expected(x.size());
        for (int row = 0; row < op.tokens; ++row) {
            double sum = 0;
            for (int c = 0; c < 768; ++c) sum += double(float(x[row * 768 + c])) * float(x[row * 768 + c]);
            double inverse = 1.0 / sqrt(sum / 768.0 + 0x1p-23);
            for (int c = 0; c < 768; ++c) expected[row * 768 + c] = float(x[row * 768 + c]) * inverse;
        }
        layer_near("body normalization", layer_read(op.raw, x.size()), expected);
    }
    for (const auto &op : schedule.plan.mixes) {
        std::vector<double> expected(size_t(op.tokens) * 768, 0);
        for (int i = 0; i < op.count; ++i) {
            const auto &term = op.terms[i];
            auto x = layer_read(term.value, expected.size());
            auto coef = term.coefficient ? layer_read(term.coefficient,
                size_t(op.tokens - 1) * term.coefficient_row_stride + (term.groups - 1) * term.coefficient_group_stride + 1) : std::vector<bf16>{};
            auto delta = term.group_delta ? layer_read(term.group_delta,
                size_t(op.tokens - 1) * term.delta_row_stride + term.groups) : std::vector<bf16>{};
            for (int row = 0; row < op.tokens; ++row)
                for (int c = 0; c < 768; ++c) {
                    int group = c / (768 / term.groups);
                    double weight = term.constant;
                    if (!coef.empty()) weight += float(coef[row * term.coefficient_row_stride + group * term.coefficient_group_stride]);
                    if (!delta.empty()) weight += float(delta[row * term.delta_row_stride + group]);
                    expected[row * 768 + c] += weight * float(x[row * 768 + c]);
                }
        }
        layer_near("body residual/MUDD mix", layer_read(op.raw, expected.size()), expected);
    }
}

void check_evaluation_body(int tokens, int workers, int steps) {
    EvaluationBodyStorage storage(tokens);
    EvaluationBodySchedule schedule(storage);
    LayerGraphControl control(schedule), bounded(schedule, true);
    for (int step = 0; step < steps; ++step) {
        storage.change(step); schedule.poison(); schedule.run(workers, true);
        CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify(); check_evaluation_body_math(schedule);
        std::vector<std::vector<bf16>> expected;
        auto boundaries = schedule.boundaries();
        for (auto [ptr, count] : boundaries) {
            expected.push_back(layer_read(ptr, count));
            for (auto value : expected.back()) if (!std::isfinite(float(value))) throw std::runtime_error("non-finite body output");
        }
        for (int mode = 0; mode < 3; ++mode) {
            schedule.poison();
            if (mode == 0) control.run(); else if (mode == 1) bounded.run(); else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize());
            for (size_t i = 0; i < boundaries.size(); ++i) {
                auto [ptr, count] = boundaries[i]; auto actual = layer_read(ptr, count);
                if (std::memcmp(actual.data(), expected[i].data(), count * sizeof(bf16)))
                    throw std::runtime_error("body boundary replay mismatch");
            }
        }
        printf("BODY PASS tokens=%d workers=%d step=%d tasks=%zu stages=%zu matrices=%zu attention=%zu norms=%zu mixes=%zu\n",
               tokens, workers, step, schedule.tasks.n, schedule.plan.stages.size(), schedule.plan.matrices.size(),
               schedule.plan.attention.size(), schedule.plan.norms.size(), schedule.plan.mixes.size());
    }
}

int evaluation_body_main(bool quick, bool main_shapes) {
    check_evaluation_body(16, 1, quick ? 1 : 3);
    if (!quick) check_evaluation_body(80, 17, 2);
    if (!quick && main_shapes) {
        cudaDeviceProp properties{};
        CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        check_evaluation_body(16384, properties.multiProcessorCount * 4, 1);
    }
    puts("PASS: eleven-layer BF16 body; prepared embeddings/gates, loss, training and optimizer integration remain");
    return 0;
}

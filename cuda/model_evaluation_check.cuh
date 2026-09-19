#pragma once
#include "model_evaluation.cuh"

struct GateNetworkStorage {
    int columns;
    DeviceBuffer<bf16> up, down, bias, pre, hidden, product;
    GateNetworkStorage(int tokens, int n) : columns(n), up(64 * 768), down(size_t(n > 14 ? 41 : 14) * 64),
        bias(n > 14 ? 41 : 14), pre(size_t(tokens) * 64), hidden(pre.n), product(size_t(tokens) * n) {}
    GateNetwork buffers() { return {up.p, down.p, bias.p, pre.p, hidden.p, product.p}; }
};
struct AuxiliaryGateStorage {
    DeviceBuffer<bf16> weight, input, product, output;
    explicit AuxiliaryGateStorage(int tokens) : weight(6 * 12), input(size_t(tokens) * 12),
        product(size_t(tokens) * 6), output(size_t(tokens) * 768) {}
    AuxiliaryGate buffers() { return {weight.p, input.p, product.p, output.p}; }
};

struct EmbeddingStorage {
    int t;
    DeviceBuffer<int32_t> ids, rows;
    DeviceBuffer<bf16> token_weight, value_weight, cache, signs, embedded, smear_weight, smear_product, smeared;
    DeviceBuffer<float> smear_lambda;
    explicit EmbeddingStorage(int tokens) : t(tokens), ids(t), rows(size_t(t) * 2),
        token_weight(size_t(50304) * 768), value_weight(4 * token_weight.n), cache(size_t(t) * 2 * 768),
        signs(size_t(8192) * 768), embedded(size_t(t) * 768), smear_weight(12), smear_product(t), smeared(embedded.n), smear_lambda(1) {
        std::vector<bf16> table(signs.n);
        for (size_t i = 0; i < table.size(); ++i) {
            uint32_t h = uint32_t(i) * 2654435761u; h ^= h >> 13;
            table[i] = bf16((h & 1) ? 1.0f : -1.0f);
        }
        signs.put(table);
    }
    void change(int step) {
        std::vector<int32_t> tokens(t), indexes(rows.n);
        for (int i = 0; i < t; ++i) {
            tokens[i] = i % 17 == 0 ? 50256 : (i * 997 + step * 131) % 50304;
            indexes[i] = (i * 3 + step) % (2 * t); indexes[t + i] = (i * 7 + t + step) % (2 * t);
        }
        tokens[1] = 0; tokens.back() = 50303;
        ids.put(tokens); rows.put(indexes);
        std::sort(tokens.begin(), tokens.end()); tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
        std::mt19937 rng(4703 + step);
        std::normal_distribution<float> normal;
        std::vector<bf16> row(768);
        // Keep the full canonical table shapes; initialize only rows this fixture
        // reads. Untouched rows are neither copied back nor loaded by the graph.
        for (int id : tokens) {
            for (int plane = -1; plane < 4; ++plane) {
                for (auto &v : row) v = bf16(normal(rng) * (plane < 0 ? 0.5f : 0.2f));
                bf16 *target = plane < 0 ? token_weight.p + int64_t(id) * 768 : value_weight.p + (int64_t(plane) * 50304 + id) * 768;
                CHECK_CUDA(cudaMemcpy(target, row.data(), row.size() * sizeof(bf16), cudaMemcpyHostToDevice));
            }
        }
        for (auto *buffer : {&cache, &smear_weight}) {
            std::vector<bf16> data(buffer->n);
            for (auto &v : data) v = bf16(step == 2 && buffer == &smear_weight ? 0.0f : normal(rng) * 0.1f);
            buffer->put(data);
        }
        smear_lambda.put({step == 0 ? 0.0f : step == 1 ? 0.3f : -0.2f});
    }
    EmbeddingRead buffers() {
        return {ids.p, rows.p, token_weight.p, value_weight.p, cache.p, signs.p, embedded.p, nullptr,
                {nullptr, nullptr, nullptr, nullptr}, t};
    }
};

struct EvaluationBodyStorage {
    int t;
    EmbeddingStorage embedding;
    DeviceBuffer<bf16> x0, bigram, pre_gate, post_gate, skip_gate, residual_gains, mlp_gains, mu_last, mu_post, mu_groups;
    DeviceBuffer<bf16> initial, last_pre, last_aux, parallel, post_mix, output;
    DeviceBuffer<float> gate_scale, skip_lambda;
    DeviceBuffer<bf16> group_weight, group_product;
    std::array<std::unique_ptr<GateNetworkStorage>, 4> networks;
    std::array<std::unique_ptr<AuxiliaryGateStorage>, 3> auxiliary;
    std::array<std::unique_ptr<DeviceBuffer<bf16>>, 4> values;
    std::array<std::unique_ptr<DeviceBuffer<bf16>>, 11> after_attention, residual;
    std::array<std::unique_ptr<LayerStorage>, 11> attention;
    std::array<std::unique_ptr<MLPEvaluationStorage>, 12> mlp;
    explicit EvaluationBodyStorage(int tokens)
        : t(tokens), embedding(tokens), x0(size_t(t) * 768), bigram(x0.n), pre_gate(size_t(t) * 26), post_gate(size_t(t) * 41),
          skip_gate(t), residual_gains(22), mlp_gains(11), mu_last(size_t(t) * 14), mu_post(size_t(t) * 10),
          mu_groups(size_t(t) * 120), initial(x0.n), last_pre(x0.n), last_aux(x0.n), parallel(x0.n), post_mix(x0.n), output(x0.n),
          gate_scale(1), skip_lambda(1), group_weight(14 * 12 * 64), group_product(size_t(t) * 120) {
        int columns[] = {26, 41, 14, 10};
        for (int i = 0; i < 4; ++i) networks[i] = std::make_unique<GateNetworkStorage>(t, columns[i]);
        for (auto &a : auxiliary) a = std::make_unique<AuxiliaryGateStorage>(t);
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
        embedding.change(step);
        std::mt19937 rng(8173 + t + step * 31);
        std::normal_distribution<float> normal;
        for (auto *buffer : {&residual_gains, &mlp_gains}) {
            std::vector<bf16> data(buffer->n);
            float scale = 0.05f;
            for (auto &value : data) value = bf16(scale * normal(rng));
            if (buffer == &residual_gains) for (auto &value : data) value = bf16(1.0f + float(value));
            if (buffer == &mlp_gains) for (auto &value : data) value = bf16(0.5f + float(value));
            buffer->put(data);
        }
        gate_scale.put({step == 0 ? 0.1f : step == 1 ? -0.075f : 0.0f});
        skip_lambda.put({step == 0 ? -1.5f : step == 1 ? 8.0f : -8.0f});
        for (int i = 0; i < 4; ++i) {
            auto &net = *networks[i];
            for (auto *buffer : {&net.up, &net.down, &net.bias}) {
                std::vector<bf16> data(buffer->n);
                for (auto &value : data) value = bf16(normal(rng) * 0.025f);
                if (buffer == &net.down && step == 2) std::fill(data.begin(), data.end(), bf16(0));
                if (buffer == &net.bias) {
                    if (i < 2) for (int c = 12; c < 18; ++c) data[c] = bf16(2.5f);
                    if (i == 1) data[28] = bf16(5.0f);
                    if (i == 2) {
                        data[6] = data[7] = bf16(20.0f);
                        for (int c : {8, 9, 12, 13}) data[c] = bf16(c == 8 || c == 12 ? float(sqrt(1.1) / 0.1) : 10.0f);
                    }
                    if (i == 3) data[1] = bf16(-5.0f);
                }
                buffer->put(data);
            }
        }
        std::vector<bf16> gw(group_weight.n);
        for (auto &value : gw) value = bf16(step == 2 ? 0.0f : normal(rng) * 0.025f);
        group_weight.put(gw);
        for (auto &a : auxiliary) {
            std::vector<bf16> weight(a->weight.n);
            for (auto &value : weight) value = bf16(step == 2 ? 0.0f : normal(rng) * 0.25f);
            a->weight.put(weight);
        }
    }
    EvaluationBodyBuffers buffers() {
        EvaluationBodyBuffers b{};
        b.tokens = t; b.x0 = x0.p; b.bigram = bigram.p; b.pre_gate = pre_gate.p; b.post_gate = post_gate.p;
        b.skip_gate = skip_gate.p; b.residual_gains = residual_gains.p; b.mlp_gains = mlp_gains.p;
        b.embedding = embedding.buffers(); b.smear_weight = embedding.smear_weight.p;
        b.smear_lambda = embedding.smear_lambda.p; b.smear_product = embedding.smear_product.p; b.smeared = embedding.smeared.p;
        b.mu_last = mu_last.p; b.mu_post = mu_post.p; b.mu_groups = mu_groups.p;
        b.gate_scale = gate_scale.p; b.skip_lambda = skip_lambda.p;
        b.group_weight = group_weight.p; b.group_product = group_product.p;
        for (int i = 0; i < 4; ++i) b.networks[i] = networks[i]->buffers();
        for (int i = 0; i < 3; ++i) b.auxiliary[i] = auxiliary[i]->buffers();
        b.initial = initial.p; b.last_pre = last_pre.p; b.last_aux = last_aux.p; b.parallel = parallel.p;
        b.post_mix = post_mix.p; b.output = output.p;
        for (int i = 0; i < 4; ++i) b.values[i] = values[i]->p;
        for (int i = 0; i < 11; ++i) {
            b.after_attention[i] = after_attention[i]->p; b.residual[i] = residual[i]->p;
            if (attention[i]) {
                b.attention[i] = attention[i]->buffers(frontier::qk_width[i] == 128 ? 384 : 128);
                b.attention_input[i] = attention[i]->input.p;
                b.xsa[i] = attention[i]->alpha.p; b.head_gate[i] = attention[i]->gate.p;
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
    DeviceBuffer<GateTransform> gates;
    DeviceBuffer<EmbeddingRead> embeddings;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> raw_norms, raw_mixes, raw_gates, raw_matrices;
    explicit EvaluationBodySchedule(EvaluationBodyStorage &b) : plan(b.buffers()), matrices(plan.matrices.size()),
        qkv(plan.qkv.size()), attention(plan.attention.size()), post(plan.post.size()), setup(plan.setup.size()),
        mixes(plan.mixes.size()), norms(plan.norms.size()), gates(plan.gates.size()), embeddings(plan.embeddings.size()),
        tasks(plan.tasks.size()), groups(plan.groups.size()),
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
        for (auto &op : plan.gates) {
            raw_gates.push_back(std::make_unique<DeviceBuffer<float>>(size_t(op.tokens) * op.columns)); op.raw = raw_gates.back()->p;
        }
        for (auto &op : plan.matrices) if (!op.raw) {
            raw_matrices.push_back(std::make_unique<DeviceBuffer<float>>(size_t(op.m) * op.n)); op.raw = raw_matrices.back()->p;
        }
        matrices.put(plan.matrices); qkv.put(plan.qkv); attention.put(plan.attention); post.put(plan.post);
        setup.put(plan.setup); mixes.put(plan.mixes); norms.put(plan.norms); gates.put(plan.gates); embeddings.put(plan.embeddings);
        tasks.put(plan.tasks); groups.put(plan.groups);
        if (plan.attention.size() != 7 || plan.matrices.size() != 51 || plan.norms.size() != 18 || plan.mixes.size() != 27 || plan.gates.size() != 21)
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
        const bf16 *network_sources[] = {b.x0.p, b.residual[3]->p, b.residual[9]->p, b.residual[10]->p};
        int net = 0;
        for (const auto &op : plan.matrices) if (op.n == 64 && op.k == 768) {
            if (net == 4 || op.a != network_sources[net++]) throw std::runtime_error("gate network source mismatch");
        }
        if (net != 4) throw std::runtime_error("missing gate network");
        if (plan.matrices.back().a != b.networks[3]->hidden.p)
            throw std::runtime_error("post-loop group branch did not share hidden activation");
    }
    Graph graph(bool checked = false) {
        Graph g{}; g.bf16_ops = matrices.p; g.qkv = qkv.p; g.attention = attention.p;
        g.attention_post = post.p; g.layer_setup = setup.p; g.residual_mix = mixes.p; g.residual_norm = norms.p;
        g.gates = gates.p;
        g.embeddings = embeddings.p;
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
        for (const auto &op : plan.gates) result.push_back({op.output, size_t(op.tokens) * op.columns});
        for (const auto &op : plan.embeddings) {
            result.push_back({op.token_output, size_t(op.tokens) * 768});
            result.push_back({op.ngram_output, size_t(op.tokens) * 768});
            for (auto *value : op.value_output) result.push_back({value, size_t(op.tokens) * 768});
        }
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
    for (const auto &op : schedule.plan.embeddings) {
        auto ids = layer_read(op.token_ids, op.tokens), rows = layer_read(op.cache_rows, size_t(op.tokens) * 2);
        auto output = layer_read(op.token_output, size_t(op.tokens) * 768);
        auto ngram = layer_read(op.ngram_output, size_t(op.tokens) * 768);
        std::array<std::vector<bf16>, 4> values;
        for (int p = 0; p < 4; ++p) values[p] = layer_read(op.value_output[p], output.size());
        for (int row = 0; row < op.tokens; ++row) {
            auto token = layer_read(op.token_weight + int64_t(ids[row]) * 768, 768);
            uint64_t cur = ids[row], prev = row ? ids[row - 1] : 0, prev2 = row >= 2 ? ids[row - 2] : 0;
            int sign0 = row ? ((30011 * prev) ^ (48271 * cur)) % 8192 : 0;
            int sign1 = row >= 2 ? ((26801 * prev2) ^ (39779 * prev) ^ (58699 * cur)) % 8192 : 0;
            auto a = layer_read(op.cache + int64_t(rows[row]) * 768, 768);
            auto b = layer_read(op.cache + int64_t(rows[op.tokens + row]) * 768, 768);
            auto sa = layer_read(op.signs + int64_t(sign0) * 768, 768);
            auto sb = layer_read(op.signs + int64_t(sign1) * 768, 768);
            for (int c = 0; c < 768; ++c) {
                if (float(output[row * 768 + c]) != float(token[c])) throw std::runtime_error("token gather mismatch");
                bf16 expected(float(a[c]) * float(sa[c]) + float(b[c]) * float(sb[c]));
                if (float(ngram[row * 768 + c]) != float(expected)) throw std::runtime_error("signed n-gram gather mismatch");
            }
            for (int plane = 0; plane < 4; ++plane) {
                auto selected = layer_read(op.value_weight + (int64_t(plane) * 50304 + ids[row]) * 768, 768);
                if (std::memcmp(selected.data(), values[plane].data() + size_t(row) * 768, 768 * sizeof(bf16)))
                    throw std::runtime_error("value-plane gather mismatch");
            }
        }
        printf("BODY embedding gather/sign PASS tokens=%d vocabulary=50304 value_planes=4\n", op.tokens);
    }
    for (const auto &op : schedule.plan.gates) {
        bool auxiliary = op.kind == GateKind::auxiliary_input || op.kind == GateKind::auxiliary_output;
        bool smear = op.kind == GateKind::smear;
        size_t input_count = smear ? size_t(op.tokens) : auxiliary ? size_t(op.tokens) * (op.kind == GateKind::auxiliary_input ? 768 : 6) :
            size_t(op.tokens - 1) * op.input_stride + op.columns;
        auto input = layer_read(op.input, input_count);
        auto value = auxiliary || smear ? layer_read(op.value, size_t(op.tokens) * 768) : std::vector<bf16>{};
        auto bias = op.bias ? layer_read(op.bias, op.columns) : std::vector<bf16>{};
        float scale = op.scalar ? layer_read(op.scalar, 1)[0] : op.scale;
        if (op.round_scale) scale = float(bf16(scale));
        std::vector<double> expected(size_t(op.tokens) * op.columns);
        for (int row = 0; row < op.tokens; ++row)
            for (int col = 0; col < op.columns; ++col) {
                double x;
                if (smear) {
                    x = float(value[row * 768 + col]);
                    if (row) x += double(scale) / (1 + std::exp(-double(float(input[row])))) * float(value[(row - 1) * 768 + col]);
                } else if (op.kind == GateKind::auxiliary_input) {
                    x = col < 6 ? float(input[row * 768 + col]) : float(value[row * 768 + col - 6]);
                } else if (op.kind == GateKind::auxiliary_output) {
                    x = 2.0 / (1 + std::exp(-double(float(input[row * 6 + col / 128])))) * float(value[row * 768 + col]);
                } else {
                    x = float(input[row * op.input_stride + col]);
                    if (op.kind == GateKind::gelu) x = 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
                    else if (op.kind == GateKind::affine) x = (x + (bias.empty() ? 0 : float(bias[col]))) * double(scale);
                    else if (op.kind == GateKind::skip) x /= 1 + std::exp(-double(scale));
                }
                expected[row * op.columns + col] = x;
            }
        auto raw = layer_read(op.raw, expected.size());
        auto output = layer_read(op.output, expected.size());
        layer_near("gate GELU/scale/packing/auxiliary", raw, expected);
        for (size_t i = 0; i < raw.size(); ++i)
            if (float(output[i]) != float(bf16(raw[i]))) throw std::runtime_error("gate BF16 output rounding mismatch");
    }
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
        printf("BODY PASS tokens=%d workers=%d step=%d tasks=%zu stages=%zu matrices=%zu attention=%zu norms=%zu mixes=%zu gates=%zu\n",
               tokens, workers, step, schedule.tasks.n, schedule.plan.stages.size(), schedule.plan.matrices.size(),
               schedule.plan.attention.size(), schedule.plan.norms.size(), schedule.plan.mixes.size(), schedule.plan.gates.size());
    }
    if (tokens == 16384) {
        device_time("BODY CUDA Graph", [&] { control.run(); });
        device_time("BODY CUDA Graph four-block occupancy", [&] { bounded.run(); });
        device_time("BODY persistent+reset", [&] { schedule.run(workers); });
    }
}

int evaluation_body_profile() {
    EvaluationBodyStorage storage(16384);
    EvaluationBodySchedule schedule(storage);
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    int workers = properties.multiProcessorCount * 4;
    for (int i = 0; i < 5; ++i) schedule.run(workers);
    reset_layer<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
    CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
    megakernel<false, false, true, true, true, true, true><<<workers, 128>>>(schedule.graph());
    CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop());
    printf("BODY PROFILE tokens=16384 workers=%d tasks=%zu stages=%zu\n", workers, schedule.tasks.n, schedule.plan.stages.size());
    return 0;
}

int evaluation_body_main(bool quick, bool main_shapes) {
    check_evaluation_body(16, 1, quick ? 1 : 3);
    if (!quick) check_evaluation_body(80, 17, 2);
    if (!quick && main_shapes) {
        cudaDeviceProp properties{};
        CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        check_evaluation_body(16384, properties.multiProcessorCount * 4, 1);
    }
    puts("PASS: token-to-hidden BF16 body with embedding/gate producers; loss, training, sparse-cache transport and optimizer integration remain");
    return 0;
}

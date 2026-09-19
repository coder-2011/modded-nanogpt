#pragma once
#include "attention_layer.cuh"
#include "mlp_evaluation.cuh"
#include <array>
#include <algorithm>

namespace nano {

// Prepared embedding/gate/coefficient inputs remain explicit until their
// producer graphs are ported. This is the eleven-layer evaluation body only.
struct EvaluationBodyBuffers {
    int tokens;
    const bf16 *x0, *bigram, *pre_gate, *post_gate, *skip_gate;
    const bf16 *residual_gains, *mlp_gains, *mu_last, *mu_post, *mu_groups;
    std::array<const bf16 *, 4> values; // layers 1, 2, 8, 10
    bf16 *initial, *last_pre, *last_aux, *parallel, *post_mix, *output;
    std::array<bf16 *, 11> after_attention, residual;
    std::array<bf16 *, 11> attention_input;
    std::array<bf16 *, 12> mlp_input;
    std::array<AttentionLayerBuffers, 11> attention;
    std::array<MLPEvaluationBuffers, 12> mlp;
};

struct EvaluationBodyPlan {
    std::vector<BF16Matmul> matrices;
    std::vector<QKVTransform> qkv;
    std::vector<Attention> attention;
    std::vector<AttentionPost> post;
    std::vector<AttentionLayerSetup> setup;
    std::vector<std::pair<int, int>> setup_offsets;
    std::vector<ResidualMix> mixes;
    std::vector<ResidualNorm> norms;
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    std::pair<int, int> leaves{0, 0};
    int roots = 0;

    void append(std::vector<Task> part, const std::vector<Group> &deps,
                const std::vector<std::pair<int, int>> &part_stages, int part_roots) {
        int base = int(tasks.size()), group_base = int(groups.size());
        for (auto &task : part)
            for (int &signal : task.signal) if (signal >= 0) signal += group_base;
        for (auto group : deps) {
            group.begin += base; group.end += base; groups.push_back(group);
        }
        if (base) {
            int dependency = int(groups.size());
            groups.push_back({leaves.second - leaves.first, base, base + part_roots});
            for (int i = leaves.first; i < leaves.second; ++i) {
                if (tasks[i].signal[1] != -1) throw std::runtime_error("body dependency slot already used");
                tasks[i].signal[1] = dependency;
            }
        } else roots = part_roots;
        tasks.insert(tasks.end(), part.begin(), part.end());
        for (auto [begin, end] : part_stages) stages.push_back({base + begin, base + end});
        leaves = stages.back();
    }
    void point(TaskKind kind, int id, int tokens) {
        std::vector<Task> part;
        for (int row = 0; row < (tokens + 3) / 4; ++row) part.push_back({id, row, 0, {-1, -1}, 0, 0, kind});
        int count = int(part.size());
        append(std::move(part), {}, {{0, count}}, count);
    }
    void norm(const bf16 *input, bf16 *output, int tokens) {
        int id = int(norms.size());
        norms.push_back({input, nullptr, output, nullptr, tokens});
        point(TaskKind::residual_norm, id, tokens);
    }
    static ResidualTerm fixed(const bf16 *value, float factor = 1.0f) {
        ResidualTerm term{}; term.value = value; term.constant = factor; return term;
    }
    static ResidualTerm scaled(const bf16 *value, const bf16 *coefficient, int stride = 0) {
        auto term = fixed(value, 0); term.coefficient = coefficient; term.coefficient_row_stride = stride; return term;
    }
    void mix(std::initializer_list<ResidualTerm> terms, bf16 *output, int tokens) {
        ResidualMix op{};
        op.tokens = tokens; op.output = output; op.count = int(terms.size());
        std::copy(terms.begin(), terms.end(), op.terms);
        add_mix(op);
    }
    void add_mix(const ResidualMix &op) {
        if (op.count <= 0 || op.count > 11) throw std::runtime_error("invalid model residual source count");
        int id = int(mixes.size()); mixes.push_back(op); point(TaskKind::residual_mix, id, op.tokens);
    }
    void add_attention(const AttentionLayerBuffers &buffers) {
        AttentionLayerPlan plan(buffers);
        if (!plan.evaluation) throw std::runtime_error("evaluation body received training attention");
        int matrix = int(matrices.size()), id = int(attention.size());
        matrices.insert(matrices.end(), plan.bf16_ops.begin(), plan.bf16_ops.end());
        qkv.push_back(plan.qkv); attention.push_back(plan.attention); post.push_back(plan.post);
        setup.push_back(plan.setup(nullptr, nullptr, nullptr)); setup_offsets.push_back({matrix, id});
        for (auto &task : plan.tasks) task.op += task.kind == TaskKind::bf16_matmul ? matrix : id;
        append(std::move(plan.tasks), plan.groups, plan.stages, 1);
    }
    void add_mlp(const MLPEvaluationBuffers &buffers) {
        MLPEvaluationPlan plan(buffers);
        int base = int(matrices.size());
        matrices.insert(matrices.end(), plan.ops.begin(), plan.ops.end());
        for (auto &task : plan.tasks) task.op += base;
        append(std::move(plan.tasks), plan.groups, plan.stages, plan.roots);
    }

    explicit EvaluationBodyPlan(const EvaluationBodyBuffers &b) {
        const int t = b.tokens;
        if (t <= 0 || t % 16) throw std::runtime_error("unsupported evaluation body token count");
        mix({fixed(b.x0), scaled(b.bigram, b.pre_gate + 19, 26)}, b.initial, t);
        const bf16 *x = b.initial;
        const bf16 *shared_norm = nullptr;
        for (int layer = 0; layer < 11; ++layer) {
            if (frontier::qk_width[layer]) {
                auto a = b.attention[layer];
                if (!a.evaluation || a.tokens != t) throw std::runtime_error("body attention mode/shape mismatch");
                const auto role = frontier::attention_role[layer];
                if (a.qk_dim != frontier::qk_width[layer] || a.value_dim != frontier::value_width[layer] ||
                    a.paired != role.paired || bool(a.alpha) != role.xsa || bool(a.gate) != role.head_gate ||
                    (layer != 10 && bool(a.aux) != role.auxiliary))
                    throw std::runtime_error("body attention role mismatch");
                a.x_bf16 = b.attention_input[layer];
                if (layer == 10) a.x_bf16 = shared_norm;
                else {
                    norm(layer == 8 ? b.residual[7] : x, b.attention_input[layer], t);
                    if (layer == 8) shared_norm = a.x_bf16;
                }
                if (layer == 10) {
                    if (!shared_norm) throw std::runtime_error("missing shared layer-8 normalization");
                    auto current = scaled(x, b.mu_last + 5, 14); current.constant = 1;
                    mix({current, scaled(b.initial, b.mu_last + 3, 14), scaled(b.residual[7], b.mu_last + 4, 14)},
                        b.last_pre, t);
                    auto ve = scaled(b.values[3], b.mu_last + 6, 14);
                    ve.groups = 2; ve.coefficient_group_stride = 1;
                    mix({scaled(b.initial, b.mu_last, 14), scaled(b.residual[7], b.mu_last + 1, 14),
                         scaled(x, b.mu_last + 2, 14), ve}, b.last_aux, t);
                    a.aux = b.last_aux;
                    x = b.last_pre;
                }
                add_attention(a);
            }
            if (layer == 6) {
                mix({fixed(x), scaled(b.residual[3], b.skip_gate, 1)}, b.after_attention[layer], t);
            } else if (layer == 10) {
                mix({scaled(x, b.mu_last + 8, 14), scaled(b.attention[layer].output, b.mu_last + 9, 14),
                     scaled(b.initial, b.mu_last + 10, 14), scaled(b.bigram, b.mu_last + 11, 14)},
                    b.after_attention[layer], t);
            } else {
                ResidualMix op{}; op.tokens = t; op.output = b.after_attention[layer];
                op.terms[op.count++] = scaled(x, b.residual_gains + layer * 2);
                if (frontier::qk_width[layer]) op.terms[op.count++] = fixed(b.attention[layer].output);
                int offset = layer < 4 ? 18 + layer * 2 : layer == 4 ? 18 : layer == 5 ? 20 :
                             layer == 7 ? 22 : layer == 8 ? 24 : 26;
                const bf16 *gate = layer < 4 ? b.pre_gate : b.post_gate;
                int stride = layer < 4 ? 26 : 41;
                if (layer != 3 && layer != 8 && layer != 9)
                    op.terms[op.count++] = scaled(b.x0, gate + offset, stride);
                if (layer != 0 && layer != 2 && layer != 3 && layer != 7 && layer != 8)
                    op.terms[op.count++] = scaled(b.bigram, gate + offset + 1, stride);
                add_mix(op);
            }
            x = b.after_attention[layer];
            if (layer == 7) {
                mix({scaled(x, b.residual_gains + 2 * layer + 1)}, b.residual[layer], t);
            } else {
                auto mlp = b.mlp[layer];
                if (mlp.tokens != t) throw std::runtime_error("body MLP shape mismatch");
                mlp.input = b.mlp_input[layer];
                norm(x, b.mlp_input[layer], t);
                add_mlp(mlp);
                if (layer == 10) {
                    mix({scaled(x, b.mu_last + 12, 14), scaled(mlp.output, b.mu_last + 13, 14)}, b.residual[layer], t);
                } else {
                    mix({scaled(x, b.residual_gains + 2 * layer + 1), scaled(mlp.output, b.mlp_gains + layer)},
                        layer == 8 ? b.parallel : b.residual[layer], t);
                }
                if (layer == 8) {
                    auto extra = b.mlp[11]; extra.input = mlp.input;
                    if (extra.tokens != t) throw std::runtime_error("body parallel MLP shape mismatch");
                    add_mlp(extra);
                    mix({fixed(b.parallel), scaled(extra.output, b.mlp_gains + layer)}, b.residual[layer], t);
                }
            }
            x = b.residual[layer];
        }
        const bf16 *sources[] = {b.initial, b.residual[7], b.residual[9], b.values[0], b.residual[3],
                                 b.values[1], b.values[3], b.values[2], b.mlp_input[10], shared_norm};
        ResidualMix final{}; final.tokens = t; final.output = b.post_mix; final.count = 11;
        final.terms[0] = fixed(x);
        for (int i = 0; i < 10; ++i) {
            auto term = scaled(sources[i], b.mu_post + i, 10);
            term.groups = 12; term.group_delta = b.mu_groups + i * 12; term.delta_row_stride = 120;
            final.terms[i + 1] = term;
        }
        add_mix(final);
        norm(b.post_mix, b.output, t);
    }
};

} // namespace nano

#pragma once
#include "megakernel.cuh"
#include "frontier.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace nano {

// Matches the attention-call boundary: normalized activation/weight caches are
// supplied by the model, and compact weight gradients return to its bank views.
struct AttentionLayerBuffers {
    int tokens, qk_dim, value_dim, documents, window;
    bool paired, key_offset;
    float attention_scale;
    const int *sequences;
    const float *scalars;
    const fp8 *x, *x_transposed, *weight, *weight_transposed;
    const bf16 *weight_bf16, *output_weight, *dy, *factor1, *factor2, *aux, *alpha, *gate;
    bf16 *projected, *projected_v, *q, *k, *v, *attention_y, *post_y, *output;
    bf16 *dpost, *attention_dy, *dq, *dk, *dv, *daux, *dalpha, *dgate, *dx;
    bf16 *dw_qkv_unscaled, *dw_qkv, *dw_o_unscaled, *dw_o;
    fp8 *packed_gradient, *packed_gradient_transposed;
    float *lse, *delta, *value_side, *qkv_partial, *o_partial, *gain_gradients;
    float *fp8_raw[4] = {};
    float *bf16_raw[5] = {};
    float *attention_raw[4] = {};
    float *post_raw[4] = {}; // output, dY, dAlpha, dGate; value_side is already FP32
    float *qkv_raw_gradient = nullptr;
    bool evaluation = false;
    const bf16 *x_bf16 = nullptr;
};

struct AttentionLayerPlan {
    std::vector<Matmul> ops;
    std::vector<BF16Matmul> bf16_ops;
    QKVTransform qkv;
    Attention attention;
    AttentionPost post;
    std::vector<ProjectionGradient> gradient;
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    const float *scalars;
    int forwards;
    bool evaluation;

    explicit AttentionLayerPlan(const AttentionLayerBuffers &b)
        : scalars(b.scalars), forwards((b.evaluation ? b.value_dim == 64 : b.paired) ? 2 : 1), evaluation(b.evaluation) {
        constexpr int h = frontier::heads, m = frontier::model_dim;
        if (b.tokens <= 0 || b.tokens % 16 || b.documents <= 0 || b.window < 0 ||
            b.tokens > INT32_MAX / (3 * h * 128) - 1024 ||
            !((b.qk_dim == 64 && (b.value_dim == 64 || b.value_dim == 128)) ||
              (b.qk_dim == 128 && b.value_dim == 128)) ||
            (b.key_offset && (b.paired || b.qk_dim != 128)) || (b.paired && (b.alpha || b.qk_dim != 64)) ||
            !std::isfinite(b.attention_scale) || b.attention_scale <= 0 || (evaluation && !b.x_bf16))
            throw std::runtime_error("unsupported frontier attention layer geometry");
        int t = b.tokens, qk = 2 * h * b.qk_dim, v = h * b.value_dim, features = qk + v;
        bool separate = forwards == 2;
        auto product = [&](const fp8 *a, const fp8 *w, bf16 *out, int rows, int cols, int k,
                           int ar, int ak, int br, int bk) {
            Matmul op{a, w, nullptr, nullptr, out, rows, cols, k, ar, ak, br, bk,
                      0, 1, 1, Epilogue::linear};
            op.raw = b.fp8_raw[ops.size()];
            ops.push_back(op);
        };
        bf16_ops.push_back({b.post_y, b.output_weight, nullptr, b.output, b.bf16_raw[0], t, m, v, v, 1, v, 1});
        if (evaluation) {
            bf16_ops.push_back({b.x_bf16, b.weight_bf16, nullptr, b.projected, b.bf16_raw[3],
                               t, separate ? qk : features, m, m, 1, m, 1});
            if (separate)
                bf16_ops.push_back({b.x_bf16, b.weight_bf16 + int64_t(qk) * m, nullptr, b.projected_v,
                                   b.bf16_raw[4], t, v, m, m, 1, m, 1});
        } else {
            product(b.x, b.weight, b.projected, t, separate ? qk : features, m, m, 1, m, 1);
            if (separate)
                product(b.x, b.weight + int64_t(qk) * m, b.projected_v, t, v, m, m, 1, m, 1);
            product(b.packed_gradient_transposed, b.x_transposed, b.dw_qkv_unscaled,
                    features, m, t, t, 1, t, 1);
            product(b.packed_gradient, b.weight_transposed, b.dx, t, m, features, features, 1, features, 1);
            bf16_ops.push_back({b.dy, b.output_weight, nullptr, b.dpost, b.bf16_raw[1], t, v, m, m, 1, 1, v});
            bf16_ops.push_back({b.dy, b.post_y, nullptr, b.dw_o_unscaled, b.bf16_raw[2], m, v, t, 1, m, 1, v});
        }
        qkv = {b.projected, separate ? b.projected_v : b.projected + qk,
               b.factor1, b.factor2, b.aux, b.q, b.k, b.v, b.dq, b.dk, b.dv,
               b.packed_gradient, b.packed_gradient_transposed, t, h, b.qk_dim, b.value_dim,
               separate ? qk : features, separate ? v : features, 128,
               b.paired, b.key_offset, 0, b.qkv_raw_gradient};
        attention = {b.q, b.k, b.v, b.attention_dy, b.attention_y, b.dq, b.dk, b.dv,
                     b.lse, b.delta, b.sequences, b.paired ? t * 2 : t, b.paired ? h / 2 : h,
                     b.qk_dim, b.value_dim, b.documents, b.window, b.attention_scale,
                     b.attention_raw[0], b.attention_raw[1], b.attention_raw[2], b.attention_raw[3]};
        post = {b.attention_y, b.v, b.dpost, b.alpha, b.gate, b.post_y, b.attention_dy,
                b.dv, b.daux, b.dalpha, b.dgate, b.value_side, t, h, b.value_dim, 128,
                b.post_raw[0], b.post_raw[1], nullptr, b.post_raw[2], b.post_raw[3]};
        if (!evaluation) gradient = {
            {b.weight_bf16, b.dw_qkv_unscaled, b.dw_qkv, b.scalars, b.qkv_partial, b.gain_gradients,
             nullptr, features * m, false},
            {b.output_weight, b.dw_o_unscaled, b.dw_o, b.scalars, b.o_partial, b.gain_gradients + 1,
             b.gain_gradients + 2, m * v, true}
        };
        int previous = 0;
        auto finish = [&] {
            int end = int(tasks.size());
            if (!stages.empty()) {
                auto p = stages.back();
                int group = int(groups.size());
                groups.push_back({p.second - p.first, previous, end});
                for (int i = p.first; i < p.second; ++i) tasks[i].signal[0] = group;
            }
            stages.push_back({previous, end});
            previous = end;
        };
        auto matrix_tasks = [&](int id, bool bf) {
            int rows = bf ? bf16_ops[id].m : ops[id].m, cols = bf ? bf16_ops[id].n : ops[id].n;
            for (int r = 0; r < (rows + 63) / 64; ++r)
                for (int c = 0; c < (cols + 63) / 64; ++c)
                    tasks.push_back({id, r, c, {-1, -1}, 0, 0, bf ? TaskKind::bf16_matmul : TaskKind::matmul});
        };
        auto head_tasks = [&](TaskKind kind, int stage = 0) {
            bool core = kind == TaskKind::attention_forward || kind == TaskKind::attention_dq || kind == TaskKind::attention_dkv;
            int count = core ? attention.tokens : t, heads = core ? attention.heads : h;
            for (int row = 0; row < (count + 3) / 4; ++row)
                for (int head = 0; head < heads; ++head)
                    tasks.push_back({0, row, head, {-1, -1}, stage, 0, kind});
        };
        auto gain_tasks = [&](int id, bool reduce) {
            int count = reduce ? 1 : (gradient[id].elements + projection_elements - 1) / projection_elements;
            for (int i = 0; i < count; ++i)
                tasks.push_back({id, i, int(reduce), {-1, -1}, 0, 0, TaskKind::projection_gradient});
        };
        tasks.push_back({0, 0, 0, {-1, -1}, 0, 0, TaskKind::layer_setup}); finish();
        for (int i = 0; i < forwards; ++i) matrix_tasks(evaluation ? i + 1 : i, evaluation);
        finish();
        head_tasks(TaskKind::qkv_forward); finish();
        head_tasks(TaskKind::attention_forward); finish();
        head_tasks(TaskKind::attention_post, int(AttentionPostStage::forward)); finish();
        matrix_tasks(0, true); finish();
        if (evaluation) return;
        matrix_tasks(1, true); matrix_tasks(2, true); finish();
        head_tasks(TaskKind::attention_post, int(AttentionPostStage::backward)); gain_tasks(1, false); finish();
        head_tasks(TaskKind::attention_dq); gain_tasks(1, true); finish();
        head_tasks(TaskKind::attention_dkv); finish();
        head_tasks(TaskKind::attention_post, int(AttentionPostStage::value_gradient)); finish();
        head_tasks(TaskKind::qkv_backward); finish();
        matrix_tasks(forwards, false); matrix_tasks(forwards + 1, false); finish();
        gain_tasks(0, false); finish();
        gain_tasks(0, true); finish();
    }
    AttentionLayerSetup setup(Matmul *matrices, BF16Matmul *bf16_matrices, QKVTransform *transform) const {
        return {scalars, matrices, bf16_matrices, transform, forwards, evaluation};
    }
};

} // namespace nano

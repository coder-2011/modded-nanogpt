#pragma once
#include "megakernel.cuh"
#include "frontier.cuh"
#include <stdexcept>
#include <vector>

namespace nano {

struct MLPEvaluationBuffers {
    const bf16 *input, *up_weight, *down_weight;
    bf16 *pre, *post, *output;
    float *raw_up, *raw_down;
    int tokens;
};

struct MLPEvaluationPlan {
    std::vector<BF16Matmul> ops;
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    int roots;
    explicit MLPEvaluationPlan(const MLPEvaluationBuffers &b) {
        constexpr int m = frontier::model_dim, h = frontier::mlp_width;
        if (b.tokens <= 0 || b.tokens % 16 || b.tokens > INT32_MAX / h - 64)
            throw std::runtime_error("unsupported frontier evaluation MLP geometry");
        ops = {
            {b.input, b.up_weight, nullptr, b.post, b.raw_up, b.tokens, h, m, m, 1, m, 1},
            {b.post, b.down_weight, nullptr, b.output, b.raw_down, b.tokens, m, h, h, 1, 1, m}
        };
        // Eval preserves the BF16 pre-activation and squared-ReLU boundaries.
        // The post-lambda is applied by the residual site, outside this MLP.
        ops[0].relu_square = true;
        ops[0].pre_activation = b.pre;
        int rows = (b.tokens + 63) / 64, up_columns = (h + 63) / 64, down_columns = (m + 63) / 64;
        roots = rows * up_columns;
        tasks.reserve(size_t(rows) * (up_columns + down_columns));
        for (int r = 0; r < rows; ++r) {
            groups.push_back({up_columns, roots + r * down_columns, roots + (r + 1) * down_columns});
            for (int c = 0; c < up_columns; ++c)
                tasks.push_back({0, r, c, {r, -1}, 0, 0, TaskKind::bf16_matmul});
        }
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < down_columns; ++c)
                tasks.push_back({1, r, c, {-1, -1}, 0, 0, TaskKind::bf16_matmul});
        stages = {{0, roots}, {roots, int(tasks.size())}};
    }
};

} // namespace nano

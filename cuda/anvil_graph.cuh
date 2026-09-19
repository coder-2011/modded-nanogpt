#pragma once
#include "megakernel.cuh"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace nano {

inline constexpr double anvil_maps[6][3] = {
    {3.923798038567, -6.095026865488, 3.905234618423},
    {3.278126713798, -3.328923386476, 0.989127286973},
    {3.505298394150, -5.137358782410, 1.968325560615},
    {2.815058591845, -3.685181239622, 1.417196497642},
    {2.245503932403, -2.443826979899, 0.963091710461},
    {2.256537145403, -2.166840097229, 0.929501253245},
};

// Each matrix has independent stage dependencies; different banks can make progress together.
struct AnvilPlan {
    std::vector<Anvil> matrices;
    std::vector<BF16Matmul> products;
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<int> roots;
    std::vector<std::pair<int, int>> stages;

    explicit AnvilPlan(const std::vector<Anvil> &ops) : matrices(ops) {
        for (int matrix = 0; matrix < int(ops.size()); ++matrix) {
            const auto &op = ops[matrix];
            int64_t short_dim = std::min(op.rows, op.cols);
            if (op.rows <= 0 || op.cols <= 0 ||
                int64_t(op.rows) * op.cols + short_dim * short_dim > INT32_MAX - anvil_elements)
                throw std::runtime_error("invalid ANVIL matrix dimensions");
            int size = op.rows * op.cols, d = std::min(op.rows, op.cols);
            int previous_begin = 0, previous_end = 0;
            auto finish_stage = [&](int begin) {
                int end = int(tasks.size());
                if (previous_end > previous_begin) {
                    int group = int(groups.size());
                    groups.push_back({previous_end - previous_begin, begin, end});
                    for (int i = previous_begin; i < previous_end; ++i)
                        tasks[i].signal[0] = group;
                } else {
                    for (int i = begin; i < end; ++i)
                        roots.push_back(i);
                }
                stages.push_back({begin, end});
                previous_begin = begin;
                previous_end = end;
            };
            auto pointwise = [&](AnvilStage stage, int tiles) {
                int begin = int(tasks.size());
                for (int i = 0; i < tiles; ++i)
                    tasks.push_back({matrix, i, int(stage), {-1, -1}, 0, 0, TaskKind::anvil});
                finish_stage(begin);
            };
            auto multiply = [&](BF16Matmul product) {
                int begin = int(tasks.size()), id = int(products.size());
                products.push_back(product);
                for (int r = 0; r < (product.m + 63) / 64; ++r)
                    for (int c = 0; c < (product.n + 63) / 64; ++c)
                        if (!product.symmetric || r >= c)
                            tasks.push_back({id, r, c, {-1, -1}, 0, 0, TaskKind::bf16_matmul});
                finish_stage(begin);
            };
            auto gram = [&](const __nv_bfloat16 *x) {
                bool tall = op.rows > op.cols;
                int stride = tall ? 1 : op.cols, inner = tall ? op.cols : 1;
                multiply({x, x, nullptr, op.gram, nullptr, d, d, tall ? op.rows : op.cols,
                          stride, inner, stride, inner, 1, 0, 1, false, true});
            };
            pointwise(AnvilStage::velocity, (size + anvil_elements - 1) / anvil_elements);
            gram(op.x);
            pointwise(AnvilStage::trace, 1);
            pointwise(AnvilStage::normalize, (size + d * d + anvil_elements - 1) / anvil_elements);
            auto *x = op.x, *work = op.work;
            for (int i = 0; i < 6; ++i) {
                if (i)
                    gram(x);
                multiply({op.gram, op.gram, op.gram, op.polynomial, nullptr, d, d, d,
                          d, 1, d, 1, float(anvil_maps[i][2]), float(anvil_maps[i][1]), 1, false, true});
                if (op.rows > op.cols)
                    multiply({x, op.polynomial, x, work, nullptr, op.rows, op.cols, d,
                              op.cols, 1, d, 1, 1, float(anvil_maps[i][0]), 1, op.rows > 1024});
                else
                    multiply({op.polynomial, x, x, work, nullptr, op.rows, op.cols, d,
                              d, 1, 1, op.cols, 1, float(anvil_maps[i][0]), 1, op.rows > 1024});
                std::swap(x, work);
            }
            pointwise(AnvilStage::lane_power, std::max(op.rows, op.cols));
            pointwise(AnvilStage::norm_restore, 1);
            pointwise(AnvilStage::update, (size + anvil_elements - 1) / anvil_elements);
        }
        if (roots.empty())
            throw std::runtime_error("empty ANVIL plan");
    }
};

} // namespace nano

#pragma once
#include "megakernel.cuh"
#include "frontier.cuh"
#include <vector>

namespace nano {

struct MLPTrainingBuffers {
    int tokens;
    const fp8 *w1, *w1t, *w2, *w2t;
    fp8 *x, *xt, *g, *gt, *post, *postt, *dpre, *dpret;
    bf16 *output, *dx, *dw1, *dw2;
    float *amax;
    float *raw[6] = {};
};

inline std::vector<Matmul> mlp_training_products(const MLPTrainingBuffers &b) {
    constexpr int c = frontier::model_dim, h = frontier::mlp_width;
    int t = b.tokens;
    std::vector<Matmul> ops{
        {b.x, b.w1, nullptr, b.post, nullptr, t, h, c, c, 1, c, 1, 0, 1, 1, Epilogue::relu_square},
        {b.post, b.w2t, nullptr, nullptr, b.output, t, c, h, h, 1, h, 1, 0, 1, 1, Epilogue::linear},
        {b.g, b.w2, b.post, b.dpre, nullptr, t, h, c, c, 1, c, 1, 0, 1, 1, Epilogue::relu_backward},
        {b.dpre, b.w1t, nullptr, nullptr, b.dx, t, c, h, h, 1, h, 1, 0, 1, 1, Epilogue::linear},
        {b.dpret, b.xt, nullptr, nullptr, b.dw1, h, c, t, t, 1, t, 1, 0, 1, 1, Epilogue::linear},
        {b.postt, b.gt, nullptr, nullptr, b.dw2, h, c, t, t, 1, t, 1, 0, 1, 1, Epilogue::linear}};
    ops[0].quantized_t = b.postt; ops[0].amax = b.amax;
    ops[2].quantized_t = b.dpret; ops[2].amax = b.amax + 1;
    ops[2].a_e5 = ops[2].quantized_e5 = true;
    ops[3].a_e5 = ops[4].a_e5 = ops[5].b_e5 = true;
    // The pinned scaled_mm gradient products all disable fast accumulation.
    ops[3].precise = ops[4].precise = ops[5].precise = true;
    for (int i = 0; i < 6; ++i) ops[i].raw = b.raw[i];
    return ops;
}

} // namespace nano

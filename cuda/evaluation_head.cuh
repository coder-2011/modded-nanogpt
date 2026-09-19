#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nano {

struct EvaluationHead {
    const int32_t *targets;
    float *partial, *target_logit, *loss;
    int tokens, vocabulary;
    __nv_bfloat16 *softcapped = nullptr;
};

__device__ __forceinline__ __nv_bfloat16 evaluation_softcap(float product) {
    float logit = float(__float2bfloat16_rn(product));
    return __float2bfloat16_rn(23.0f / (1.0f + expf(-(logit + 5.0f) / 7.5f)));
}

__device__ __forceinline__ float head_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__device__ __forceinline__ void evaluation_head_partial(const EvaluationHead &op,
        const __nv_bfloat16 *tile, int tile_row, int tile_col) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int tiles = (op.vocabulary + 63) / 64;
    for (int row = warp; row < 64 && tile_row * 64 + row < op.tokens; row += 4) {
        float sum = 0;
#pragma unroll
        for (int col = lane; col < 64; col += 32)
            if (tile_col * 64 + col < op.vocabulary)
                sum += expf(float(tile[row * 64 + col]) - 23.0f);
        sum = head_sum(sum);
        if (!lane) op.partial[int64_t(tile_row * 64 + row) * tiles + tile_col] = sum;
    }
}

__device__ __forceinline__ void evaluation_loss(const EvaluationHead &op, int row) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    int tiles = (op.vocabulary + 63) / 64;
    float sum = 0;
    for (int col = lane; col < tiles; col += 32) sum += op.partial[int64_t(token) * tiles + col];
    sum = head_sum(sum);
    // The BF16 softcap lies in [0,23], so a fixed shift is stable even when all
    // logits saturate low. No vocabulary entry or target is dropped.
    if (!lane) op.loss[token] = (logf(sum) + 23.0f) - op.target_logit[token];
}

} // namespace nano

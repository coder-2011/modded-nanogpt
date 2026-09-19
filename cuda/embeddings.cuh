#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nano {

struct EmbeddingRead {
    const int32_t *token_ids, *cache_rows;
    const __nv_bfloat16 *token_weight, *value_weight, *cache, *signs;
    __nv_bfloat16 *token_output, *ngram_output, *value_output[4];
    int tokens;
};

// Cache rows are already resolved by the sparse row-exchange path. Token ids
// are canonical [0,50304) ids, and all row indices must refer to resident rows.
__device__ __forceinline__ void embedding_read(const EmbeddingRead &op, int row) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    uint32_t cur = uint32_t(op.token_ids[token]);
    uint32_t prev = token >= 1 ? uint32_t(op.token_ids[token - 1]) : 0;
    uint32_t prev2 = token >= 2 ? uint32_t(op.token_ids[token - 2]) : 0;
    uint32_t sign0 = token >= 1 ? ((30011u * prev) ^ (48271u * cur)) & 8191u : 0;
    uint32_t sign1 = token >= 2 ? ((26801u * prev2) ^ (39779u * prev) ^ (58699u * cur)) & 8191u : 0;
    int64_t br = int64_t(op.cache_rows[token]) * 768, tr = int64_t(op.cache_rows[op.tokens + token]) * 768;
    for (int col = lane; col < 768; col += 32) {
        int64_t out = int64_t(token) * 768 + col, in = int64_t(cur) * 768 + col;
        op.token_output[out] = op.token_weight[in];
        float bigram = float(op.cache[br + col]) * float(op.signs[int64_t(sign0) * 768 + col]);
        float trigram = float(op.cache[tr + col]) * float(op.signs[int64_t(sign1) * 768 + col]);
        op.ngram_output[out] = __float2bfloat16_rn(bigram + trigram);
#pragma unroll
        for (int plane = 0; plane < 4; ++plane)
            op.value_output[plane][out] = op.value_weight[int64_t(plane) * 50304 * 768 + in];
    }
}

} // namespace nano

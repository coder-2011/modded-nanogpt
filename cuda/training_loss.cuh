#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nano {

constexpr int loss_tile = 512;
struct TrainingLoss {
    // Vocabulary is even; targets are in range, prefixes are -1 or in range,
    // and predictions is 1..3. Sampled training supplies candidate positions.
    const __nv_fp8_e4m3 *logits;
    const int64_t *targets, *prefix_targets;
    const float *mtp_weights, *parameters; // E5M2 scale, loss scale, prefix weight
    float *partial, *lse, *loss;
    __nv_fp8_e5m2 *gradient;
    int tokens, vocabulary, predictions;
    float *raw_gradient = nullptr;
};

__device__ __forceinline__ float training_sigmoid(__nv_fp8_e4m3 logit) {
    float value = 0.5f + __tanhf((float(logit) * (1.0f / 7.5f) + 5.0f / 7.5f) * 0.5f) * 0.5f;
    return __half2float(__float2half_rn(value));
}

__device__ __forceinline__ float loss_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__device__ __forceinline__ void training_loss_partial(const TrainingLoss &op, int row, int tile_col) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    float sum = 0;
    for (int col = tile_col * loss_tile + lane; col < (tile_col + 1) * loss_tile && col < op.vocabulary; col += 32)
        sum += __expf(23.0f * training_sigmoid(op.logits[int64_t(token) * op.vocabulary + col]) - 23.0f);
    sum = loss_sum(sum);
    if (!lane) op.partial[int64_t(token) * ((op.vocabulary + loss_tile - 1) / loss_tile) + tile_col] = sum;
}

__device__ __forceinline__ void training_loss_reduce(const TrainingLoss &op, int row) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    int columns = (op.vocabulary + loss_tile - 1) / loss_tile;
    float sum = 0;
    for (int col = lane; col < columns; col += 32) sum += op.partial[int64_t(token) * columns + col];
    sum = loss_sum(sum);
    if (!lane) {
        float lse = 23.0f + __logf(sum), loss = 0;
        for (int k = 0; k < op.predictions && token + k < op.tokens; ++k)
            loss += op.mtp_weights[k] * (lse - 23.0f * training_sigmoid(op.logits[int64_t(token) * op.vocabulary + op.targets[token + k]]));
        int64_t prefix = op.prefix_targets[token];
        if (prefix >= 0)
            loss += op.parameters[2] * (lse - 23.0f * training_sigmoid(op.logits[int64_t(token) * op.vocabulary + prefix]));
        op.lse[token] = lse;
        op.loss[token] = loss;
    }
}

__device__ __forceinline__ void training_loss_gradient(const TrainingLoss &op, int row, int tile_col) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    float weight = 0;
    // The pinned kernel includes every MTP weight, even at the final rows.
    for (int k = 0; k < op.predictions; ++k) weight += op.mtp_weights[k];
    int64_t prefix = op.prefix_targets[token];
    if (prefix >= 0) weight += op.parameters[2];
    float scale = op.parameters[1] * (1.0f / 7.5f * 23.0f) * (1.0f / op.parameters[0]);
    for (int col = tile_col * loss_tile + lane * 2; col < (tile_col + 1) * loss_tile && col < op.vocabulary; col += 64) {
        float gradient[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            int64_t index = int64_t(token) * op.vocabulary + col + j;
            float sigmoid = training_sigmoid(op.logits[index]);
            float correction = 0;
            for (int k = 0; k < op.predictions && token + k < op.tokens; ++k)
                if (op.targets[token + k] == col + j) correction += op.mtp_weights[k];
            if (prefix == col + j) correction += op.parameters[2];
            gradient[j] = scale * (weight * __expf(23.0f * sigmoid - op.lse[token]) - correction) * sigmoid * (1.0f - sigmoid);
            if (op.raw_gradient) op.raw_gradient[index] = gradient[j];
        }
        // One lane owns both bytes, including duplicate MTP/prefix targets.
        uint16_t packed;
        asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(gradient[1]), "f"(gradient[0]));
        *reinterpret_cast<uint16_t *>(op.gradient + int64_t(token) * op.vocabulary + col) = packed;
    }
}

} // namespace nano

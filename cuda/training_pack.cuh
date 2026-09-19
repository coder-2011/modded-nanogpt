#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace nano {

struct ActivationPack {
    const __nv_bfloat16 *input;
    __nv_fp8_e4m3 *row, *transposed;
    const float *scale;
    int tokens, columns;
    bool e5 = false;
    const float *raw_input = nullptr;
};

__device__ __forceinline__ void activation_pack(const ActivationPack &op, int row_tile, int col_tile, __nv_fp8_e4m3 *scratch) {
    float inverse = 1.0f / op.scale[0], limit = op.e5 ? 57344.0f : 448.0f;
    for (int i = threadIdx.x; i < 4096; i += 128) {
        int row = row_tile * 64 + i / 64, col = col_tile * 64 + i % 64;
        if (row < op.tokens && col < op.columns) {
            int64_t index = int64_t(row) * op.columns + col;
            float x = op.raw_input ? op.raw_input[index] : float(op.input[index]);
            x = fminf(limit, fmaxf(-limit, x * inverse));
            __nv_fp8_e4m3 value;
            value.__x = op.e5 ? __nv_fp8_e5m2(x).__x : __nv_fp8_e4m3(x).__x;
            op.row[index] = scratch[i] = value;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4096; i += 128) {
        int row = row_tile * 64 + i % 64, col = col_tile * 64 + i / 64;
        if (row < op.tokens && col < op.columns) op.transposed[int64_t(col) * op.tokens + row] = scratch[i % 64 * 64 + i / 64];
    }
    __syncthreads();
}

struct GradientCast {
    const float *input;
    __nv_bfloat16 *output;
    int elements, output_stride;
    int input_stride = 1;
    float scale = 1;
    bool round_input = false;
    const __nv_bfloat16 *rounded_input = nullptr;
};

__device__ __forceinline__ void gradient_cast(const GradientCast &op, int row) {
    int index = row * 128 + threadIdx.x;
    if (index < op.elements) {
        int64_t source = int64_t(index) * op.input_stride;
        float value = op.rounded_input ? float(op.rounded_input[source]) : op.input[source];
        if (op.round_input) value = float(__float2bfloat16_rn(value));
        op.output[int64_t(index) * op.output_stride] = __float2bfloat16_rn(value * op.scale);
    }
}

struct GradientSum {
    const __nv_bfloat16 *input[6];
    __nv_bfloat16 *output;
    int elements, count;
};

__device__ __forceinline__ void gradient_sum(const GradientSum &op, int row) {
    int index = row * 128 + threadIdx.x;
    if (index < op.elements) {
        float sum = 0;
        for (int i = 0; i < op.count; ++i) sum += float(op.input[i][index]);
        op.output[index] = __float2bfloat16_rn(sum);
    }
}

struct NetworkBackward {
    const __nv_bfloat16 *pre, *dh, *dmu;
    __nv_bfloat16 *dpre, *dbias;
    int tokens, coefficients;
    float *raw_dpre = nullptr;
};

__device__ __forceinline__ void network_backward(const NetworkBackward &op, int row, bool bias) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (bias) {
        for (int col = warp; col < op.coefficients; col += 4) {
            float sum = 0;
            for (int t = lane; t < op.tokens; t += 32) sum += float(op.dmu[int64_t(t) * op.coefficients + col]);
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, offset);
            if (!lane) op.dbias[col] = __float2bfloat16_rn(sum);
        }
    } else {
        int token = row * 4 + warp;
        if (token >= op.tokens) return;
        for (int col = lane; col < 64; col += 32) {
            int64_t i = int64_t(token) * 64 + col; float x = float(op.pre[i]);
            float derivative = 0.5f * (1.0f + erff(x * 0.7071067811865475f)) + x * 0.3989422804014327f * expf(-0.5f * x * x);
            float value = float(op.dh[i]) * derivative;
            if (op.raw_dpre) op.raw_dpre[i] = value;
            op.dpre[i] = __float2bfloat16_rn(value);
        }
    }
}

} // namespace nano

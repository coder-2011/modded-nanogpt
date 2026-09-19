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
};

__device__ __forceinline__ void gradient_cast(const GradientCast &op, int row) {
    int index = row * 128 + threadIdx.x;
    if (index < op.elements) op.output[int64_t(index) * op.output_stride] = __float2bfloat16_rn(op.input[index]);
}

} // namespace nano

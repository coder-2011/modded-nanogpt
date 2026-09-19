#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nano {

struct ResidualTerm {
    const __nv_bfloat16 *value;
    const __nv_bfloat16 *coefficient = nullptr, *group_delta = nullptr;
    int coefficient_row_stride = 0, coefficient_group_stride = 0;
    int delta_row_stride = 0, groups = 1;
    float constant = 0;
    __nv_bfloat16 *dvalue = nullptr;
    // Unrounded per-token/group adjoints. Their consumers own shared-source
    // accumulation and the parameter's final BF16 cast.
    float *dcoefficient_rows = nullptr;
};

struct ResidualMix {
    ResidualTerm terms[11];
    int count, tokens;
    __nv_bfloat16 *output;
    const __nv_bfloat16 *dy = nullptr;
    float *raw = nullptr;
};

struct ResidualNorm {
    const __nv_bfloat16 *input, *dy;
    __nv_bfloat16 *output, *dx;
    int tokens;
    float *rstd = nullptr, *raw = nullptr, *raw_dx = nullptr;
};

__device__ __forceinline__ float routing_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset);
    return x;
}

__device__ __forceinline__ float residual_coefficient(const ResidualTerm &term, int token, int group) {
    float c = term.constant;
    if (term.coefficient)
        c += float(term.coefficient[int64_t(token) * term.coefficient_row_stride + group * term.coefficient_group_stride]);
    if (term.group_delta) c += float(term.group_delta[int64_t(token) * term.delta_row_stride + group]);
    return c;
}

// Four warps own four complete 768-channel rows. Descriptors and source buffers
// are immutable throughout a task; outputs cannot alias a source needed later.
__device__ __forceinline__ void residual_mix_forward(const ResidualMix &op, int row) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    for (int channel = lane; channel < 768; channel += 32) {
        int64_t index = int64_t(token) * 768 + channel;
        float sum = 0;
        for (int i = 0; i < op.count; ++i) {
            const auto &term = op.terms[i];
            float c = residual_coefficient(term, token, channel / (768 / term.groups));
            sum = fmaf(c, float(term.value[index]), sum);
        }
        if (op.raw) op.raw[index] = sum;
        op.output[index] = __float2bfloat16_rn(sum);
    }
}

__device__ __forceinline__ void residual_mix_backward(const ResidualMix &op, int row, int source) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    const auto &term = op.terms[source];
    int width = 768 / term.groups;
    for (int group = 0; group < term.groups; ++group) {
        float c = residual_coefficient(term, token, group), sum = 0;
        for (int j = lane; j < width; j += 32) {
            int64_t index = int64_t(token) * 768 + group * width + j;
            float dy = float(op.dy[index]);
            if (term.dvalue) term.dvalue[index] = __float2bfloat16_rn(c * dy);
            sum = fmaf(dy, float(term.value[index]), sum);
        }
        sum = routing_sum(sum);
        if (lane == 0 && term.dcoefficient_rows)
            term.dcoefficient_rows[int64_t(token) * term.groups + group] = sum;
    }
}

__device__ __forceinline__ void residual_norm(const ResidualNorm &op, int row, bool backward) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    int64_t base = int64_t(token) * 768;
    float square = 0, dot = 0;
    for (int channel = lane; channel < 768; channel += 32) {
        float x = float(op.input[base + channel]);
        square = fmaf(x, x, square);
        if (backward) dot = fmaf(x, float(op.dy[base + channel]), dot);
    }
    // The pinned trainer normalizes in FP32 with torch.finfo(float32).eps.
    float rstd = rsqrtf(routing_sum(square) / 768.0f + 0x1p-23f);
    if (op.rstd && lane == 0 && !backward) op.rstd[token] = rstd;
    float correction = backward ? routing_sum(dot) / 768.0f * rstd * rstd : 0;
    for (int channel = lane; channel < 768; channel += 32) {
        int64_t index = base + channel;
        float x = float(op.input[index]);
        float value = backward ? rstd * (float(op.dy[index]) - x * correction) : x * rstd;
        if (backward) {
            if (op.raw_dx) op.raw_dx[index] = value;
            op.dx[index] = __float2bfloat16_rn(value);
        } else {
            if (op.raw) op.raw[index] = value;
            op.output[index] = __float2bfloat16_rn(value);
        }
    }
}

} // namespace nano

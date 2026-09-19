#pragma once
#include "routing.cuh"

namespace nano {

enum class TailStep : int { coefficients, gelu, bias, input_sum };
struct TailBackward {
    const float *coefficient_rows[10];
    __nv_bfloat16 *dmu, *dmug;
    const __nv_bfloat16 *pre, *dh_mu, *dh_group;
    __nv_bfloat16 *dpre, *dbias, *dw2, *dwg;
    const __nv_bfloat16 *dx_direct, *dx_network;
    __nv_bfloat16 *dx;
    int tokens;
    float *raw_dpre = nullptr;
};

__device__ __forceinline__ void tail_backward(const TailBackward &op, int row, TailStep step) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (step == TailStep::bias) {
        for (int col = threadIdx.x / 32; col < 14; col += 4) {
            float sum = 0;
            if (col < 10) for (int t = lane; t < op.tokens; t += 32) sum += float(op.dmu[int64_t(t) * 10 + col]);
            sum = routing_sum(sum);
            if (!lane) op.dbias[col] = __float2bfloat16_rn(sum);
        }
        for (int i = threadIdx.x; i < 4 * 64; i += 128) op.dw2[10 * 64 + i] = __float2bfloat16_rn(0.0f);
        for (int i = threadIdx.x; i < 4 * 12 * 64; i += 128) op.dwg[10 * 12 * 64 + i] = __float2bfloat16_rn(0.0f);
        return;
    }
    if (token >= op.tokens) return;
    if (step == TailStep::coefficients) {
        for (int term = 0; term < 10; ++term) {
            float d = lane < 12 ? float(__float2bfloat16_rn(op.coefficient_rows[term][int64_t(token) * 12 + lane])) : 0;
            if (lane < 12) op.dmug[int64_t(token) * 120 + term * 12 + lane] = __float2bfloat16_rn(d * 0.1f);
            float sum = routing_sum(d);
            if (!lane) op.dmu[int64_t(token) * 10 + term] = __float2bfloat16_rn(float(__float2bfloat16_rn(sum)) * 0.1f);
        }
    } else if (step == TailStep::gelu) {
        for (int col = lane; col < 64; col += 32) {
            int64_t i = int64_t(token) * 64 + col;
            float x = float(op.pre[i]);
            float dh = float(__float2bfloat16_rn(float(op.dh_mu[i]) + float(op.dh_group[i])));
            float derivative = 0.5f * (1.0f + erff(x * 0.7071067811865475f)) + x * 0.3989422804014327f * expf(-0.5f * x * x);
            float value = dh * derivative;
            if (op.raw_dpre) op.raw_dpre[i] = value;
            op.dpre[i] = __float2bfloat16_rn(value);
        }
    } else {
        for (int col = lane; col < 768; col += 32) {
            int64_t i = int64_t(token) * 768 + col;
            op.dx[i] = __float2bfloat16_rn(float(op.dx_direct[i]) + float(op.dx_network[i]));
        }
    }
}

} // namespace nano

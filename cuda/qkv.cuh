#pragma once
#include "attention.cuh"
#include <cuda_fp8.h>

namespace nano {

struct QKVTransform {
    const __nv_bfloat16 *qk_input, *v_input, *factor1, *factor2, *aux_v;
    __nv_bfloat16 *q, *k, *v;
    const __nv_bfloat16 *dq, *dk, *dv;
    __nv_fp8_e4m3 *grad, *grad_transposed;
    int tokens, heads, qk_dim, v_dim;
    int qk_token_stride, v_token_stride, aux_head_stride;
    bool paired, key_offset;
    float grad_scale;
    float *raw_grad = nullptr;
};

template <int D, int V>
__device__ __forceinline__ void qkv_transform_tile(const QKVTransform &a, bool backward,
                                                  int row, int head) {
    const int lane = threadIdx.x % 32, token = row * 4 + threadIdx.x / 32;
    if (token >= a.tokens)
        return;
    const int features = a.heads * (2 * D + V);
    const int64_t fi = int64_t(token) * D * (a.paired ? 2 : 1) +
                       (a.paired ? (head % 2) * D : 0);
    const int64_t oi = (int64_t(token) * a.heads + head) * D;
    // Reinterpreting [T,6,D] as [2*T,3,D] preserves the packed address. Only
    // the rotary-factor parity and document offsets change for paired heads.
    for (int kind = 0; kind < 2; ++kind) {
        const int input_head = head + kind * a.heads;
        const int64_t xi = int64_t(token) * a.qk_token_stride + input_head * D;
        float x[D / 32], normalized[D / 32], grad[D / 32];
        float square = 0.0f;
#pragma unroll
        for (int d = 0; d < D / 32; ++d) {
            x[d] = float(a.qk_input[xi + lane + 32 * d]);
            square = fmaf(x[d], x[d], square);
        }
        float rstd = rsqrtf(attention_sum(square) / D + 1.1920928955078125e-7f);
#pragma unroll
        for (int d = 0; d < D / 32; ++d)
            normalized[d] = x[d] * rstd;

        if (!backward) {
            float previous[D / 32] = {}, previous_square = 0.0f;
            if (a.key_offset && kind == 1 && token > 0) {
#pragma unroll
                for (int d = 0; d < D / 32; ++d) {
                    previous[d] = float(a.qk_input[xi - a.qk_token_stride + lane + d * 32]);
                    previous_square = fmaf(previous[d], previous[d], previous_square);
                }
            }
            float previous_rstd = rsqrtf(attention_sum(previous_square) / D + 1.1920928955078125e-7f);
#pragma unroll
            for (int d = 0; d < D / 32; ++d) {
                int c = lane + d * 32;
                float value = fmaf(float(a.factor1[fi + c]), normalized[d],
                                   float(a.factor2[fi + c]) * __shfl_xor_sync(0xffffffff, normalized[d], 1));
                // The pinned reference shifts globally, including across document
                // boundaries; masking attention must not silently alter this shift.
                if (a.key_offset && kind == 1 && token > 0 && c >= D / 2)
                    value = previous[d] * previous_rstd;
                (kind ? a.k : a.q)[oi + c] = __float2bfloat16_rn(value);
            }
        } else {
            float correction = 0.0f;
#pragma unroll
            for (int d = 0; d < D / 32; ++d) {
                int c = lane + d * 32;
                const auto *incoming = kind ? a.dk : a.dq;
                float g = float(incoming[oi + c]);
                grad[d] = fmaf(float(a.factor1[fi + c]), g,
                               float(a.factor2[fi + (c ^ 1)]) * __shfl_xor_sync(0xffffffff, g, 1));
                if (a.key_offset && kind == 1 && c >= D / 2) {
                    float next = token + 1 < a.tokens ? float(a.dk[oi + a.heads * D + c]) : 0.0f;
                    grad[d] = token == 0 ? g + next : next;
                }
                correction = fmaf(grad[d], normalized[d], correction);
            }
            correction = attention_sum(correction) / D;
#pragma unroll
            for (int d = 0; d < D / 32; ++d) {
                int feature = input_head * D + lane + d * 32;
                float dx = rstd * (grad[d] - normalized[d] * correction);
                auto packed = __nv_fp8_e4m3(float(__float2bfloat16_rn(dx)) / a.grad_scale);
                a.grad[int64_t(token) * features + feature] = packed;
                a.grad_transposed[int64_t(feature) * a.tokens + token] = packed;
                if (a.raw_grad)
                    a.raw_grad[int64_t(token) * features + feature] = dx;
            }
        }
    }
    const int64_t vi = (int64_t(token) * a.heads + head) * V;
#pragma unroll
    for (int d = 0; d < V / 32; ++d) {
        int c = lane + d * 32;
        if (!backward) {
            float value = float(a.v_input[int64_t(token) * a.v_token_stride + head * V + c]);
            if (a.aux_v)
                value += float(a.aux_v[(int64_t(token) * a.heads + head) * a.aux_head_stride + c]);
            a.v[vi + c] = __float2bfloat16_rn(value);
        } else {
            float dx = float(a.dv[vi + c]);
            int feature = 2 * a.heads * D + head * V + c;
            auto packed = __nv_fp8_e4m3(dx / a.grad_scale);
            a.grad[int64_t(token) * features + feature] = packed;
            a.grad_transposed[int64_t(feature) * a.tokens + token] = packed;
            if (a.raw_grad)
                a.raw_grad[int64_t(token) * features + feature] = dx;
        }
    }
}

__device__ __forceinline__ void execute_qkv(const QKVTransform &a, bool backward, int row, int head) {
    if (a.qk_dim == 128)
        qkv_transform_tile<128, 128>(a, backward, row, head);
    else if (a.v_dim == 128)
        qkv_transform_tile<64, 128>(a, backward, row, head);
    else
        qkv_transform_tile<64, 64>(a, backward, row, head);
}

} // namespace nano

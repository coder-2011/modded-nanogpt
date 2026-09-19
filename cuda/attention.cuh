#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace nano {

// Packed [token, head, channel]. Paired heads arrive as [2*T, H/2, channel]
// with doubled document offsets, exactly as at the frontier FA3 boundary.
struct Attention {
    const __nv_bfloat16 *q, *k, *v, *dy;
    __nv_bfloat16 *y, *dq, *dk, *dv;
    float *lse, *delta;
    const int *cu_seqlens;
    int tokens, heads, qk_dim, v_dim, documents, window;
    float scale;
    float *raw_y = nullptr, *raw_dq = nullptr, *raw_dk = nullptr, *raw_dv = nullptr;
};

enum class TaskKind : int {
    matmul, attention_forward, attention_dq, attention_dkv, qkv_forward, qkv_backward, bf16_matmul, anvil,
    layer_setup, attention_post, projection_gradient, residual_mix, residual_norm, gate_transform, embedding_read, evaluation_loss,
    loss_partial, loss_reduce, loss_gradient
};

__device__ __forceinline__ float attention_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset);
    return x;
}

__device__ __forceinline__ int attention_document(const Attention &a, int token) {
    int lo = 0, hi = a.documents;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (a.cu_seqlens[mid] <= token)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// One warp owns a query (forward/dQ) or key (dK/dV). The first implementation
// keeps exclusive gradient ownership and FP32 softmax state; it is the scalar
// correctness baseline for a tensor-core implementation of these same tasks.
template <int D, int V>
__device__ __forceinline__ void attention_tile(const Attention &a, TaskKind kind,
                                              int row, int head) {
    const int lane = threadIdx.x % 32;
    const int token = row * 4 + threadIdx.x / 32;
    if (token >= a.tokens)
        return;
    const int doc = attention_document(a, token);
    const int begin = a.cu_seqlens[doc], end = a.cu_seqlens[doc + 1];
    const int64_t qi = (int64_t(token) * a.heads + head) * D;
    const int64_t vi = (int64_t(token) * a.heads + head) * V;
    const int si = token * a.heads + head;
    float fixed_qk[D / 32], fixed_v[V / 32], grad_qk[D / 32] = {}, grad_v[V / 32] = {};
#pragma unroll
    for (int d = 0; d < D / 32; ++d)
        fixed_qk[d] = float((kind == TaskKind::attention_dkv ? a.k : a.q)[qi + lane + d * 32]);
#pragma unroll
    for (int d = 0; d < V / 32; ++d)
        fixed_v[d] = kind == TaskKind::attention_forward ? 0.0f :
            float((kind == TaskKind::attention_dkv ? a.v : a.dy)[vi + lane + d * 32]);

    if (kind == TaskKind::attention_forward) {
        float maximum = -CUDART_INF_F, sum = 0.0f;
        for (int key = max(begin, token - a.window); key <= token; ++key) {
            const int64_t ki = (int64_t(key) * a.heads + head) * D;
            const int64_t kv = (int64_t(key) * a.heads + head) * V;
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < D / 32; ++d)
                dot = fmaf(fixed_qk[d], float(a.k[ki + lane + d * 32]), dot);
            float score = attention_sum(dot) * a.scale;
            float next = fmaxf(maximum, score);
            float old = expf(maximum - next), weight = expf(score - next);
            sum = sum * old + weight;
#pragma unroll
            for (int d = 0; d < V / 32; ++d)
                fixed_v[d] = fmaf(weight, float(a.v[kv + lane + d * 32]), fixed_v[d] * old);
            maximum = next;
        }
#pragma unroll
        for (int d = 0; d < V / 32; ++d) {
            int64_t index = vi + lane + d * 32;
            float value = fixed_v[d] / sum;
            auto rounded = __float2bfloat16_rn(value);
            a.y[index] = rounded;
            if (a.raw_y)
                a.raw_y[index] = value;
        }
        if (lane == 0) {
            a.lse[si] = maximum + logf(sum);
        }
        return;
    }

    const bool key_owner = kind == TaskKind::attention_dkv;
    float query_delta = 0.0f;
    if (!key_owner) {
#pragma unroll
        for (int d = 0; d < V / 32; ++d)
            query_delta = fmaf(fixed_v[d], float(a.y[vi + lane + d * 32]), query_delta);
        query_delta = attention_sum(query_delta);
        // FA3 forms delta from the saved BF16 output, not an unrounded copy.
        if (lane == 0)
            a.delta[si] = query_delta;
    }
    const int first = key_owner ? token : max(begin, token - a.window);
    const int last = key_owner ? min(end - 1, token + a.window) : token;
    for (int other = first; other <= last; ++other) {
        const int64_t oi = (int64_t(other) * a.heads + head) * D;
        const int64_t ov = (int64_t(other) * a.heads + head) * V;
        const int os = (key_owner ? other : token) * a.heads + head;
        float moving_qk[D / 32], moving_v[V / 32];
        float dot = 0.0f, dp = 0.0f;
#pragma unroll
        for (int d = 0; d < D / 32; ++d) {
            moving_qk[d] = float((key_owner ? a.q : a.k)[oi + lane + d * 32]);
            dot = fmaf(fixed_qk[d], moving_qk[d], dot);
        }
#pragma unroll
        for (int d = 0; d < V / 32; ++d) {
            moving_v[d] = float((key_owner ? a.dy : a.v)[ov + lane + d * 32]);
            dp = fmaf(fixed_v[d], moving_v[d], dp);
        }
        float p = expf(attention_sum(dot) * a.scale - a.lse[os]);
        float ds = p * (attention_sum(dp) - (key_owner ? a.delta[os] : query_delta)) * a.scale;
#pragma unroll
        for (int d = 0; d < D / 32; ++d)
            grad_qk[d] = fmaf(ds, moving_qk[d], grad_qk[d]);
        if (key_owner) {
#pragma unroll
            for (int d = 0; d < V / 32; ++d)
                grad_v[d] = fmaf(p, moving_v[d], grad_v[d]);
        }
    }
#pragma unroll
    for (int d = 0; d < D / 32; ++d) {
        int64_t index = qi + lane + d * 32;
        (key_owner ? a.dk : a.dq)[index] = __float2bfloat16_rn(grad_qk[d]);
        float *raw = key_owner ? a.raw_dk : a.raw_dq;
        if (raw)
            raw[index] = grad_qk[d];
    }
    if (key_owner) {
#pragma unroll
        for (int d = 0; d < V / 32; ++d) {
            int64_t index = vi + lane + d * 32;
            a.dv[index] = __float2bfloat16_rn(grad_v[d]);
            if (a.raw_dv)
                a.raw_dv[index] = grad_v[d];
        }
    }
}

__device__ __forceinline__ void execute_attention(const Attention &a, TaskKind kind,
                                                 int row, int head) {
    if (a.qk_dim == 128)
        attention_tile<128, 128>(a, kind, row, head);
    else if (a.v_dim == 128)
        attention_tile<64, 128>(a, kind, row, head);
    else
        attention_tile<64, 64>(a, kind, row, head);
}

} // namespace nano

#pragma once
#include "attention.cuh"
#include "anvil.cuh"

namespace nano {

struct AttentionPost {
    const __nv_bfloat16 *y, *v, *dy, *alpha, *gate;
    __nv_bfloat16 *output, *attention_dy, *attention_dv, *aux_dv, *dalpha, *dgate;
    float *value_side_gradient;
    int tokens, heads, value_dim, aux_stride;
    float *raw_output = nullptr, *raw_dy = nullptr, *raw_value_side = nullptr;
    float *raw_dalpha = nullptr, *raw_dgate = nullptr;
};

enum class AttentionPostStage : int { forward, backward, value_gradient };

template <int V>
__device__ __forceinline__ void attention_post_tile(const AttentionPost &op, AttentionPostStage stage,
                                                    int row, int head) {
    int lane = threadIdx.x % 32, token = row * 4 + threadIdx.x / 32;
    if (token >= op.tokens)
        return;
    int scalar = token * op.heads + head;
    int64_t base = int64_t(scalar) * V;
    if (stage == AttentionPostStage::value_gradient) {
#pragma unroll
        for (int j = 0; j < V / 32; ++j) {
            int d = lane + j * 32;
            auto gradient = __float2bfloat16_rn(float(op.attention_dv[base + d]) + op.value_side_gradient[base + d]);
            op.attention_dv[base + d] = gradient;
            if (op.aux_dv)
                op.aux_dv[int64_t(scalar) * op.aux_stride + d] = gradient;
        }
        if (op.aux_dv)
            for (int d = V + lane; d < op.aux_stride; d += 32)
                op.aux_dv[int64_t(scalar) * op.aux_stride + d] = __float2bfloat16_rn(0.0f);
        return;
    }
    float y[V / 32], v[V / 32], dot = 0, square = 0;
#pragma unroll
    for (int j = 0; j < V / 32; ++j) {
        y[j] = float(op.y[base + lane + j * 32]);
        v[j] = float(op.v[base + lane + j * 32]);
        dot = fmaf(y[j], v[j], dot);
        square = fmaf(v[j], v[j], square);
    }
    dot = attention_sum(dot);
    square = attention_sum(square);
    float denominator = square < 1e-8f ? 1e-8f : square;
    float alpha = op.alpha ? tanhf(float(op.alpha[scalar])) : 0.0f;
    float gate = op.gate ? float(op.gate[scalar]) : 1.0f;
    float ratio = dot / denominator;
    if (stage == AttentionPostStage::forward) {
#pragma unroll
        for (int j = 0; j < V / 32; ++j) {
            int64_t i = base + lane + j * 32;
            float z = op.alpha ? fmaf(-alpha * ratio, v[j], y[j]) : y[j];
            float out = z * gate;
            op.output[i] = __float2bfloat16_rn(out);
            if (op.raw_output) op.raw_output[i] = out;
        }
        return;
    }
    float g[V / 32], gv = 0, dg = 0;
#pragma unroll
    for (int j = 0; j < V / 32; ++j) {
        float incoming = float(op.dy[base + lane + j * 32]);
        float z = op.alpha ? fmaf(-alpha * ratio, v[j], y[j]) : y[j];
        g[j] = incoming * gate;
        gv = fmaf(g[j], v[j], gv);
        dg = fmaf(incoming, z, dg);
    }
    gv = attention_sum(gv);
    dg = attention_sum(dg);
    float dot_gradient = op.alpha ? -alpha * gv / denominator : 0.0f;
    // clamp_min differentiates as identity at equality and zero below the floor.
    float square_gradient = op.alpha && square >= 1e-8f ? alpha * gv * dot / (denominator * denominator) : 0.0f;
#pragma unroll
    for (int j = 0; j < V / 32; ++j) {
        int64_t i = base + lane + j * 32;
        float dy = fmaf(dot_gradient, v[j], g[j]);
        float dv = op.alpha ? fmaf(2.0f * square_gradient, v[j],
                                  fmaf(dot_gradient, y[j], -alpha * ratio * g[j])) : 0.0f;
        op.attention_dy[i] = __float2bfloat16_rn(dy);
        // Join the side path with FA's BF16 dV before the final BF16 cast.
        op.value_side_gradient[i] = dv;
        if (op.raw_dy) op.raw_dy[i] = dy;
        if (op.raw_value_side) op.raw_value_side[i] = dv;
    }
    if (lane == 0) {
        float da = op.alpha ? -ratio * gv * (1.0f - alpha * alpha) : 0.0f;
        if (op.dalpha) op.dalpha[scalar] = __float2bfloat16_rn(da);
        if (op.dgate) op.dgate[scalar] = __float2bfloat16_rn(op.gate ? dg : 0.0f);
        if (op.raw_dalpha) op.raw_dalpha[scalar] = da;
        if (op.raw_dgate) op.raw_dgate[scalar] = op.gate ? dg : 0.0f;
    }
}

__device__ __forceinline__ void execute_attention_post(const AttentionPost &op, AttentionPostStage stage,
                                                        int row, int head) {
    if (op.value_dim == 64)
        attention_post_tile<64>(op, stage, row, head);
    else
        attention_post_tile<128>(op, stage, row, head);
}

// scalars: x scale, packed weight scale, gradient scale, QKV gain, O gain, extra O gain.
struct ProjectionGradient {
    const __nv_bfloat16 *weight, *unscaled_gradient;
    __nv_bfloat16 *gradient;
    const float *scalars;
    float *partial, *gain_gradient, *extra_gain_gradient;
    int elements;
    bool output_projection;
    const __nv_bfloat16 *fold_gain = nullptr;
    __nv_bfloat16 *fold_gradient = nullptr;
};
constexpr int projection_elements = 1024;

__device__ __forceinline__ void projection_gradient_tile(const ProjectionGradient &op, bool reduce,
                                                         int tile, float *scratch) {
    float sum = 0;
    if (!reduce) {
        float gain = op.fold_gain ? float(*op.fold_gain) :
            op.output_projection ? op.scalars[4] * op.scalars[5] : op.scalars[3];
        int begin = tile * projection_elements;
        for (int i = begin + threadIdx.x; i < min(begin + projection_elements, op.elements); i += 128) {
            float gradient = float(op.unscaled_gradient[i]);
            sum = fmaf(gradient, float(op.weight[i]), sum);
            op.gradient[i] = __float2bfloat16_rn(gradient * gain);
        }
        sum = anvil_sum(sum, scratch);
        if (threadIdx.x == 0) op.partial[tile] = sum;
    } else {
        int parts = (op.elements + projection_elements - 1) / projection_elements;
        for (int i = threadIdx.x; i < parts; i += 128) sum += op.partial[i];
        sum = anvil_sum(sum, scratch);
        if (threadIdx.x == 0) {
            *op.gain_gradient = sum * (op.output_projection ? op.scalars[5] : 1.0f);
            if (op.fold_gradient) *op.fold_gradient = __float2bfloat16_rn(sum);
            if (op.extra_gain_gradient)
                *op.extra_gain_gradient = sum * op.scalars[4];
        }
    }
}

} // namespace nano

#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nano {

enum class GateKind : int { gelu, affine, copy, skip, auxiliary_input, auxiliary_output, smear };
struct GateTransform {
    GateKind kind;
    const __nv_bfloat16 *input, *bias, *value;
    __nv_bfloat16 *output;
    int tokens, columns, input_stride;
    const float *scalar = nullptr;
    float scale = 1;
    bool round_scale = false;
    float *raw = nullptr;
};

__device__ __forceinline__ void gate_transform(const GateTransform &op, int row) {
    int token = row * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (token >= op.tokens) return;
    float scale = op.scalar ? *op.scalar : op.scale;
    if (op.round_scale) scale = float(__float2bfloat16_rn(scale));
    for (int col = lane; col < op.columns; col += 32) {
        float x;
        if (op.kind == GateKind::smear) {
            x = float(op.value[int64_t(token) * 768 + col]);
            if (token) x += (scale / (1.0f + expf(-float(op.input[token])))) * float(op.value[int64_t(token - 1) * 768 + col]);
        } else if (op.kind == GateKind::auxiliary_input) {
            x = float(col < 6 ? op.input[int64_t(token) * 768 + col] : op.value[int64_t(token) * 768 + col - 6]);
        } else if (op.kind == GateKind::auxiliary_output) {
            float gate = float(op.input[int64_t(token) * 6 + col / 128]);
            x = (2.0f / (1.0f + expf(-gate))) * float(op.value[int64_t(token) * 768 + col]);
        } else {
            x = float(op.input[int64_t(token) * op.input_stride + col]);
            if (op.kind == GateKind::gelu) x = 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
            else if (op.kind == GateKind::affine) x = (x + (op.bias ? float(op.bias[col]) : 0.0f)) * scale;
            else if (op.kind == GateKind::skip) x = x / (1.0f + expf(-scale));
        }
        int64_t index = int64_t(token) * op.columns + col;
        if (op.raw) op.raw[index] = x;
        op.output[index] = __float2bfloat16_rn(x);
    }
}

struct GateNetwork {
    const __nv_bfloat16 *up_weight, *down_weight, *bias;
    __nv_bfloat16 *pre, *hidden, *product;
};
struct AuxiliaryGate {
    const __nv_bfloat16 *weight;
    __nv_bfloat16 *input, *product, *output;
};

} // namespace nano

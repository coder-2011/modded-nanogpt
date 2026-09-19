#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nano {

struct AnvilScalars {
    float momentum, decay, fast_beta, fast_weight, lr;
};

// One rank-local matrix. Scalars remain at stable device addresses across steps.
struct Anvil {
    const __nv_bfloat16 *gradient;
    float *fast, *slow;
    __nv_bfloat16 *x, *work, *gram, *polynomial;
    float *energy, *power, *gain, *norm; // norm[0]: cascade divisor; norm[1]: equalizer correction
    uint16_t *parameter, *mantissa;
    const AnvilScalars *scalars;
    int rows, cols;
    float energy_weight = 0.1f;
};

enum class AnvilStage : int { velocity, trace, normalize, lane_power, norm_restore, update };
constexpr int anvil_elements = 1024;

__device__ __forceinline__ float anvil_lerp(float start, float end, float weight) {
    // PyTorch 2.10's compiled lerp decomposition switches base at |weight| >= 0.5.
    bool high = fabsf(weight) >= 0.5f;
    return fmaf(high ? weight - 1.0f : weight, end - start, high ? end : start);
}

__device__ __forceinline__ float anvil_sum(float value, float *scratch) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (threadIdx.x % 32 == 0)
        scratch[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x < 4 ? scratch[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
    }
    // A caller may immediately reuse the scratch for a second reduction.
    __syncthreads();
    return value;
}

__device__ __forceinline__ void anvil_tile(const Anvil &op, AnvilStage stage, int tile,
                                           float *scratch) {
    int count = op.rows * op.cols, short_dim = min(op.rows, op.cols);
    int lanes = max(op.rows, op.cols);
    int begin = tile * anvil_elements, end = min(begin + anvil_elements, count);
    if (stage == AnvilStage::velocity) {
        const AnvilScalars s = *op.scalars;
        for (int i = begin + threadIdx.x; i < end; i += 128) {
            float grad = float(op.gradient[i]);
            float fast = anvil_lerp(op.fast[i], grad, 1.0f - s.fast_beta);
            float slow = anvil_lerp(op.slow[i], grad, 0.02f);
            op.fast[i] = fast;
            op.slow[i] = slow;
            float mix = fmaf(s.fast_weight, fast, (1.0f - s.fast_weight) * slow);
            op.x[i] = __float2bfloat16_rn(anvil_lerp(grad, mix, s.momentum));
        }
    } else if (stage == AnvilStage::trace) {
        float value = 0;
        for (int i = threadIdx.x; i < short_dim; i += 128)
            value += float(op.gram[i * short_dim + i]);
        value = anvil_sum(value, scratch);
        if (threadIdx.x == 0)
            op.norm[0] = fmaf(sqrtf(value), 1.05f, 1e-6f);
    } else if (stage == AnvilStage::normalize) {
        float d = op.norm[0];
        int total = count + short_dim * short_dim;
        for (int i = begin + threadIdx.x; i < min(begin + anvil_elements, total); i += 128) {
            if (i < count)
                op.x[i] = __float2bfloat16_rn(float(op.x[i]) / d);
            else
                op.gram[i - count] = __float2bfloat16_rn(float(op.gram[i - count]) / (d * d));
        }
    } else if (stage == AnvilStage::lane_power) {
        float value = 0;
        for (int j = threadIdx.x; j < short_dim; j += 128) {
            int i = op.rows >= op.cols ? tile * op.cols + j : j * op.cols + tile;
            float x = float(op.x[i]);
            value += x * x;
        }
        value = anvil_sum(value, scratch) / float(short_dim);
        if (threadIdx.x == 0) {
            float energy = anvil_lerp(op.energy[tile], value, op.energy_weight);
            op.power[tile] = value;
            op.energy[tile] = energy;
            op.gain[tile] = rsqrtf(energy < 1e-10f ? 1e-10f : energy);
        }
    } else if (stage == AnvilStage::norm_restore) {
        float pre = 0, post = 0;
        for (int i = threadIdx.x; i < lanes; i += 128) {
            pre += op.power[i];
            float gain = op.gain[i];
            post += (op.power[i] * short_dim) * (gain * gain);
        }
        pre = anvil_sum(pre, scratch);
        post = anvil_sum(post, scratch);
        if (threadIdx.x == 0) {
            float post_norm = sqrtf(post);
            op.norm[1] = sqrtf(pre * short_dim) / (post_norm < 1e-10f ? 1e-10f : post_norm);
        }
    } else {
        const AnvilScalars s = *op.scalars;
        for (int i = begin + threadIdx.x; i < end; i += 128) {
            int lane = op.rows >= op.cols ? i / op.cols : i % op.cols;
            float scale = float(__float2bfloat16_rn(op.gain[lane] * op.norm[1]));
            float grad = float(__float2bfloat16_rn(float(op.x[i]) * scale));
            op.x[i] = __float2bfloat16_rn(grad);
            float shadow = __uint_as_float((uint32_t(op.parameter[i]) << 16) | op.mantissa[i]);
            float aligned = op.slow[i] * shadow >= 0.0f ? 1.0f : 0.0f;
            float decay = (shadow * aligned) * s.decay;
            shadow = fmaf(-grad, s.lr, fmaf(-decay, s.lr, shadow));
            uint32_t bits = __float_as_uint(shadow);
            // The BF16 parameter stores the high bits, not a rounded FP32 shadow.
            op.parameter[i] = uint16_t(bits >> 16);
            op.mantissa[i] = uint16_t(bits);
        }
    }
}

} // namespace nano

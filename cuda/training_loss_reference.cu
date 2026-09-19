// CE_KERNEL_SOURCE from PR #360, c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7.
// Only extern "C" becomes a vocabulary template; arithmetic is unchanged.
// Upstream MIT license: ../LICENSE. Original source SHA256:
// 71728386ec63079ba65348e85165070c91a8ae5501b505d9cb469a62ff65168b
#include "training_loss.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>
#include "bench.cuh"
namespace frontier_loss_reference {
constexpr int BLOCK_SIZE = 256;


#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define __nv_fp8_e5m2 char
#define uint16_t unsigned short
#define uint8_t unsigned char
#define int64_t long long

struct __align__(16) __half8 {
    __half data[8];
    __device__ __half& operator[](int i) { return data[i]; }
    __device__ const __half& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

__device__ __forceinline__ __nv_fp8_e5m2 f32_to_fp8_e5m2_fast(float x) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(x));
    __nv_fp8_e5m2 result;
    *reinterpret_cast<uint8_t*>(&result) = (uint8_t)(packed & 0xFFu);
    return result;
}

__device__ __forceinline__ uint16_t f32x2_to_fp8_e5m2x2(float lo, float hi) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(hi), "f"(lo));
    return packed;
}

__device__ __forceinline__ unsigned int fp8_e5m2x2_pair_to_word(uint16_t a, uint16_t b) {
    unsigned int w;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(w) : "r"((unsigned int)a), "r"((unsigned int)b));
    return w;
}

struct __align__(8) __nv_uchar8 {
    unsigned char data[8];
    __device__ unsigned char& operator[](int i) { return data[i]; }
    __device__ const unsigned char& operator[](int i) const { return data[i]; }
};

__device__ __forceinline__ float ce_e4m3_logit_to_f32(unsigned char b) {
    unsigned int h2;
    unsigned short in = (unsigned short)b;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
    unsigned short lo = (unsigned short)(h2 & 0xFFFFu);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(lo));
    return f;
}

#define CE_IN_T unsigned char
#define CE_IN8_T __nv_uchar8
#define CE_LOGIT_TO_F32(v) ce_e4m3_logit_to_f32(v)

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }

//__device__ float sigmoid(float x) {
//  return 1.0f / (1.0f + __expf(-x));
//}
__device__ float sigmoid(float x) {
  return 0.5f + __tanhf(x * 0.5f) * 0.5f;
}

template <int VOCAB_SIZE>
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void ce_fwd_bwd_kernel(
    const CE_IN_T* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ mtp_weights,
    const int64_t* __restrict__ prefix_targets,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double A_param,
    double B_param,
    double C_param,
    double grad_s_param,
    double grad_scale_param,
    double prefix_weight_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float A = (float)A_param;
  float B = (float)B_param;
  float C = (float)C_param;
  float grad_s = (float)grad_s_param;
  float grad_scale = (float)grad_scale_param;

  extern __shared__ __half smem[];

  static_assert(VEC_WIDTH == 8);

  const CE_IN_T *block_logit_ptr = logits + VOCAB_SIZE * blockIdx.x;

  float inv_C = 1 / C;
  float B_div_C = B * inv_C;
  // FIXED-MAX lse. z = A*sigmoid((l+B)/C) with A=23 is bounded to (0, A], so exp(z - A) is in (exp(-A), 1] and the block sum in [VOCAB*exp(-A),
  // VOCAB] -- 5e-6 .. 5e4 at VOCAB=50304, nowhere near fp32's range. The online block max exists only to bound that exponent, so the constant A
  // stands in for it, the exp-sum rides the SAME smem pass, and the __syncthreads below is the only one before smem is read cross-thread.

  float thread_sum = 0.0f;

  #pragma unroll 25
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      CE_IN8_T result = *(CE_IN8_T*)(&block_logit_ptr[idx]);
      __half8 result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = CE_LOGIT_TO_F32(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);
        result_sigmoid[k] = __float2half(tmp);
      }
      *(__half8*)(&smem[idx]) = result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = A * __half2float(result_sigmoid[k]);
        thread_sum += __expf(tmp - A);
      }
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_sums[NUM_WARPS];

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

  if (threadIdx.x % 32 == 0) {
    block_sums[warp_id] = thread_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }

  float lse = A + __logf(block_sum);

  // Prefix token prediction target for this position (T' = longest-prefix token of
  // the immediate next-token target T). prefix_targets[i] < 0 => no valid prefix, ignored.
  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __half2float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }
    }
    // Same CE logic as MTP, but the target is the prefix token T' at this position.
    {
      int64_t ptgt = prefix_targets[blockIdx.x];
      if (ptgt >= 0 && ptgt < VOCAB_SIZE) {
        float z_p = A * __half2float(smem[ptgt]);
        total_loss += (float)prefix_weight_param * (lse - z_p);
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  // Total weight over active predictions at this position (used in the softmax-normalizer
  // gradient term). Include the prefix prediction only when it has a valid target.
  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }
  int64_t ptgt_row = prefix_targets[blockIdx.x];
  if (ptgt_row >= 0) {
    S_w += (float)prefix_weight_param;
  }

  #pragma unroll 4
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_fp8_e5m28 result;

    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __half8 sigmoid_us = *(__half8*)(&smem[idx]);
      uint16_t pk[VEC_WIDTH / 2];
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j += 2) {
        float g[2];
        #pragma unroll
        for (int t = 0; t < 2; t++) {
          float sigmoid_u = __half2float(sigmoid_us[j + t]);
          float z = A * sigmoid_u;
          float p = __expf(z - lse);

          float term1 = S_w * p;
          float term2 = 0.0f;

          float grad_z = term1 - term2;
          g[t] = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
        }
        pk[j >> 1] = f32x2_to_fp8_e5m2x2(g[0], g[1]);
      }
      unsigned long long packed8 =
          ((unsigned long long)fp8_e5m2x2_pair_to_word(pk[2], pk[3]) << 32)
          | (unsigned long long)fp8_e5m2x2_pair_to_word(pk[0], pk[1]);
      *(unsigned long long*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = packed8;
    }
  }

  __syncthreads();

  // Sparse correction for target columns. Threads [0, n_predict) handle the future MTP
  // targets; thread n_predict handles the prefix target. term2 for a column sums the
  // weights of every prediction (MTP + prefix) whose target lands on that column, so
  // duplicate columns across threads write identical values (idempotent, race-free).
  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && ((float)prefix_weight_param > 0.0f);
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx) {
    int i = threadIdx.x;
    int64_t target = is_pfx ? ptgt_row : targets[blockIdx.x + i];

    float sigmoid_u = __half2float(smem[target]);
    float z = A * sigmoid_u;
    float p = __expf(z - lse);

    float term1 = S_w * p;
    float term2 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size && k < n_predict) {
        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }
      }
    }
    if (ptgt_row >= 0 && ptgt_row == target) {
      term2 += (float)prefix_weight_param;
    }

    float grad_z = term1 - term2;
    float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
    auto result_tmp = f32_to_fp8_e5m2_fast(grad_x);
    auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;
  }
}
#undef __nv_fp8_e5m2
#undef uint16_t
#undef uint8_t
#undef int64_t
#undef CE_IN_T
#undef CE_IN8_T
#undef CE_LOGIT_TO_F32
} // namespace frontier_loss_reference

template <int Vocabulary>
void reference_loss(const nano::TrainingLoss &op, const float *parameters) {
    auto kernel = frontier_loss_reference::ce_fwd_bwd_kernel<Vocabulary>;
    static bool configured = false;
    if (!configured) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Vocabulary * 2));
        configured = true;
    }
    kernel<<<op.tokens, 256, Vocabulary * 2>>>(reinterpret_cast<const unsigned char *>(op.logits),
        reinterpret_cast<const long long *>(op.targets), op.mtp_weights,
        reinterpret_cast<const long long *>(op.prefix_targets), op.loss,
        reinterpret_cast<char *>(op.gradient), op.tokens, op.predictions,
        23.0, 5.0, 7.5, parameters[0], parameters[1], parameters[2]);
    CHECK_CUDA(cudaGetLastError());
}

void launch_reference_loss(const nano::TrainingLoss &op, const float *parameters) {
    switch (op.vocabulary) {
        case 2048: reference_loss<2048>(op, parameters); break;
        case 10240: reference_loss<10240>(op, parameters); break;
        case 14336: reference_loss<14336>(op, parameters); break;
        case 24576: reference_loss<24576>(op, parameters); break;
        case 50304: reference_loss<50304>(op, parameters); break;
        default: throw std::runtime_error("unsupported pinned loss reference vocabulary");
    }
}

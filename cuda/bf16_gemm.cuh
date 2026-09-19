#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "evaluation_head.cuh"

namespace nano {

// A[M,K], B[N,K], C[M,N]. Input strides are in elements. Output may alias C,
// but must not overlap A or B while other tiles are still reading them.
struct BF16Matmul {
    const __nv_bfloat16 *a, *b, *c;
    __nv_bfloat16 *output;
    float *raw;
    int m, n, k;
    int64_t ar, ak, br, bk;
    float alpha = 1.0f, beta = 0.0f, b_scale = 1.0f;
    bool round_product = false;
    bool symmetric = false; // Schedule only lower-triangular tiles, mirror each owned element.
    bool relu_square = false;
    __nv_bfloat16 *pre_activation = nullptr;
    const EvaluationHead *head = nullptr;
};

__device__ __forceinline__ void mma_bf16(float *d, const uint32_t *a, const uint32_t *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int bf16_k_tile = 64, bf16_operand_elements = 64 * bf16_k_tile;
constexpr int bf16_scratch_bytes = 4 * bf16_operand_elements * sizeof(__nv_bfloat16);

__device__ __forceinline__ int bf16_operand_index(int row, int k) {
    return row * bf16_k_tile + (k ^ ((row & 7) * 8));
}

#ifdef NANO_BF16_INDEPENDENT_LOAD
__device__ __forceinline__ void load_bf16_operand(const __nv_bfloat16 *input, __nv_bfloat16 *shared,
        int rows, int inner, int row0, int k0, int64_t row_stride, int64_t inner_stride) {
    if (inner_stride == 1 && inner % 8 == 0 && row_stride % 8 == 0 && (reinterpret_cast<uintptr_t>(input) & 15) == 0) {
        for (int i = threadIdx.x; i < bf16_operand_elements / 8; i += 128) {
            int row = i / 8, k = i % 8 * 8;
            const auto *source = input + int64_t(row0 + row < rows ? row0 + row : 0) * row_stride + (k0 + k < inner ? k0 + k : 0);
            unsigned destination = __cvta_generic_to_shared(shared + bf16_operand_index(row, k));
            int bytes = row0 + row < rows && k0 + k < inner ? 16 : 0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                         :: "r"(destination), "l"(source), "r"(bytes) : "memory");
        }
    } else {
        for (int i = threadIdx.x; i < bf16_operand_elements; i += 128) {
            int row = i / 64, k = i % 64;
#ifdef NANO_BF16_COALESCED
            if (row_stride == 1) { row = i % 64; k = i / 64; }
#endif
            shared[bf16_operand_index(row, k)] = row0 + row < rows && k0 + k < inner ?
                input[int64_t(row0 + row) * row_stride + int64_t(k0 + k) * inner_stride] : __float2bfloat16_rn(0.0f);
        }
    }
}
#endif

__device__ __forceinline__ void load_bf16_operands(const BF16Matmul &op, int row0, int col0,
                                                 int k0, __nv_bfloat16 *sa, __nv_bfloat16 *sb) {
    bool paired = op.ak == 1 && op.bk == 1 && op.k % 8 == 0 && op.ar % 8 == 0 && op.br % 8 == 0;
#ifdef NANO_BF16_INDEPENDENT_LOAD
    paired = paired && ((reinterpret_cast<uintptr_t>(op.a) | reinterpret_cast<uintptr_t>(op.b)) & 15) == 0;
#endif
    if (paired) {
        for (int i = threadIdx.x; i < bf16_operand_elements / 8; i += 128) {
            int r = i / (bf16_k_tile / 8), k = k0 + i % (bf16_k_tile / 8) * 8;
            const auto *ap = op.a + int64_t(row0 + r < op.m ? row0 + r : 0) * op.ar + (k < op.k ? k : 0);
            const auto *bp = op.b + int64_t(col0 + r < op.n ? col0 + r : 0) * op.br + (k < op.k ? k : 0);
            unsigned as = __cvta_generic_to_shared(sa + bf16_operand_index(r, k - k0));
            unsigned bs = __cvta_generic_to_shared(sb + bf16_operand_index(r, k - k0));
            int av = row0 + r < op.m && k < op.k ? 16 : 0;
            int bv = col0 + r < op.n && k < op.k ? 16 : 0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(as), "l"(ap), "r"(av) : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(bs), "l"(bp), "r"(bv) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    } else {
#ifdef NANO_BF16_INDEPENDENT_LOAD
        // A transposed operand must not force its contiguous partner onto scalar loads.
        load_bf16_operand(op.a, sa, op.m, op.k, row0, k0, op.ar, op.ak);
        load_bf16_operand(op.b, sb, op.n, op.k, col0, k0, op.br, op.bk);
        asm volatile("cp.async.commit_group;" ::: "memory");
#else
        for (int i = threadIdx.x; i < bf16_operand_elements; i += 128) {
            int ar = i / bf16_k_tile, ak = i % bf16_k_tile;
            int br = ar, bk = ak;
#ifdef NANO_BF16_COALESCED
            if (op.ar == 1) {
                ar = i % 64;
                ak = i / 64;
            }
            if (op.br == 1) {
                br = i % 64;
                bk = i / 64;
            }
#endif
            sa[bf16_operand_index(ar, ak)] = row0 + ar < op.m && k0 + ak < op.k ?
                op.a[int64_t(row0 + ar) * op.ar + int64_t(k0 + ak) * op.ak] : __float2bfloat16_rn(0.0f);
            sb[bf16_operand_index(br, bk)] = col0 + br < op.n && k0 + bk < op.k ?
                op.b[int64_t(col0 + br) * op.br + int64_t(k0 + bk) * op.bk] : __float2bfloat16_rn(0.0f);
        }
#endif
    }
}

__device__ __forceinline__ void bf16_matmul_tile(const BF16Matmul &op, int row, int col,
                                                void *storage) {
    auto *scratch = static_cast<__nv_bfloat16 *>(storage);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int wm = warp / 2 * 32, wn = warp % 2 * 32;
    const int row0 = row * 64, col0 = col * 64;
    float acc[2][4][4] = {};
    load_bf16_operands(op, row0, col0, 0, scratch, scratch + bf16_operand_elements);
    for (int k0 = 0; k0 < op.k; k0 += bf16_k_tile) {
        auto *sa = scratch + (k0 / bf16_k_tile % 2) * 2 * bf16_operand_elements;
        auto *sb = sa + bf16_operand_elements;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (k0 + bf16_k_tile < op.k) {
            auto *next = scratch + ((k0 / bf16_k_tile + 1) % 2) * 2 * bf16_operand_elements;
            load_bf16_operands(op, row0, col0, k0 + bf16_k_tile, next, next + bf16_operand_elements);
        }
        if (op.b_scale != 1.0f) {
            // The O projection scales and rounds its BF16 weights before GEMM.
            for (int i = threadIdx.x; i < bf16_operand_elements; i += 128)
                sb[i] = __float2bfloat16_rn(float(sb[i]) * op.b_scale);
            __syncthreads();
        }
#pragma unroll 1
        for (int ki = 0; ki < bf16_k_tile && k0 + ki < op.k; ki += 16) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                uint32_t a[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int r = wm + mi * 16 + lane / 4 + j % 2 * 8;
                    int k = ki + lane % 4 * 2 + j / 2 * 8;
                    a[j] = *reinterpret_cast<const uint32_t *>(sa + bf16_operand_index(r, k));
                }
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    uint32_t b[2];
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int r = wn + ni * 8 + lane / 4, k = ki + lane % 4 * 2 + j * 8;
                        b[j] = *reinterpret_cast<const uint32_t *>(sb + bf16_operand_index(r, k));
                    }
                    mma_bf16(acc[mi][ni], a, b);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int r = row0 + wm + mi * 16 + lane / 4 + j / 2 * 8;
                int c = col0 + wn + ni * 8 + lane % 4 * 2 + j % 2;
                if (r < op.m && c < op.n && (!op.symmetric || r >= c)) {
                    int64_t index = int64_t(r) * op.n + c;
                    float x = op.alpha * acc[mi][ni][j];
                    // ANVIL's split matmul/add path has an intervening BF16 cast.
                    if (op.round_product)
                        x = float(__float2bfloat16_rn(x));
                    if (op.beta != 0.0f)
                        x = fmaf(op.beta, float(op.c[index]), x);
                    if (op.raw)
                        op.raw[index] = x;
                    float result = x;
                    if (op.relu_square) {
                        auto pre = __float2bfloat16_rn(x);
                        if (op.pre_activation) op.pre_activation[index] = pre;
                        float positive = float(pre);
                        positive = positive < 0.0f ? 0.0f : positive;
                        result = positive * positive;
                    }
                    if (op.output)
                        op.output[index] = __float2bfloat16_rn(result);
                    if (op.head) {
                        auto capped = evaluation_softcap(x);
                        scratch[(r - row0) * 64 + c - col0] = capped;
                        if (op.head->softcapped) op.head->softcapped[index] = capped;
                        if (op.head->targets[r] == c) op.head->target_logit[r] = float(capped);
                    }
                    if (op.symmetric && r != c) {
                        int64_t mirror = int64_t(c) * op.n + r;
                        if (op.raw)
                            op.raw[mirror] = x;
                        if (op.relu_square && op.pre_activation)
                            op.pre_activation[mirror] = __float2bfloat16_rn(x);
                        if (op.output)
                            op.output[mirror] = __float2bfloat16_rn(result);
                    }
                }
            }
    if (op.head) {
        __syncthreads();
        evaluation_head_partial(*op.head, scratch, row, col);
    }
}

} // namespace nano

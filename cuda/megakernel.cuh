#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "attention.cuh"
#include "qkv.cuh"
#include "bf16_gemm.cuh"
#include "anvil.cuh"
#include "attention_post.cuh"
#include "routing.cuh"
#include "gates.cuh"
#include "embeddings.cuh"
#include "training_loss.cuh"
#include "tail_backward.cuh"
#include "training_pack.cuh"

#ifndef NANO_IDLE_NS
#define NANO_IDLE_NS 1024
#endif

namespace nano {

#ifndef NANO_ANVIL_IDLE_NS
#define NANO_ANVIL_IDLE_NS 1024
#endif

constexpr int tile = 64;
constexpr int threads = 128;
#ifndef NANO_K_TILE
#define NANO_K_TILE 128
#endif
constexpr int k_tile = NANO_K_TILE;
constexpr int operand_bytes = tile * k_tile;
constexpr int stage_bytes = 2 * operand_bytes;
constexpr int scratch_bytes = 2 * stage_bytes;
#ifndef NANO_TRANSPOSE_PAD
#define NANO_TRANSPOSE_PAD 0
#endif
constexpr int transpose_stride = tile + NANO_TRANSPOSE_PAD;
static_assert(tile * transpose_stride <= scratch_bytes);
static_assert(k_tile == 32 || k_tile == 64 || k_tile == 128);
using fp8 = __nv_fp8_e4m3;
using bf16 = __nv_bfloat16;

// FP8 buffers carry raw bytes; gradient descriptors select E5M2 interpretation.
__host__ __device__ inline fp8 encode_fp8(float value, bool e5 = false) {
    if (!e5)
        return fp8(value);
    fp8 out;
    out.__x = __nv_fp8_e5m2(value).__x;
    return out;
}

__host__ __device__ inline float decode_fp8(fp8 value, bool e5 = false) {
    if (!e5)
        return float(value);
    __nv_fp8_e5m2 out;
    out.__x = value.__x;
    return float(out);
}

enum class Epilogue : int { linear, relu_square, relu_backward };

// A is logically M x K, B is logically N x K. Strides describe both layouts.
struct Matmul {
    const fp8 *a, *b, *post;
    fp8 *quantized;
    bf16 *output;
    int m, n, k, ar, ak, br, bk;
    float scale, output_scale, post_scale;
    Epilogue epilogue;
    fp8 *quantized_t = nullptr;
    float *raw = nullptr;
    float *accumulation = nullptr;
    bool a_e5 = false, b_e5 = false, quantized_e5 = false;
    bool precise = false;
    float *amax = nullptr;
};

struct MLPSetup {
    Matmul *ops;
    const float *scales; // input, up weight, down weight, incoming gradient, post, dpre
    float *amax;
    const bf16 *fold_p = nullptr;
};

__device__ __forceinline__ void mlp_setup(const MLPSetup &op) {
    if (threadIdx.x) return;
    const float *s = op.scales;
    op.ops[0].scale = s[0] * s[1]; op.ops[0].output_scale = s[4];
    op.ops[1].scale = s[4] * s[2];
    op.ops[2].scale = s[3] * s[2]; op.ops[2].output_scale = s[5]; op.ops[2].post_scale = s[4];
    if (op.fold_p) {
        float p = float(*op.fold_p);
        op.ops[1].scale = s[4] * (s[2] * p);
        op.ops[2].scale = (s[2] * s[3]) * p;
    }
    op.ops[3].scale = s[5] * s[1];
    op.ops[4].scale = s[5] * s[0]; op.ops[5].scale = s[4] * s[3];
    op.amax[0] = op.amax[1] = 0;
}

struct HeadSetup {
    Matmul *ops;
    const float *scales, *loss_parameters;
};

__device__ __forceinline__ void head_setup(const HeadSetup &op) {
    if (!threadIdx.x) {
        op.ops[0].scale = op.scales[0] * op.scales[1];
        op.ops[1].scale = op.loss_parameters[0] * op.scales[1];
        op.ops[2].scale = op.scales[0] * op.loss_parameters[0];
    }
}

struct Task {
    int op, row, col;
    int signal[2];
    int k_begin = 0, k_end = 0;
    TaskKind kind = TaskKind::matmul;
};

struct AttentionLayerSetup {
    const float *scalars;
    Matmul *ops;
    BF16Matmul *bf16_ops;
    QKVTransform *qkv;
    int forwards;
    bool evaluation = false;
};

__device__ __forceinline__ void setup_attention_layer(const AttentionLayerSetup &op) {
    if (threadIdx.x == 0) {
        const float *s = op.scalars;
        if (op.evaluation) {
            op.bf16_ops[0].b_scale = s[4] * s[5];
            for (int i = 0; i < op.forwards; ++i) op.bf16_ops[i + 1].b_scale = s[3];
            return;
        }
        float scaled_weight = s[1] * s[3];
        for (int i = 0; i < op.forwards; ++i) op.ops[i].scale = s[0] * scaled_weight;
        op.ops[op.forwards].scale = s[2] * s[0];
        op.ops[op.forwards + 1].scale = s[2] * scaled_weight;
        op.bf16_ops[0].b_scale = s[4] * s[5];
        op.bf16_ops[1].b_scale = s[4] * s[5];
        op.qkv->grad_scale = s[2];
    }
}

struct Group {
    int expected, begin, end;
};

struct Graph {
    const Matmul *ops;
    const Task *tasks;
    const Group *groups;
    int *counters;
    int *queue;
    int *state; // ready head, ready tail, unfinished, root head
    int task_count;
    int root_count;
    int *audit = nullptr; // task visits; MLP-only completed roots and early-gradient count
    const Attention *attention = nullptr;
    const QKVTransform *qkv = nullptr;
    const BF16Matmul *bf16_ops = nullptr;
    const Anvil *anvil = nullptr;
    const AttentionLayerSetup *layer_setup = nullptr;
    const AttentionPost *attention_post = nullptr;
    const ProjectionGradient *projection_gradient = nullptr;
    const ResidualMix *residual_mix = nullptr;
    const ResidualNorm *residual_norm = nullptr;
    const GateTransform *gates = nullptr;
    const EmbeddingRead *embeddings = nullptr;
    const EvaluationHead *evaluation_heads = nullptr;
    const TrainingLoss *training_losses = nullptr;
    const HeadInput *head_inputs = nullptr;
    const HeadSetup *head_setups = nullptr;
    const TailBackward *tail_backward_ops = nullptr;
    const ActivationPack *activation_packs = nullptr;
    const MLPSetup *mlp_setups = nullptr;
    const GradientCast *gradient_casts = nullptr;
    const GradientSum *gradient_sums = nullptr;
    const NetworkBackward *network_backwards = nullptr;
};

__device__ __forceinline__ int acquire(const int *p) {
    int value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ __forceinline__ void release(int *p, int value) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(value) : "memory");
}

__device__ __forceinline__ int add_acq_rel(int *p, int value) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(p), "r"(value)
                 : "memory");
    return old;
}

__device__ __forceinline__ void push(Graph g, int task) {
    int slot = atomicAdd(g.state + 1, 1);
    release(g.queue + slot, task + 1);
}

__device__ __forceinline__ int pop(Graph g, int &ticket) {
#ifdef NANO_FIFO
    const int head = atomicAdd(g.state, 1);
    if (head >= g.task_count)
        return -2;
#else
    // Keep one ready-work reservation while executing independent roots. No CAS retry loop.
    if (ticket < 0)
        ticket = atomicAdd(g.state, 1);
    if (ticket < g.task_count) {
        int value = acquire(g.queue + ticket);
        if (value) {
            ticket = -1;
            return value - 1;
        }
    }
    if (acquire(g.state + 3) < g.root_count) {
        int root = atomicAdd(g.state + 3, 1);
        if (root < g.root_count)
            return g.queue[root] - 1;
    }
    return ticket >= g.task_count ? -2 : -1;
#endif
#ifdef NANO_FIFO
    int value;
    // A producer reserves a slot before publishing it; it never waits for a consumer.
    while (!(value = acquire(g.queue + head)))
        __nanosleep(32);
    return value - 1;
#endif
}

__device__ __forceinline__ void signal(Graph g, int id) {
    if (id < 0)
        return;
    const Group group = g.groups[id];
    // Acq_rel chains every producer's stores through the final arrival.
    if (add_acq_rel(g.counters + id, 1) + 1 == group.expected) {
#ifdef NANO_FIFO
        for (int task = group.begin; task < group.end; ++task)
            push(g, task);
#else
        int slot = atomicAdd(g.state + 1, group.end - group.begin);
        for (int task = group.begin; task < group.end; ++task)
            release(g.queue + slot++, task + 1);
#endif
    }
}

#ifdef NANO_COOP_PUBLISH
__device__ __forceinline__ void prepare_publication(Graph g, int id, int *range) {
    range[1] = 0;
    if (id < 0) return;
    const Group group = g.groups[id];
    if (add_acq_rel(g.counters + id, 1) + 1 == group.expected) {
        range[0] = group.begin;
        range[1] = group.end - group.begin;
        range[2] = atomicAdd(g.state + 1, range[1]);
    }
}
#endif

__device__ __forceinline__ void mma(float *d, const uint32_t *a, const uint32_t *b,
                                    bool a_e5 = false, bool b_e5 = false) {
#ifdef NANO_FRONTIER
    if (a_e5) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        return;
    }
    if (b_e5) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        return;
    }
#endif
#ifdef NANO_DIRECT_ACCUM
    uint32_t ah[8], bh[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint16_t lo = uint16_t(a[j]), hi = uint16_t(a[j] >> 16);
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(ah[j * 2]) : "h"(lo));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(ah[j * 2 + 1]) : "h"(hi));
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        uint16_t lo = uint16_t(b[j]), hi = uint16_t(b[j] >> 16);
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(bh[j * 2]) : "h"(lo));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(bh[j * 2 + 1]) : "h"(hi));
    }
    // Match the compiler's low-pair/high-pair decomposition while accumulating in HMMA.
#pragma unroll
    for (int part = 0; part < 2; ++part)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(ah[part]), "r"(ah[part + 2]), "r"(ah[part + 4]), "r"(ah[part + 6]),
                       "r"(bh[part]), "r"(bh[part + 2]));
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ int operand_index(int row, int k) {
    // Consecutive lane quartets access distinct shared-memory banks, with 16-byte copy alignment.
    return row * k_tile + (k ^ ((row / (128 / k_tile) & (k_tile / 16 - 1)) * 16));
}

template <bool Full = false>
__device__ __forceinline__ void load_operands(const Matmul &op, int row0, int col0, int k0,
                                              int k_end, fp8 *sa, fp8 *sb) {
    if (Full ||
        (op.ak == 1 && op.bk == 1 && op.k % 32 == 0 && op.ar % 16 == 0 && op.br % 16 == 0)) {
#pragma unroll
        for (int i = threadIdx.x; i < operand_bytes / 16; i += threads) {
            int r = i / (k_tile / 16), k = k0 + i % (k_tile / 16) * 16;
            const fp8 *ap = op.a + (Full || row0 + r < op.m ? row0 + r : 0) * op.ar +
                            (Full || k < k_end ? k : 0);
            const fp8 *bp = op.b + (Full || col0 + r < op.n ? col0 + r : 0) * op.br +
                            (Full || k < k_end ? k : 0);
            unsigned as = __cvta_generic_to_shared(sa + operand_index(r, k - k0));
            unsigned bs = __cvta_generic_to_shared(sb + operand_index(r, k - k0));
            int av = Full || (row0 + r < op.m && k < k_end) ? 16 : 0;
            int bv = Full || (col0 + r < op.n && k < k_end) ? 16 : 0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(as), "l"(ap), "r"(av)
                         : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(bs), "l"(bp), "r"(bv)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    } else {
        for (int i = threadIdx.x; i < operand_bytes; i += threads) {
            const int r = i / k_tile, k = k0 + i % k_tile;
            int index = operand_index(r, i % k_tile);
            sa[index].__x =
                row0 + r < op.m && k < k_end ? op.a[(row0 + r) * op.ar + k * op.ak].__x : 0;
            sb[index].__x =
                col0 + r < op.n && k < k_end ? op.b[(col0 + r) * op.br + k * op.bk].__x : 0;
        }
    }
}

template <bool Full = false, bool Precise = false>
__device__ __forceinline__ void matmul_tile(const Matmul &op, const Task &task, fp8 *scratch) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
    const int row0 = task.row * tile, col0 = task.col * tile;
    float acc[2][4][4] = {};
    const int k_end = task.k_end ? task.k_end : op.k;
    if (task.k_begin) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int r = row0 + wm + mi * 16 + lane / 4 + j / 2 * 8;
                    int c = col0 + wn + ni * 8 + lane % 4 * 2 + j % 2;
                    if (Full || (r < op.m && c < op.n))
                        acc[mi][ni][j] = op.accumulation[r * op.n + c];
                }
    }
#ifndef NANO_SERIAL
    load_operands<Full>(op, row0, col0, task.k_begin, k_end, scratch, scratch + operand_bytes);
#endif
    for (int k0 = task.k_begin; k0 < k_end; k0 += k_tile) {
#ifdef NANO_SERIAL
        fp8 *sa = scratch;
        load_operands<Full>(op, row0, col0, k0, k_end, sa, sa + operand_bytes);
#else
        fp8 *sa = scratch + ((k0 - task.k_begin) / k_tile % 2) * stage_bytes;
#endif
        fp8 *sb = sa + operand_bytes;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
#ifndef NANO_SERIAL
        // The previous iteration's reader barrier released the other stage.
        if (k0 + k_tile < k_end) {
            fp8 *next = scratch + (((k0 - task.k_begin) / k_tile + 1) % 2) * stage_bytes;
            load_operands<Full>(op, row0, col0, k0 + k_tile, k_end, next, next + operand_bytes);
        }
#endif
#ifdef NANO_K_LOOP
#pragma unroll 1
#else
#pragma unroll
#endif
        for (int ki = 0; ki < k_tile; ki += 32) {
            if (!Full && k0 + ki >= k_end)
                break;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                uint32_t a[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int r = wm + mi * 16 + lane / 4 + (j % 2) * 8;
                    int k = ki + lane % 4 * 4 + (j / 2) * 16;
                    a[j] = *reinterpret_cast<const uint32_t *>(sa + operand_index(r, k));
                }
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    uint32_t b[2];
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int r = wn + ni * 8 + lane / 4;
                        int k = ki + lane % 4 * 4 + j * 16;
                        b[j] = *reinterpret_cast<const uint32_t *>(sb + operand_index(r, k));
                    }
                    if constexpr (Precise) {
                        // Scaled-mm gradient products retain cross-instruction accumulation in FP32.
                        float product[4] = {};
                        mma(product, a, b, op.a_e5, op.b_e5);
#pragma unroll
                        for (int j = 0; j < 4; ++j) acc[mi][ni][j] += product[j];
                    } else mma(acc[mi][ni], a, b, op.a_e5, op.b_e5);
                }
            }
        }
        __syncthreads();
    }
    float maximum = 0;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = row0 + wm + mi * 16 + lane / 4 + j / 2 * 8;
                const int c = col0 + wn + ni * 8 + lane % 4 * 2 + j % 2;
                if (Full || (r < op.m && c < op.n)) {
                    const int index = r * op.n + c;
                    if (k_end < op.k) {
                        op.accumulation[index] = acc[mi][ni][j];
                        continue;
                    }
                    float x = acc[mi][ni][j] * op.scale;
                    if (op.raw)
                        op.raw[index] = x;
                    if (op.epilogue == Epilogue::relu_square) {
                        x = fmaxf(round_bf16(x), 0.0f);
                        x = round_bf16(x * x);
                    } else if (op.epilogue == Epilogue::relu_backward) {
                        float relu = sqrtf(float(op.post[index]) * op.post_scale);
                        if (op.quantized_e5)
                            x = round_bf16(2.0f * x * relu);
                        else
                            x = round_bf16(2.0f * round_bf16(x) * round_bf16(relu));
                    }
                    if (op.output)
                        op.output[index] = __float2bfloat16_rn(x);
                    if (op.amax) maximum = fmaxf(maximum, fabsf(x));
#ifdef NANO_REUSE_QUANT
                    if (op.quantized || op.quantized_t) {
                        const fp8 q = encode_fp8(x / op.output_scale, op.quantized_e5);
                        if (op.quantized)
                            op.quantized[index] = q;
                        if (op.quantized_t)
                            scratch[(r - row0) * transpose_stride + c - col0] = q;
                    }
#else
                    if (op.quantized)
                        op.quantized[index] = encode_fp8(x / op.output_scale, op.quantized_e5);
                    if (op.quantized_t)
                        scratch[(r - row0) * transpose_stride + c - col0] =
                            encode_fp8(x / op.output_scale, op.quantized_e5);
#endif
                }
            }
        }
    }
    if (op.amax) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        if (!lane) atomicMax(reinterpret_cast<int *>(op.amax), __float_as_int(maximum));
    }
    if (op.quantized_t) {
        __syncthreads();
        for (int i = threadIdx.x; i < tile * tile; i += threads) {
            int r = row0 + i % tile, c = col0 + i / tile;
            if (Full || (r < op.m && c < op.n))
                op.quantized_t[c * op.m + r] = scratch[i % tile * transpose_stride + i / tile];
        }
        __syncthreads();
    }
}

} // namespace nano

#include "hopper.cuh"

namespace nano {

template <bool Full = false, bool WithPrecise = false>
__device__ __forceinline__ void execute_tile(const Matmul &op, const Task &task, fp8 *scratch) {
    if constexpr (WithPrecise) {
        if (op.precise) { matmul_tile<Full, true>(op, task, scratch); return; }
    }
#ifdef NANO_WGMMA
    hopper_tile(op, task, scratch);
#else
    matmul_tile<Full>(op, task, scratch);
#endif
}

template <bool Audited = false, bool Full = false, bool WithAttention = false, bool WithBF16 = false,
          bool WithAnvil = false, bool WithProjection = false, bool WithRouting = false, bool WithLoss = false>
__global__
#ifdef NANO_MIN_BLOCKS
__launch_bounds__(threads, NANO_MIN_BLOCKS)
#else
__launch_bounds__(threads)
#endif
    void megakernel(Graph g) {
    static_assert(!WithAnvil || WithBF16, "ANVIL requires BF16 matrix tasks");
    static_assert(!WithProjection || (WithBF16 && WithAttention), "attention projections require BF16 and attention tasks");
    __shared__ __align__(1024) fp8 scratch[WithBF16 ? bf16_scratch_bytes : scratch_bytes];
    __shared__ int next, ticket;
#ifdef NANO_COOP_PUBLISH
    __shared__ int publication[6];
#endif
    if (threadIdx.x == 0)
        ticket = -1;
    for (;;) {
        if (threadIdx.x == 0) {
            next = pop(g, ticket);
        }
        __syncthreads();
        const int task_id = next;
        // Idle iterations also need a reader barrier before lane 0 reuses next.
        __syncthreads();
        if (task_id == -2)
            return;
        if (task_id == -1) {
            __nanosleep(WithAnvil ? NANO_ANVIL_IDLE_NS : NANO_IDLE_NS);
            continue;
        }
        const Task task = g.tasks[task_id];
        if (Audited && threadIdx.x == 0) {
            atomicAdd(g.audit + task_id, 1);
            if (!WithAttention && !WithBF16 && !WithAnvil && !WithProjection && task.op >= 4 &&
                acquire(g.audit + g.task_count) < g.root_count)
                atomicAdd(g.audit + g.task_count + 1, 1);
        }
        if (WithLoss && WithRouting && task.kind == TaskKind::activation_pack) {
            activation_pack(g.activation_packs[task.op], task.row, task.col, scratch);
        } else if (WithLoss && WithRouting && task.kind == TaskKind::mlp_setup) {
            mlp_setup(g.mlp_setups[task.op]);
        } else if (WithLoss && WithRouting && task.kind == TaskKind::gradient_cast) {
            gradient_cast(g.gradient_casts[task.op], task.row);
        } else if (WithLoss && WithRouting && task.kind == TaskKind::gradient_sum) {
            gradient_sum(g.gradient_sums[task.op], task.row);
        } else if (WithLoss && WithRouting && task.kind == TaskKind::network_backward) {
            network_backward(g.network_backwards[task.op], task.row, task.col != 0);
        } else if (WithLoss && WithRouting && task.kind == TaskKind::tail_backward) {
            tail_backward(g.tail_backward_ops[task.op], task.row, TailStep(task.col));
        } else if (WithLoss && task.kind == TaskKind::head_setup) {
            head_setup(g.head_setups[task.op]);
        } else if (WithLoss && task.kind == TaskKind::head_input) {
            head_input(g.head_inputs[task.op], task.row, task.col, scratch);
        } else if (WithLoss && task.kind == TaskKind::loss_partial) {
            training_loss_partial(g.training_losses[task.op], task.row, task.col);
        } else if (WithLoss && task.kind == TaskKind::loss_reduce) {
            training_loss_reduce(g.training_losses[task.op], task.row);
        } else if (WithLoss && task.kind == TaskKind::loss_gradient) {
            training_loss_gradient(g.training_losses[task.op], task.row, task.col);
        } else if (WithRouting && task.kind == TaskKind::evaluation_loss) {
            evaluation_loss(g.evaluation_heads[task.op], task.row);
        } else if (WithRouting && task.kind == TaskKind::embedding_read) {
            embedding_read(g.embeddings[task.op], task.row);
        } else if (WithRouting && task.kind == TaskKind::gate_transform) {
            gate_transform(g.gates[task.op], task.row);
        } else if (WithRouting && task.kind == TaskKind::residual_mix) {
            const auto &op = g.residual_mix[task.op];
            if (task.k_begin) residual_mix_backward(op, task.row, task.col);
            else residual_mix_forward(op, task.row);
        } else if (WithRouting && task.kind == TaskKind::residual_norm) {
            residual_norm(g.residual_norm[task.op], task.row, task.k_begin != 0);
        } else if (WithProjection && task.kind == TaskKind::layer_setup) {
            const AttentionLayerSetup op = g.layer_setup[task.op];
            setup_attention_layer(op);
        } else if (WithProjection && task.kind == TaskKind::attention_post) {
            const AttentionPost op = g.attention_post[task.op];
            execute_attention_post(op, AttentionPostStage(task.k_begin), task.row, task.col);
        } else if (WithProjection && task.kind == TaskKind::projection_gradient) {
            const ProjectionGradient op = g.projection_gradient[task.op];
            projection_gradient_tile(op, task.col != 0, task.row, reinterpret_cast<float *>(scratch));
        } else if (WithAnvil && task.kind == TaskKind::anvil) {
            const Anvil op = g.anvil[task.op];
            anvil_tile(op, AnvilStage(task.col), task.row, reinterpret_cast<float *>(scratch));
        } else if (WithBF16 && task.kind == TaskKind::bf16_matmul) {
            const BF16Matmul op = g.bf16_ops[task.op];
            bf16_matmul_tile(op, task.row, task.col, scratch);
        } else if constexpr (WithAttention) {
            if (task.kind == TaskKind::qkv_forward || task.kind == TaskKind::qkv_backward) {
                const QKVTransform op = g.qkv[task.op];
                execute_qkv(op, task.kind == TaskKind::qkv_backward, task.row, task.col);
            } else if (task.kind != TaskKind::matmul) {
                const Attention op = g.attention[task.op];
                execute_attention(op, task.kind, task.row, task.col);
            } else {
                const Matmul op = g.ops[task.op];
                execute_tile<Full, WithLoss>(op, task, scratch);
            }
        } else {
            const Matmul op = g.ops[task.op];
            execute_tile<Full, WithLoss>(op, task, scratch);
        }
        // All output-writing threads publish before the controller signals readiness.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            if (Audited && !WithAttention && !WithBF16 && !WithAnvil && !WithProjection && task.op == 0)
                add_acq_rel(g.audit + g.task_count, 1);
            // Task descriptors are immutable. Reload only on the controller so
            // every worker lane need not retain both signals across the math.
            const volatile Task *finished = g.tasks + task_id;
#ifdef NANO_COOP_PUBLISH
            prepare_publication(g, finished->signal[0], publication);
            prepare_publication(g, finished->signal[1], publication + 3);
#else
            signal(g, finished->signal[0]);
            signal(g, finished->signal[1]);
#endif
            add_acq_rel(g.state + 2, -1);
        }
        __syncthreads();
#ifdef NANO_COOP_PUBLISH
        // The controller acquires all producer stores; the block barrier passes
        // that visibility to the threads publishing the reserved queue range.
        for (int group = 0; group < 2; ++group) {
            int *range = publication + group * 3;
            for (int i = threadIdx.x; i < range[1]; i += threads)
                release(g.queue + range[2] + i, range[0] + i + 1);
        }
        __syncthreads();
#endif
    }
}

} // namespace nano

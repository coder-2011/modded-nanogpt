#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace nano {

constexpr int tile = 64;
constexpr int threads = 128;
using fp8 = __nv_fp8_e4m3;
using bf16 = __nv_bfloat16;

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
};

struct Task {
    int op, row, col;
    int signal[2];
};

struct Group {
    int expected, begin, end;
};

struct Graph {
    const Matmul *ops;
    const Task *tasks;
    const Group *groups;
    int *counters;
    int *queue;
    int *state; // head, tail, unfinished
    int task_count;
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

__device__ __forceinline__ int pop(Graph g) {
    const int head = atomicAdd(g.state, 1);
    if (head >= g.task_count)
        return -2;
    int value;
    // A producer reserves a slot before publishing it; it never waits for a consumer.
    while (!(value = acquire(g.queue + head)))
        __nanosleep(32);
    return value - 1;
}

__device__ __forceinline__ void signal(Graph g, int id) {
    if (id < 0)
        return;
    const Group group = g.groups[id];
    // Acq_rel chains every producer's stores through the final arrival.
    if (add_acq_rel(g.counters + id, 1) + 1 == group.expected)
        for (int task = group.begin; task < group.end; ++task)
            push(g, task);
}

__device__ __forceinline__ void mma(float *d, const uint32_t *a, const uint32_t *b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ void matmul_tile(const Matmul &op, const Task &task, fp8 *sa, fp8 *sb) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
    const int row0 = task.row * tile, col0 = task.col * tile;
    float acc[2][4][4] = {};
    for (int k0 = 0; k0 < op.k; k0 += 32) {
        if (op.ak == 1 && op.bk == 1 && op.k % 32 == 0 && op.ar % 16 == 0 && op.br % 16 == 0) {
            int r = threadIdx.x / 2, k = k0 + threadIdx.x % 2 * 16;
            const fp8 *ap = op.a + (row0 + r < op.m ? row0 + r : 0) * op.ar + k;
            const fp8 *bp = op.b + (col0 + r < op.n ? col0 + r : 0) * op.br + k;
            unsigned as = __cvta_generic_to_shared(sa + r * 32 + threadIdx.x % 2 * 16);
            unsigned bs = __cvta_generic_to_shared(sb + r * 32 + threadIdx.x % 2 * 16);
            int av = row0 + r < op.m ? 16 : 0, bv = col0 + r < op.n ? 16 : 0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(as), "l"(ap), "r"(av)
                         : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(bs), "l"(bp), "r"(bv)
                         : "memory");
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
        } else {
            for (int i = threadIdx.x; i < tile * 32; i += threads) {
                const int r = i / 32, k = k0 + i % 32;
                sa[i].__x =
                    row0 + r < op.m && k < op.k ? op.a[(row0 + r) * op.ar + k * op.ak].__x : 0;
                sb[i].__x =
                    col0 + r < op.n && k < op.k ? op.b[(col0 + r) * op.br + k * op.bk].__x : 0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            uint32_t a[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int r = wm + mi * 16 + lane / 4 + (j % 2) * 8;
                int k = lane % 4 * 4 + (j / 2) * 16;
                a[j] = *reinterpret_cast<const uint32_t *>(sa + r * 32 + k);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                uint32_t b[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int r = wn + ni * 8 + lane / 4;
                    int k = lane % 4 * 4 + j * 16;
                    b[j] = *reinterpret_cast<const uint32_t *>(sb + r * 32 + k);
                }
                mma(acc[mi][ni], a, b);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = row0 + wm + mi * 16 + lane / 4 + j / 2 * 8;
                const int c = col0 + wn + ni * 8 + lane % 4 * 2 + j % 2;
                if (r < op.m && c < op.n) {
                    const int index = r * op.n + c;
                    float x = acc[mi][ni][j] * op.scale;
                    if (op.raw)
                        op.raw[index] = x;
                    if (op.epilogue == Epilogue::relu_square) {
                        x = fmaxf(round_bf16(x), 0.0f);
                        x = round_bf16(x * x);
                    } else if (op.epilogue == Epilogue::relu_backward) {
                        float relu = round_bf16(sqrtf(float(op.post[index]) * op.post_scale));
                        x = round_bf16(2.0f * round_bf16(x) * relu);
                    }
                    if (op.output)
                        op.output[index] = __float2bfloat16_rn(x);
                    if (op.quantized)
                        op.quantized[index] = fp8(x / op.output_scale);
                    if (op.quantized_t)
                        sa[(r - row0) * tile + c - col0] = fp8(x / op.output_scale);
                }
            }
        }
    }
    if (op.quantized_t) {
        __syncthreads();
        for (int i = threadIdx.x; i < tile * tile; i += threads) {
            int r = row0 + i % tile, c = col0 + i / tile;
            if (r < op.m && c < op.n)
                op.quantized_t[c * op.m + r] = sa[i % tile * tile + i / tile];
        }
        __syncthreads();
    }
}

} // namespace nano

#include "hopper.cuh"

namespace nano {

__device__ __forceinline__ void execute_tile(const Matmul &op, const Task &task, fp8 *scratch) {
#ifdef NANO_WGMMA
    hopper_tile(op, task, scratch);
#else
    matmul_tile(op, task, scratch, scratch + tile * 32);
#endif
}

__global__ __launch_bounds__(threads) void megakernel(Graph g) {
    __shared__ __align__(1024) fp8 scratch[8192];
    __shared__ int next;
    for (;;) {
        if (threadIdx.x == 0) {
            next = pop(g);
        }
        __syncthreads();
        const int task_id = next;
        // Idle iterations also need a reader barrier before lane 0 reuses next.
        __syncthreads();
        if (task_id == -2)
            return;
        const Task task = g.tasks[task_id];
        const Matmul op = g.ops[task.op];
        execute_tile(op, task, scratch);
        // All output-writing threads publish before the controller signals readiness.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            signal(g, task.signal[0]);
            signal(g, task.signal[1]);
            add_acq_rel(g.state + 2, -1);
        }
        __syncthreads();
    }
}

} // namespace nano

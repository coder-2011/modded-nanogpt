#pragma once

namespace nano {

// K-major FP8 tiles: 32-byte swizzle, eight rows per 256-byte core matrix.
__device__ __forceinline__ unsigned swizzle32(unsigned index) {
    return index ^ ((index >> 7 & 1) << 4);
}

__device__ __forceinline__ uint64_t descriptor(const void *p) {
    const uint64_t address = __cvta_generic_to_shared(p);
    return ((address & 0x3ffff) >> 4) | (1ull << 16) | (16ull << 32) | (3ull << 62);
}

__device__ __forceinline__ void wgmma(float *d, uint64_t a, uint64_t b) {
    asm volatile("{ .reg .pred accumulate; setp.ne.b32 accumulate, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
                 "%32, %33, accumulate, 1, 1; }"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
                   "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
                   "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                   "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                   "+f"(d[30]), "+f"(d[31])
                 : "l"(a), "l"(b));
}

__device__ __forceinline__ void copy_wgmma_operands(const Matmul &op, int row, int col, int k0,
                                                    fp8 *a, fp8 *b) {
    if (op.ak == 1 && op.bk == 1 && op.k % 32 == 0 && op.ar % 16 == 0 && op.br % 16 == 0) {
        const int r = threadIdx.x / 2, offset = threadIdx.x % 2 * 16;
        const fp8 *ap = op.a + (row + r < op.m ? row + r : 0) * op.ar + k0 + offset;
        const fp8 *bp = op.b + (col + r < op.n ? col + r : 0) * op.br + k0 + offset;
        const unsigned as = __cvta_generic_to_shared(a + swizzle32(r * 32 + offset));
        const unsigned bs = __cvta_generic_to_shared(b + swizzle32(r * 32 + offset));
        const int av = row + r < op.m ? 16 : 0, bv = col + r < op.n ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(as), "l"(ap), "r"(av)
                     : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(bs), "l"(bp), "r"(bv)
                     : "memory");
    } else {
        for (int i = threadIdx.x; i < tile * 32; i += threads) {
            const int r = i / 32, k = k0 + i % 32;
            a[swizzle32(i)].__x =
                row + r < op.m && k < op.k ? op.a[(row + r) * op.ar + k * op.ak].__x : 0;
            b[swizzle32(i)].__x =
                col + r < op.n && k < op.k ? op.b[(col + r) * op.br + k * op.bk].__x : 0;
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void hopper_tile(const Matmul &op, const Task &task, fp8 *scratch) {
    float acc[32] = {};
    float partial[32] = {};
    const int row = task.row * tile, col = task.col * tile;
    copy_wgmma_operands(op, row, col, 0, scratch, scratch + 2048);
    for (int k = 0, stage = 0; k < op.k; k += 32, stage ^= 1) {
        fp8 *a = scratch + stage * 4096;
        fp8 *b = a + 2048;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 32; ++i)
            asm volatile("" : "+f"(partial[i])::"memory");
        asm volatile("fence.proxy.async.shared::cta; wgmma.fence.sync.aligned;" ::: "memory");
        wgmma(partial, descriptor(a), descriptor(b));
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        if (k + 32 < op.k) {
            fp8 *next = scratch + (stage ^ 1) * 4096;
            copy_wgmma_operands(op, row, col, k + 32, next, next + 2048);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
        for (int i = 0; i < 32; ++i)
            asm volatile("" : "+f"(partial[i])::"memory");
// Hopper FP8 accumulation has reduced mantissa precision; promote each K tile.
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            acc[i] += partial[i];
            partial[i] = 0;
        }
        __syncthreads();
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int r = row + warp * 16 + lane / 4 + (i % 4) / 2 * 8;
        const int c = col + (i / 4) * 8 + lane % 4 * 2 + i % 2;
        if (r < op.m && c < op.n) {
            int index = r * op.n + c;
            float x = acc[i] * op.scale;
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
                op.output[index] = bf16(x);
            if (op.quantized)
                op.quantized[index] = fp8(x / op.output_scale);
            if (op.quantized_t)
                scratch[(r - row) * tile + c - col] = fp8(x / op.output_scale);
        }
    }
    if (op.quantized_t) {
        __syncthreads();
        for (int i = threadIdx.x; i < tile * tile; i += threads) {
            const int r = row + i % tile, c = col + i / tile;
            if (r < op.m && c < op.n)
                op.quantized_t[c * op.m + r] = scratch[i % tile * tile + i / tile];
        }
        __syncthreads();
    }
}

} // namespace nano

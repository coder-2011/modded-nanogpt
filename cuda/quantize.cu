#include "bench.cuh"
#include "megakernel.cuh"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <random>
using namespace nano;

struct Quantized {
    const bf16 *input;
    fp8 *values;
    float *peaks;
    int rows, cols, group_rows, group_cols;
    __host__ __device__ int groups_k() const { return (cols + group_cols - 1) / group_cols; }
    __host__ __device__ int group(int r, int k) const {
        return r / group_rows * groups_k() + k / group_cols;
    }
};

__global__ void minmax(Quantized q) {
    int row = blockIdx.y * min(q.group_rows, 64), k = blockIdx.x * min(q.group_cols, 128);
    int nr = min(min(q.group_rows, 64), q.rows - row), nk = min(min(q.group_cols, 128), q.cols - k);
    float lo = 0, hi = 0;
    for (int i = threadIdx.x; i < nr * nk; i += blockDim.x) {
        float v = float(q.input[(row + i / nk) * q.cols + k + i % nk]);
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
    }
    for (int d = 16; d; d /= 2) {
        lo = fminf(lo, __shfl_down_sync(~0u, lo, d));
        hi = fmaxf(hi, __shfl_down_sync(~0u, hi, d));
    }
    __shared__ float peaks[8];
    if (threadIdx.x % 32 == 0)
        peaks[threadIdx.x / 32] = fmaxf(-lo, hi);
    __syncthreads();
    if (threadIdx.x < 32) {
        float p = threadIdx.x < 8 ? peaks[threadIdx.x] : 0;
        for (int d = 16; d; d /= 2)
            p = fmaxf(p, __shfl_down_sync(~0u, p, d));
        if (threadIdx.x == 0)
            atomicMax(reinterpret_cast<unsigned *>(q.peaks + q.group(row, k)), __float_as_uint(p));
    }
}
__global__ void quantize(Quantized q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q.rows * q.cols) {
        float scale = fmaxf(q.peaks[q.group(i / q.cols, i % q.cols)] / 448.0f, 1e-12f);
        q.values[i] = fp8(float(q.input[i]) / scale);
    }
}
__global__ void unpack_quantized(Quantized q, float *out, bool original) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q.rows * q.cols)
        out[i] = original ? float(q.input[i])
                          : float(q.values[i]) *
                                fmaxf(q.peaks[q.group(i / q.cols, i % q.cols)] / 448.0f, 1e-12f);
}
void encode(Quantized q) {
    size_t count = size_t((q.rows + q.group_rows - 1) / q.group_rows) * q.groups_k();
    CHECK_CUDA(cudaMemsetAsync(q.peaks, 0, count * sizeof(float)));
    minmax<<<dim3((q.cols + std::min(q.group_cols, 128) - 1) / std::min(q.group_cols, 128),
                  (q.rows + std::min(q.group_rows, 64) - 1) / std::min(q.group_rows, 64)),
             256>>>(q);
    quantize<<<(q.rows * q.cols + 255) / 256, 256>>>(q);
}

template <bool InlineQuant = false>
__global__ void scaled_gemm(Quantized a, Quantized b, bf16 *output) {
    __shared__ __align__(1024) fp8 scratch[2 * operand_bytes];
    extern __shared__ bf16 raw[];
    __shared__ float extrema[InlineQuant ? 8 : 1], scales[InlineQuant ? 2 : 1];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int wm = warp / 2 * 32, wn = warp % 2 * 32;
    int r0 = blockIdx.y * 64, c0 = blockIdx.x * 64;
    float result[2][4][4] = {};
    Matmul op{a.values, b.values, nullptr, nullptr, nullptr, a.rows, b.rows, a.cols,
              a.cols,   1,        b.cols,  1,       1,       0,      0,      Epilogue::linear};
    // A K-dependent scale must be applied before adding the next scaled partial.
    for (int begin = 0; begin < a.cols; begin += a.group_cols) {
        int end = min(begin + a.group_cols, a.cols);
        float acc[2][4][4] = {};
        for (int k0 = begin; k0 < end; k0 += k_tile) {
            if constexpr (InlineQuant) {
                static_assert(k_tile == 128);
                float ap = 0, bp = 0;
                for (int i = threadIdx.x; i < operand_bytes; i += blockDim.x) {
                    int r = i / k_tile, k = k0 + i % k_tile;
                    bf16 av = r0 + r < a.rows && k < end ? a.input[(r0 + r) * a.cols + k] : bf16(0);
                    bf16 bv = c0 + r < b.rows && k < end ? b.input[(c0 + r) * b.cols + k] : bf16(0);
                    raw[i] = av;
                    raw[operand_bytes + i] = bv;
                    ap = fmaxf(ap, fabsf(float(av)));
                    bp = fmaxf(bp, fabsf(float(bv)));
                }
                for (int d = 16; d; d /= 2) {
                    ap = fmaxf(ap, __shfl_down_sync(~0u, ap, d));
                    bp = fmaxf(bp, __shfl_down_sync(~0u, bp, d));
                }
                if (lane == 0) {
                    extrema[warp] = ap;
                    extrema[4 + warp] = bp;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    ap = bp = 0;
                    for (int i = 0; i < 4; ++i) {
                        ap = fmaxf(ap, extrema[i]);
                        bp = fmaxf(bp, extrema[4 + i]);
                    }
                    scales[0] = fmaxf(ap / 448.0f, 1e-12f);
                    scales[1] = fmaxf(bp / 448.0f, 1e-12f);
                }
                __syncthreads();
                for (int i = threadIdx.x; i < operand_bytes; i += blockDim.x) {
                    int index = operand_index(i / k_tile, i % k_tile);
                    scratch[index] = fp8(float(raw[i]) / scales[0]);
                    scratch[operand_bytes + index] = fp8(float(raw[operand_bytes + i]) / scales[1]);
                }
            } else {
                load_operands(op, r0, c0, k0, end, scratch, scratch + operand_bytes);
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
            for (int ki = 0; ki < k_tile && k0 + ki < end; ki += 32) {
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    uint32_t av[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        av[j] = *reinterpret_cast<const uint32_t *>(
                            scratch + operand_index(wm + mi * 16 + lane / 4 + (j % 2) * 8,
                                                    ki + lane % 4 * 4 + j / 2 * 16));
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        uint32_t bv[2];
#pragma unroll
                        for (int j = 0; j < 2; ++j)
                            bv[j] = *reinterpret_cast<const uint32_t *>(
                                scratch + operand_bytes +
                                operand_index(wn + ni * 8 + lane / 4, ki + lane % 4 * 4 + j * 16));
                        mma(acc[mi][ni], av, bv);
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
                    int r = r0 + wm + mi * 16 + lane / 4 + j / 2 * 8,
                        c = c0 + wn + ni * 8 + lane % 4 * 2 + j % 2;
                    if (r < a.rows && c < b.rows) {
                        float as, bs;
                        if constexpr (InlineQuant) {
                            as = scales[0];
                            bs = scales[1];
                        } else {
                            as = fmaxf(a.peaks[a.group(r, begin)] / 448.0f, 1e-12f);
                            bs = fmaxf(b.peaks[b.group(c, begin)] / 448.0f, 1e-12f);
                        }
                        result[mi][ni][j] = fmaf(acc[mi][ni][j], as * bs, result[mi][ni][j]);
                    }
                }
        if constexpr (InlineQuant)
            __syncthreads(); // All readers finish before the next tile overwrites its scales.
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int r = r0 + wm + mi * 16 + lane / 4 + j / 2 * 8,
                    c = c0 + wn + ni * 8 + lane % 4 * 2 + j % 2;
                if (r < a.rows && c < b.rows)
                    output[r * b.rows + c] = bf16(result[mi][ni][j]);
            }
}

double relative_error(const std::vector<bf16> &got, const std::vector<float> &expected,
                      bool rounded) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double e = rounded ? float(bf16(expected[i])) : expected[i], v = float(got[i]);
        if (!std::isfinite(v) || !std::isfinite(e))
            throw std::runtime_error("nonfinite GEMM");
        error += (v - e) * (v - e);
        norm += e * e;
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}
double tile_error(Quantized q) {
    std::vector<bf16> src(size_t(q.rows) * q.cols);
    std::vector<fp8> dst(src.size());
    std::vector<float> peaks(size_t((q.rows + q.group_rows - 1) / q.group_rows) * q.groups_k());
    CHECK_CUDA(cudaMemcpy(src.data(), q.input, src.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(dst.data(), q.values, dst.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(peaks.data(), q.peaks, peaks.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int r = 0; r < q.rows; r += 64)
        for (int k = 0; k < q.cols; k += 128) {
            double error = 0, norm = 0;
            for (int i = r; i < std::min(r + 64, q.rows); ++i)
                for (int j = k; j < std::min(k + 128, q.cols); ++j) {
                    double x = float(src[i * q.cols + j]);
                    double v = float(dst[i * q.cols + j]) *
                               std::max(peaks[q.group(i, j)] / 448.0f, 1e-12f);
                    error += (v - x) * (v - x);
                    norm += x * x;
                }
            worst = std::max(worst, std::sqrt(error / std::max(norm, 1e-30)));
        }
    return worst;
}
void experiment(int m, int n, int k, int distribution, bool timed) {
    printf("quantization shape=%dx%dx%d distribution=%d\n", m, n, k, distribution);
    DeviceBuffer<bf16> a(size_t(m) * k), b(size_t(n) * k), out(size_t(m) * n);
    DeviceBuffer<fp8> aq(a.n), bq(b.n);
    DeviceBuffer<float> af(a.n), bf(b.n), ref(out.n), exact(out.n);
    std::mt19937 random(1337);
    std::normal_distribution<float> normal;
    for (auto *buffer : {&a, &b}) {
        std::vector<bf16> v(buffer->n);
        for (size_t i = 0; i < v.size(); ++i) {
            int r = int(i / k), c = int(i % k);
            float scale =
                distribution == 1 ? std::exp2(float(((r / 64) * 7 + c / 128) % 25 - 12)) : 1;
            v[i] = bf16(distribution == 2 ? 0 : normal(random) * scale);
        }
        buffer->put(v);
    }
    cublasHandle_t blas;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasCreate");
    cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
    auto reference = [&](Quantized qa, Quantized qb, DeviceBuffer<float> &target, bool original) {
        unpack_quantized<<<(a.n + 255) / 256, 256>>>(qa, af.p, original);
        unpack_quantized<<<(b.n + 255) / 256, 256>>>(qb, bf.p, original);
        float one = 1, zero = 0;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, bf.p, k, af.p, k, &zero,
                        target.p, n) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm");
    };
    auto bf16_control = [&] {
        float one = 1, zero = 0;
        if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, b.p, CUDA_R_16BF, k, a.p,
                         CUDA_R_16BF, k, &zero, out.p, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("BF16 cublasGemmEx");
    };
    double best = 1e30;
    std::string winner = "none";
    for (const std::string mode : {"layer", "row", "row128", "tile64x128", "rows64"}) {
        int gr = mode == "layer" ? std::max(m, n) : (mode == "row" || mode == "row128" ? 1 : 64);
        int gk = mode == "row128" || mode == "tile64x128" ? 128 : k;
        DeviceBuffer<float> ap(size_t((m + gr - 1) / gr) * ((k + gk - 1) / gk));
        DeviceBuffer<float> bp(size_t((n + gr - 1) / gr) * ((k + gk - 1) / gk));
        Quantized qa{a.p, aq.p, ap.p, m, k, gr, gk}, qb{b.p, bq.p, bp.p, n, k, gr, gk};
        encode(qa);
        encode(qb);
        CHECK_CUDA(cudaMemset(out.p, 0xff, out.n * sizeof(bf16)));
        scaled_gemm<<<dim3((n + 63) / 64, (m + 63) / 64), 128>>>(qa, qb, out.p);
        reference(qa, qb, ref, false);
        reference(qa, qb, exact, true);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        auto output = out.get();
        double math_error = relative_error(output, ref.get(), true);
        double original_error = relative_error(output, exact.get(), false);
        double input_error = std::max(tile_error(qa), tile_error(qb));
        // These screening gates are separate from, and do not weaken, native arithmetic parity.
        bool admissible = input_error <= 0.04 && original_error <= 0.06;
        printf("mode=%s scale_bytes=%zu math_rel_l2=%.8g original_rel_l2=%.8g "
               "worst_tile_rel_l2=%.8g admissible=%d\n",
               mode.c_str(), (ap.n + bp.n) * 4, math_error, original_error, input_error,
               admissible);
        if (math_error > 0.0002)
            throw std::runtime_error("scaled MMA numerical mismatch");
        if (mode == "layer") {
            bf16_control();
            double control_error = relative_error(out.get(), exact.get(), true);
            printf("bf16_cublas_rel_l2=%.8g\n", control_error);
            if (control_error > 0.0002)
                throw std::runtime_error("BF16 control mismatch");
            if (timed)
                device_time("bf16_cublas_gemm", bf16_control);
        }
        if (mode == "tile64x128") {
            CHECK_CUDA(cudaMemset(out.p, 0xff, out.n * sizeof(bf16)));
            scaled_gemm<true>
                <<<dim3((n + 63) / 64, (m + 63) / 64), 128, 2 * operand_bytes * sizeof(bf16)>>>(
                    qa, qb, out.p);
            CHECK_CUDA(cudaGetLastError());
            auto fused = out.get();
            if (std::memcmp(fused.data(), output.data(), output.size() * sizeof(bf16)))
                throw std::runtime_error("inline quantization differs from separate quantization");
            puts("inline_tile64x128: exact BF16 output match");
            if (timed) {
                double ms = device_time("inline_tile64x128_total", [&] {
                    scaled_gemm<true><<<dim3((n + 63) / 64, (m + 63) / 64), 128,
                                        2 * operand_bytes * sizeof(bf16)>>>(qa, qb, out.p);
                });
                if (admissible && ms < best) {
                    best = ms;
                    winner = "inline_tile64x128";
                }
            }
        }
        if (timed) {
            device_time((mode + "_quant").c_str(), [&] {
                encode(qa);
                encode(qb);
            });
            device_time((mode + "_gemm").c_str(), [&] {
                scaled_gemm<<<dim3((n + 63) / 64, (m + 63) / 64), 128>>>(qa, qb, out.p);
            });
            double ms = device_time((mode + "_total").c_str(), [&] {
                encode(qa);
                encode(qb);
                scaled_gemm<<<dim3((n + 63) / 64, (m + 63) / 64), 128>>>(qa, qb, out.p);
            });
            if (admissible && ms < best) {
                best = ms;
                winner = mode;
            }
        }
    }
    if (timed)
        printf("fastest_admissible=%s median_ms=%.6f (synthetic screening, not convergence)\n",
               winner.c_str(), best);
    cublasDestroy(blas);
}
int main(int argc, char **argv) {
    try {
        device_info();
        CHECK_CUDA(cudaFuncSetAttribute(scaled_gemm<true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        2 * operand_bytes * sizeof(bf16)));
        cudaFuncAttributes attributes;
        CHECK_CUDA(cudaFuncGetAttributes(&attributes, scaled_gemm<true>));
        int blocks;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scaled_gemm<true>, 128,
                                                                 2 * operand_bytes * sizeof(bf16)));
        printf("inline_quant registers=%d static_shared=%zu dynamic_shared=%zu blocks_per_sm=%d\n",
               attributes.numRegs, attributes.sharedSizeBytes, 2 * operand_bytes * sizeof(bf16),
               blocks);
        bool quick = argc > 1 && std::string(argv[1]) == "--quick";
        puts("screening: worst 64x128 input tile relative L2 <=0.04; output relative L2 <=0.06; "
             "arithmetic <=0.0002");
        for (int distribution : {0, 1, 2})
            experiment(65, 97, 160, distribution, false);
        if (!quick)
            for (int distribution : {0, 1})
                experiment(16384, 3072, 768, distribution, true);
        puts("PASS: scale-granularity experiment; no full-model quality or speedup claim");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

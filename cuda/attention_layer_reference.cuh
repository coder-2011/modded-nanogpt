#pragma once
#include "bench.cuh"
#include "attention_reference.cuh"
#include "megakernel.cuh"
#include <cublas_v2.h>
#include <cmath>
#include <cstring>

namespace nano {

template <class T> std::vector<T> layer_read(const T *pointer, size_t size) {
    std::vector<T> values(size);
    CHECK_CUDA(cudaMemcpy(values.data(), pointer, size * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
}

template <class A, class B>
void layer_near(const char *name, const std::vector<A> &a, const std::vector<B> &b,
                double relative_limit = 0.0002, double absolute_limit = 0.00002) {
    if (a.size() != b.size()) throw std::runtime_error(std::string(name) + " size mismatch");
    double squared = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = double(a[i]), y = double(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) throw std::runtime_error(std::string(name) + " non-finite reference");
        squared += (x - y) * (x - y); norm += y * y;
        maximum = std::max(maximum, std::abs(x - y));
    }
    double relative = std::sqrt(squared / std::max(norm, 1e-30));
    printf("LAYER %s rel_l2=%.9g max_abs=%.9g\n", name, relative, maximum);
    if (relative > relative_limit && maximum > absolute_limit)
        throw std::runtime_error(std::string(name) + " independent reference mismatch");
}

struct LayerBlasReference {
    cublasHandle_t handle;
    static void check(cublasStatus_t s) {
        if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("layer cuBLAS error " + std::to_string(s));
    }
    LayerBlasReference() { check(cublasCreate(&handle)); check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)); }
    ~LayerBlasReference() { cublasDestroy(handle); }
    std::vector<float> multiply(const std::vector<float> &a, const std::vector<float> &b, int m, int n, int k) {
        DeviceBuffer<float> da(a.size()), db(b.size()), result(size_t(m) * n);
        da.put(a); db.put(b);
        float one = 1, zero = 0;
        check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, db.p, k, da.p, k, &zero, result.p, n));
        return result.get();
    }
    template <class T>
    std::vector<float> operand(const T *pointer, int rows, int k, int64_t sr, int64_t sk, float scale = 1) {
        auto storage = layer_read(pointer, int64_t(rows - 1) * sr + int64_t(k - 1) * sk + 1);
        std::vector<float> result(size_t(rows) * k);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < k; ++c) {
                float value = float(storage[r * sr + c * sk]);
                result[r * k + c] = scale == 1.0f ? value : float(bf16(value * scale));
            }
        return result;
    }
    void check_product(const Matmul &op) {
        auto a = operand(op.a, op.m, op.k, op.ar, op.ak);
        auto b = operand(op.b, op.n, op.k, op.br, op.bk);
        auto reference = multiply(a, b, op.m, op.n, op.k);
        for (auto &x : reference) x *= op.scale;
        auto raw = layer_read(op.raw, reference.size());
        layer_near("FP8 projection/product", raw, reference);
        auto rounded = layer_read(op.output, reference.size());
        for (size_t i = 0; i < raw.size(); ++i)
            if (float(rounded[i]) != float(bf16(raw[i]))) throw std::runtime_error("layer FP8 output rounding mismatch");
    }
    void check_product(const BF16Matmul &op) {
        auto a = operand(op.a, op.m, op.k, op.ar, op.ak);
        auto b = operand(op.b, op.n, op.k, op.br, op.bk, op.b_scale);
        auto reference = multiply(a, b, op.m, op.n, op.k);
        auto raw = layer_read(op.raw, reference.size());
        layer_near("BF16 projection/product", raw, reference);
        auto rounded = layer_read(op.output, reference.size());
        for (size_t i = 0; i < raw.size(); ++i)
            if (float(rounded[i]) != float(bf16(raw[i]))) throw std::runtime_error("layer BF16 output rounding mismatch");
    }
};

struct PostReference {
    std::vector<double> output, dy, dv, da, dg;
    PostReference(const std::vector<bf16> &y, const std::vector<bf16> &v, const std::vector<bf16> &incoming,
                  const std::vector<bf16> *alpha, const std::vector<bf16> *gate, int width)
        : output(y.size()), dy(y.size()), dv(y.size()), da(y.size() / width), dg(da.size()) {
        for (size_t row = 0; row < da.size(); ++row) {
            size_t base = row * width;
            double dot = 0, square = 0, gv = 0;
            double a = alpha ? std::tanh(double(float((*alpha)[row]))) : 0;
            double g = gate ? float((*gate)[row]) : 1;
            for (int c = 0; c < width; ++c) {
                double x = float(y[base + c]), value = float(v[base + c]);
                dot += x * value; square += value * value;
                gv += double(float(incoming[base + c])) * g * value;
            }
            double denom = std::max(square, double(1e-8f));
            for (int c = 0; c < width; ++c) {
                size_t i = base + c;
                double x = float(y[i]), value = float(v[i]), grad = float(incoming[i]) * g;
                double z = x - a * dot / denom * value;
                output[i] = z * g;
                dy[i] = grad - a * gv * value / denom;
                dv[i] = -a * (dot * grad + gv * x) / denom;
                if (square >= double(1e-8f)) dv[i] += 2 * a * gv * dot * value / (denom * denom);
                if (gate) dg[row] += double(float(incoming[i])) * z;
            }
            if (alpha) da[row] = -dot / denom * gv * (1 - a * a);
        }
    }
};

inline void post_finite_difference(const std::vector<bf16> &y, const std::vector<bf16> &v,
                                   const std::vector<bf16> &incoming, const std::vector<bf16> *alpha,
                                   const std::vector<bf16> *gate, int width, const PostReference &reference) {
    std::vector<double> x(width), value(width);
    for (int i = 0; i < width; ++i) { x[i] = float(y[i]); value[i] = float(v[i]); }
    double a = alpha ? float((*alpha)[0]) : 0, g = gate ? float((*gate)[0]) : 1;
    auto loss = [&] {
        double dot = 0, square = 0, result = 0;
        for (int i = 0; i < width; ++i) { dot += x[i] * value[i]; square += value[i] * value[i]; }
        double coefficient = std::tanh(a) * dot / std::max(square, double(1e-8f));
        for (int i = 0; i < width; ++i)
            result += double(float(incoming[i])) * g * (x[i] - coefficient * value[i]);
        return result;
    };
    std::vector<double> numerical, analytic;
    auto probe = [&](double &parameter, double step, double expected) {
        double saved = parameter;
        parameter = saved + step; double plus = loss();
        parameter = saved - step; double minus = loss();
        parameter = saved;
        numerical.push_back((plus - minus) / (2 * step)); analytic.push_back(expected);
    };
    for (int i : {0, width / 2, width - 1}) {
        probe(x[i], 1e-6, reference.dy[i]);
        probe(value[i], 1e-8, reference.dv[i]);
    }
    if (alpha) probe(a, 1e-5, reference.da[0]);
    if (gate) probe(g, 1e-5, reference.dg[0]);
    layer_near("post finite difference", numerical, analytic, 1e-5, 1e-7);
}

template <class Buffers, class Schedule>
void check_layer_math(Buffers &b, Schedule &schedule) {
    auto scalars = b.scalars.get();
    auto fp8_ops = schedule.ops.get();
    auto bf16_ops = schedule.bf16_ops.get();
    float ws = scalars[1] * scalars[3];
    for (int i = 0; i < b.forwards; ++i)
        if (fp8_ops[i].scale != scalars[0] * ws) throw std::runtime_error("stale QKV forward scale");
    if (fp8_ops[b.forwards].scale != scalars[2] * scalars[0] ||
        fp8_ops[b.forwards + 1].scale != scalars[2] * ws ||
        bf16_ops[0].b_scale != scalars[4] * scalars[5] || bf16_ops[1].b_scale != scalars[4] * scalars[5] ||
        schedule.qkv.get()[0].grad_scale != scalars[2]) throw std::runtime_error("stale projection gradient scale");
    LayerBlasReference blas;
    for (const auto &op : fp8_ops) blas.check_product(op);
    for (const auto &op : bf16_ops) blas.check_product(op);
    auto projected = b.projected.get();
    auto projected_v = b.paired ? b.projected_v.get() : std::vector<bf16>{};
    auto f1 = b.factors1.get(), f2 = b.factors2.get(), aux = b.aux.get();
    int stride = b.paired ? 12 * b.d : b.f, factor_stride = b.d * (b.paired ? 2 : 1);
    auto rms = [&](int t, int head) {
        double square = 0;
        for (int c = 0; c < b.d; ++c) {
            double value = float(projected[t * stride + head * b.d + c]); square += value * value;
        }
        return 1.0 / std::sqrt(square / b.d + 1.1920928955078125e-7);
    };
    std::vector<float> refq(b.q.n), refk(b.k.n), refv(b.value.n);
    for (int t = 0; t < b.t; ++t)
        for (int h = 0; h < 6; ++h) {
            int fi = t * factor_stride + (b.paired ? h % 2 * b.d : 0);
            for (int kind = 0; kind < 2; ++kind) {
                int ih = h + kind * 6, input = t * stride + ih * b.d;
                double r = rms(t, ih);
                for (int c = 0; c < b.d; ++c) {
                    double x = float(f1[fi + c]) * double(float(projected[input + c])) * r +
                               float(f2[fi + c]) * double(float(projected[input + (c ^ 1)])) * r;
                    if (b.shifted && kind && t && c >= b.d / 2)
                        x = float(projected[input - stride + c]) * rms(t - 1, ih);
                    (kind ? refk : refq)[(t * 6 + h) * b.d + c] = float(bf16(float(x)));
                }
            }
            for (int c = 0; c < b.v; ++c) {
                float x = b.paired ? float(projected_v[(t * 6 + h) * b.v + c]) :
                    float(projected[t * stride + 12 * b.d + h * b.v + c]);
                if (b.auxiliary) x += float(aux[(t * 6 + h) * 128 + c]);
                refv[(t * 6 + h) * b.v + c] = float(bf16(x));
            }
        }
    auto as_float = [](const std::vector<bf16> &x) { return std::vector<float>(x.begin(), x.end()); };
    layer_near("Q transform", as_float(b.q.get()), refq);
    layer_near("K transform", as_float(b.k.get()), refk);
    layer_near("V transform", as_float(b.value.get()), refv);
    struct AttentionInputs {
        int tokens, heads, d, v, window;
        float scale;
        std::vector<int> seq;
        std::vector<bf16> q, k, value, dy;
    };
    AttentionInputs inputs{b.t * (b.paired ? 2 : 1), b.paired ? 3 : 6, b.d, b.v,
                           schedule.plan.attention.window, schedule.plan.attention.scale, b.seq,
                           b.q.get(), b.k.get(), b.value.get(), b.attn_dy.get()};
    auto saved_y = b.attn_y.get();
    AttentionReference attention(inputs, &saved_y);
    layer_near("attention Y", b.raw_a_y.get(), attention.y);
    layer_near("attention dQ", b.raw_a_dq.get(), attention.dq);
    layer_near("attention dK", b.raw_a_dk.get(), attention.dk);
    layer_near("attention dV", b.raw_a_dv.get(), attention.dv);
    auto alpha = b.alpha.get(), gate = b.gate.get();
    auto dpost = b.dpost.get();
    PostReference post(saved_y, inputs.value, dpost, b.xsa ? &alpha : nullptr, b.gated ? &gate : nullptr, b.v);
    post_finite_difference(saved_y, inputs.value, dpost, b.xsa ? &alpha : nullptr, b.gated ? &gate : nullptr, b.v, post);
    layer_near("post Y", b.raw_post.get(), post.output);
    layer_near("post dY", b.raw_post_dy.get(), post.dy);
    layer_near("post dV side", b.side.get(), post.dv);
    layer_near("post dAlpha", b.raw_da.get(), post.da);
    layer_near("post dGate", b.raw_dg.get(), post.dg);
    auto rounding = [](const std::vector<bf16> &output, const std::vector<float> &raw) {
        for (size_t i = 0; i < raw.size(); ++i)
            if (float(output[i]) != float(bf16(raw[i]))) throw std::runtime_error("layer boundary BF16 rounding mismatch");
    };
    rounding(saved_y, b.raw_a_y.get()); rounding(b.dq.get(), b.raw_a_dq.get()); rounding(b.dk.get(), b.raw_a_dk.get());
    rounding(b.post_y.get(), b.raw_post.get()); rounding(b.attn_dy.get(), b.raw_post_dy.get());
    rounding(b.dalpha.get(), b.raw_da.get()); rounding(b.dgate.get(), b.raw_dg.get());
    auto side = b.side.get(), attention_dv = b.raw_a_dv.get();
    auto dv = b.dv.get();
    auto daux = b.auxiliary ? b.daux.get() : std::vector<bf16>{};
    for (size_t i = 0; i < dv.size(); ++i) {
        float expected = float(bf16(float(bf16(attention_dv[i])) + side[i]));
        if (float(dv[i]) != expected) throw std::runtime_error("attention/XSA value-gradient join mismatch");
        if (b.auxiliary && float(daux[(i / b.v) * 128 + i % b.v]) != expected)
            throw std::runtime_error("auxiliary-value gradient mismatch");
    }
    if (b.auxiliary && b.v == 64)
        for (int r = 0; r < b.t * 6; ++r)
            for (int c = 64; c < 128; ++c)
                if (float(daux[r * 128 + c]) != 0) throw std::runtime_error("inactive auxiliary gradient not zero");
    auto dq = b.dq.get(), dk = b.dk.get();
    std::vector<double> qkv_gradient(b.raw_qkv.n);
    for (int t = 0; t < b.t; ++t)
        for (int h = 0; h < 6; ++h) {
            int fi = t * factor_stride + (b.paired ? h % 2 * b.d : 0);
            for (int kind = 0; kind < 2; ++kind) {
                const auto &incoming = kind ? dk : dq;
                int ih = h + kind * 6, xi = t * stride + ih * b.d, oi = (t * 6 + h) * b.d;
                double r = rms(t, ih), correction = 0;
                std::vector<double> g(b.d);
                for (int c = 0; c < b.d; ++c) {
                    g[c] = float(f1[fi + c]) * double(float(incoming[oi + c])) +
                           float(f2[fi + (c ^ 1)]) * double(float(incoming[oi + (c ^ 1)]));
                    if (b.shifted && kind && c >= b.d / 2) {
                        g[c] = t + 1 < b.t ? float(incoming[oi + 6 * b.d + c]) : 0;
                        if (t == 0) g[c] += float(incoming[oi + c]);
                    }
                    correction += g[c] * float(projected[xi + c]) * r;
                }
                correction /= b.d;
                for (int c = 0; c < b.d; ++c)
                    qkv_gradient[t * b.f + ih * b.d + c] = r * (g[c] - float(projected[xi + c]) * r * correction);
            }
            for (int c = 0; c < b.v; ++c) qkv_gradient[t * b.f + 12 * b.d + h * b.v + c] = float(dv[(t * 6 + h) * b.v + c]);
        }
    layer_near("QKV backward", b.raw_qkv.get(), qkv_gradient);
    auto raw = b.raw_qkv.get();
    auto packed = b.grad8.get(), transposed = b.gradt8.get();
    for (int t = 0; t < b.t; ++t)
        for (int c = 0; c < b.f; ++c) {
            float x = c < 12 * b.d ? float(bf16(raw[t * b.f + c])) : raw[t * b.f + c];
            fp8 expected(x / scalars[2]);
            if (packed[t * b.f + c].__x != expected.__x || transposed[c * b.t + t].__x != expected.__x)
                throw std::runtime_error("QKV dual-layout FP8 gradient rounding mismatch");
        }
    auto check_gain = [&](const std::vector<bf16> &weight, const std::vector<bf16> &unscaled,
                          const std::vector<bf16> &gradient, float gain) {
        double sum = 0;
        for (size_t i = 0; i < weight.size(); ++i) {
            sum += double(float(weight[i])) * float(unscaled[i]);
            if (float(gradient[i]) != float(bf16(float(unscaled[i]) * gain)))
                throw std::runtime_error("projection weight-gradient gain mismatch");
        }
        return sum;
    };
    double qg = check_gain(b.weights.get(), b.dwq0.get(), b.dwq.get(), scalars[3]);
    double og = check_gain(b.wo.get(), b.dwo0.get(), b.dwo.get(), scalars[4] * scalars[5]);
    layer_near("gain gradients", b.gains.get(), std::vector<double>{qg, og * scalars[5], og * scalars[4]});
}

} // namespace nano

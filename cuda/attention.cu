#include "bench.cuh"
#include "megakernel.cuh"
#include "frontier.cuh"
#include <cuda_profiler_api.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

using namespace nano;

struct Inputs {
    int tokens, heads, d, v, window;
    float scale;
    std::vector<int> seq;
    std::vector<bf16> q, k, value, dy;
    Inputs(std::vector<int> lengths, int qk_dim, int v_dim, int w, bool paired, int pattern)
        : tokens(std::accumulate(lengths.begin(), lengths.end(), 0) * (paired ? 2 : 1)),
          heads(paired ? 3 : frontier::heads), d(qk_dim), v(v_dim), window(w),
          scale(d == 64 ? 0.13f : 0.085f), seq{0},
          q(tokens * heads * d), k(q.size()), value(tokens * heads * v), dy(value.size()) {
        for (int n : lengths)
            seq.push_back(seq.back() + n * (paired ? 2 : 1));
        std::mt19937 rng(417 + tokens + d + v);
        std::normal_distribution<float> normal;
        for (auto *buffer : {&q, &k, &value, &dy})
            for (auto &x : *buffer)
                x = __float2bfloat16_rn(normal(rng) * 0.4f);
        if (pattern == 1) {
            std::fill(q.begin(), q.end(), bf16(0.0f));
            std::fill(k.begin(), k.end(), bf16(0.0f));
        } else if (pattern == 2) {
            for (auto *buffer : {&q, &k})
                for (auto &x : *buffer)
                    x = bf16(float(x) * 8.0f);
        }
    }
};

struct Reference {
    std::vector<double> y, dq, dk, dv, lse, delta;
    explicit Reference(const Inputs &in)
        : y(in.value.size()), dq(in.q.size()), dk(in.q.size()), dv(in.value.size()),
          lse(in.tokens * in.heads), delta(lse.size()) {
        // Materialize each probability row in FP64, independently of the device's
        // online recurrence and key-owned backward traversal.
        for (size_t doc = 1; doc < in.seq.size(); ++doc)
            for (int t = in.seq[doc - 1]; t < in.seq[doc]; ++t)
                for (int h = 0; h < in.heads; ++h) {
                    const int start = std::max(in.seq[doc - 1], t - in.window);
                    const size_t qi = (size_t(t) * in.heads + h) * in.d;
                    const size_t vi = (size_t(t) * in.heads + h) * in.v;
                    const int si = t * in.heads + h;
                    std::vector<double> p(t - start + 1);
                    for (int j = start; j <= t; ++j) {
                        size_t ki = (size_t(j) * in.heads + h) * in.d;
                        double dot = 0;
                        for (int c = 0; c < in.d; ++c)
                            dot += double(float(in.q[qi + c])) * float(in.k[ki + c]);
                        p[j - start] = dot * in.scale;
                    }
                    const double maximum = *std::max_element(p.begin(), p.end());
                    double sum = 0;
                    for (auto &x : p) {
                        x = std::exp(x - maximum);
                        sum += x;
                    }
                    lse[si] = maximum + std::log(sum);
                    for (auto &x : p)
                        x /= sum;
                    for (int j = start; j <= t; ++j) {
                        size_t kv = (size_t(j) * in.heads + h) * in.v;
                        for (int c = 0; c < in.v; ++c)
                            y[vi + c] += p[j - start] * float(in.value[kv + c]);
                    }
                    for (int c = 0; c < in.v; ++c)
                        delta[si] += double(float(bf16(float(y[vi + c])))) * float(in.dy[vi + c]);
                    for (int j = start; j <= t; ++j) {
                        size_t ki = (size_t(j) * in.heads + h) * in.d;
                        size_t kv = (size_t(j) * in.heads + h) * in.v;
                        double dp = 0;
                        for (int c = 0; c < in.v; ++c) {
                            dp += double(float(in.dy[vi + c])) * float(in.value[kv + c]);
                            dv[kv + c] += p[j - start] * float(in.dy[vi + c]);
                        }
                        double ds = p[j - start] * (dp - delta[si]) * in.scale;
                        for (int c = 0; c < in.d; ++c) {
                            dq[qi + c] += ds * float(in.k[ki + c]);
                            dk[ki + c] += ds * float(in.q[qi + c]);
                        }
                    }
                }
    }
};

struct Buffers {
    DeviceBuffer<bf16> q, k, v, dy, y, dq, dk, dv;
    DeviceBuffer<float> raw_y, raw_dq, raw_dk, raw_dv, lse, delta;
    DeviceBuffer<int> seq;
    Attention a;
    explicit Buffers(const Inputs &in)
        : q(in.q.size()), k(in.k.size()), v(in.value.size()), dy(in.dy.size()), y(v.n),
          dq(q.n), dk(k.n), dv(v.n), raw_y(v.n), raw_dq(q.n), raw_dk(k.n), raw_dv(v.n),
          lse(in.tokens * in.heads), delta(lse.n), seq(in.seq.size()),
          a{q.p, k.p, v.p, dy.p, y.p, dq.p, dk.p, dv.p, lse.p, delta.p, seq.p,
            in.tokens, in.heads, in.d, in.v, int(in.seq.size() - 1), in.window, in.scale,
            raw_y.p, raw_dq.p, raw_dk.p, raw_dv.p} {
        q.put(in.q); k.put(in.k); v.put(in.value); dy.put(in.dy); seq.put(in.seq);
    }
    void poison() {
        for (auto *b : {&y, &dq, &dk, &dv})
            CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : {&raw_y, &raw_dq, &raw_dk, &raw_dv, &lse, &delta})
            CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
    }
};

__global__ void staged_attention(Attention a, TaskKind kind) {
    execute_attention(a, kind, blockIdx.x, blockIdx.y);
}

__global__ void reset_attention_state(int *p, int roots, int count) {
    p[0] = roots; p[1] = roots; p[2] = count; p[3] = 0;
}

// Includes a dependent GEMM to exercise both operation families in one launch.
// Its output is a scheduler sentinel, not an attention output projection.
struct Program {
    std::vector<Task> tasks;
    std::vector<Group> groups;
    int roots;
    Program(const Attention &a, const QKVTransform *qkv) {
        if (a.tokens <= 0 || a.documents <= 0 || a.window < 0 ||
            (a.heads != 3 && a.heads != 6) || !std::isfinite(a.scale) || a.scale <= 0 ||
            !((a.qk_dim == 64 && (a.v_dim == 64 || a.v_dim == 128)) ||
              (a.qk_dim == 128 && a.v_dim == 128)))
            throw std::runtime_error("unsupported frontier attention geometry");
        if (qkv && (qkv->heads != 6 || qkv->tokens * (qkv->paired ? 2 : 1) != a.tokens ||
                    qkv->heads / (qkv->paired ? 2 : 1) != a.heads ||
                    qkv->qk_dim != a.qk_dim || qkv->v_dim != a.v_dim ||
                    (qkv->key_offset && (qkv->paired || qkv->qk_dim != 128)) ||
                    !std::isfinite(qkv->grad_scale) || qkv->grad_scale <= 0))
            throw std::runtime_error("inconsistent QKV transform descriptor");
        int n = ((a.tokens + 3) / 4) * a.heads;
        int prep = qkv ? ((qkv->tokens + 3) / 4) * qkv->heads : 0;
        roots = qkv ? prep : n;
        groups.resize(n + (qkv ? 4 : 2));
        if (qkv) {
            for (int i = 0; i < prep; ++i)
                tasks.push_back({0, i / qkv->heads, i % qkv->heads, {n + 2, -1}, 0, 0, TaskKind::qkv_forward});
            groups[n + 2] = {prep, prep, prep + n};
        }
        for (int pass = 0; pass < 3; ++pass)
            for (int i = 0; i < n; ++i) {
                auto kind = pass == 0 ? TaskKind::attention_forward :
                            pass == 1 ? TaskKind::attention_dq : TaskKind::attention_dkv;
                int group = pass == 0 ? i : pass == 1 ? n : n + 1;
                tasks.push_back({0, i / a.heads, i % a.heads, {group, -1}, 0, 0, kind});
                if (pass == 0)
                    groups[i] = {1, prep + n + i, prep + n + i + 1};
            }
        groups[n] = {n, prep + 2 * n, prep + 3 * n};
        groups[n + 1] = {n, prep + 3 * n, prep + 3 * n + (qkv ? prep : 1)};
        if (qkv) {
            for (int i = 0; i < prep; ++i)
                tasks.push_back({0, i / qkv->heads, i % qkv->heads, {n + 3, -1}, 0, 0, TaskKind::qkv_backward});
            groups[n + 3] = {prep, int(tasks.size()), int(tasks.size()) + 1};
        }
        tasks.push_back({1, 0, 0, {-1, -1}});
    }
};

struct Schedule {
    int count, roots;
    std::vector<int> initial;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit, device_initial;
    DeviceBuffer<Attention> attention;
    DeviceBuffer<QKVTransform> qkv;
    DeviceBuffer<Matmul> matmul;
    DeviceBuffer<fp8> ma, mb;
    DeviceBuffer<bf16> mc;
    explicit Schedule(const Attention &a, const QKVTransform *transform = nullptr)
        : Schedule(a, transform, Program(a, transform)) {}
    Schedule(const Attention &a, const QKVTransform *transform, const Program &program)
        : count(int(program.tasks.size())), roots(program.roots),
          initial(count, 0), tasks(count), groups(program.groups.size()), counters(groups.n),
          queue(count), state(4), audit(count + 2), device_initial(count), attention(1),
          qkv(1), matmul(2), ma(64 * 64), mb(ma.n), mc(ma.n) {
        for (int i = 0; i < roots; ++i)
            initial[i] = i + 1;
        std::mt19937 rng(829);
        std::shuffle(initial.begin(), initial.begin() + roots, rng);
        tasks.put(program.tasks); groups.put(program.groups); attention.put({a}); device_initial.put(initial);
        qkv.put({transform ? *transform : QKVTransform{}});
        ma.put(std::vector<fp8>(ma.n, fp8(0.125f)));
        std::vector<fp8> identity(mb.n, fp8(0.0f));
        for (int i = 0; i < 64; ++i)
            identity[i * 64 + i] = fp8(1.0f);
        mb.put(identity);
        Matmul m{ma.p, mb.p, nullptr, nullptr, mc.p, 64, 64, 64, 64, 1, 64, 1,
                 1.0f, 1.0f, 1.0f, Epilogue::linear};
        matmul.put({m, m});
    }
    void reset() {
        CHECK_CUDA(cudaMemsetAsync(counters.p, 0, counters.n * sizeof(int)));
        CHECK_CUDA(cudaMemcpyAsync(queue.p, device_initial.p, count * sizeof(int), cudaMemcpyDeviceToDevice));
        reset_attention_state<<<1, 1>>>(state.p, roots, count);
    }
    Graph graph(bool checked) {
        return {matmul.p, tasks.p, groups.p, counters.p, queue.p, state.p, count, roots,
                checked ? audit.p : nullptr, attention.p, qkv.p};
    }
    void verify() {
        for (auto x : mc.get())
            if (float(x) != 0.125f)
                throw std::runtime_error("mixed graph GEMM sentinel mismatch");
        if (state.get()[2] != 0)
            throw std::runtime_error("unfinished tasks");
    }
};

void check(const char *name, const std::vector<float> &actual, const std::vector<double> &ref) {
    double error = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = double(actual[i]) - ref[i];
        if (!std::isfinite(actual[i]))
            throw std::runtime_error(std::string(name) + " non-finite output");
        error += e * e;
        norm += ref[i] * ref[i];
        maximum = std::max(maximum, std::abs(e));
    }
    double relative = std::sqrt(error / std::max(norm, 1e-30));
    printf("%s rel_l2=%.9g max_abs=%.9g\n", name, relative, maximum);
    if (relative > 0.0002 && maximum > 0.00002)
        throw std::runtime_error(std::string(name) + " FP64 reference mismatch");
}

void check_rounded(const char *name, const std::vector<bf16> &actual, const std::vector<float> &raw) {
    for (size_t i = 0; i < actual.size(); ++i)
        if (float(actual[i]) != float(bf16(raw[i])))
            throw std::runtime_error(std::string(name) + " BF16 rounding mismatch");
}

void validate_transform(int d, int v, bool paired, bool offset, int workers, bool auxiliary = true) {
    constexpr int tokens = 21, heads = 6;
    const int features = heads * (2 * d + v), input_stride = features + 16;
    const int factor_stride = d * (paired ? 2 : 1);
    printf("qkv+attention qk=%d v=%d paired=%d key_offset=%d aux=%d workers=%d\n",
           d, v, paired, offset, auxiliary, workers);
    Inputs in({1, 7, 13}, d, v, 8, paired, 0);
    Buffers b(in);
    std::vector<bf16> input(tokens * input_stride), aux(tokens * heads * 128);
    std::vector<bf16> f1(tokens * factor_stride), f2(f1.size());
    std::mt19937 rng(3817);
    std::normal_distribution<float> normal;
    for (auto &x : input)
        x = bf16(normal(rng) * 0.4f);
    for (auto &x : aux)
        x = bf16(normal(rng) * 0.15f);
    // Zero-norm rows exercise epsilon; shifted keys must still read both the
    // previous row and their own full norm in backward at the document edge.
    for (int c = 0; c < d; ++c)
        input[c] = input[heads * d + c] = bf16(0.0f);
    for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < factor_stride; ++c) {
            int channel = c % d;
            double frequency = channel < 64 ? std::pow(1.0 / 1024, (channel / 2) / 31.0) : 0.0;
            double theta = (paired ? 2 * t + c / d : t) * frequency;
            f1[t * factor_stride + c] = bf16(float(std::cos(theta)));
            f2[t * factor_stride + c] = bf16(float(std::sin(theta) * (channel % 2 ? -1 : 1)));
        }
    DeviceBuffer<bf16> di(input.size()), da(aux.size()), df1(f1.size()), df2(f2.size());
    DeviceBuffer<fp8> packed(tokens * features), transposed(packed.n);
    DeviceBuffer<float> raw(packed.n);
    di.put(input); da.put(aux); df1.put(f1); df2.put(f2);
    QKVTransform transform{di.p, di.p + 2 * heads * d, df1.p, df2.p, auxiliary ? da.p : nullptr,
                           b.q.p, b.k.p, b.v.p, b.dq.p, b.dk.p, b.dv.p,
                           packed.p, transposed.p, tokens, heads, d, v,
                           input_stride, input_stride, 128, paired, offset, 0.03125f, raw.p};
    Schedule schedule(b.a, &transform);
    auto launch = [&](bool checked) {
        b.poison();
        for (auto *buffer : {&b.q, &b.k, &b.v})
            CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(bf16)));
        CHECK_CUDA(cudaMemset(raw.p, 0xff, raw.n * sizeof(float)));
        CHECK_CUDA(cudaMemset(packed.p, 0xff, packed.n));
        CHECK_CUDA(cudaMemset(transposed.p, 0xff, transposed.n));
        CHECK_CUDA(cudaMemset(schedule.mc.p, 0xff, schedule.mc.n * sizeof(bf16)));
        CHECK_CUDA(cudaMemset(schedule.audit.p, 0, schedule.audit.n * sizeof(int)));
        schedule.reset();
        if (checked)
            megakernel<true, false, true><<<workers, threads>>>(schedule.graph(true));
        else
            megakernel<false, false, true><<<workers, threads>>>(schedule.graph(false));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        schedule.verify();
    };
    launch(true);
    auto visits = schedule.audit.get();
    for (int i = 0; i < schedule.count; ++i)
        if (visits[i] != 1)
            throw std::runtime_error("QKV graph task count mismatch");
    auto q = b.q.get(), k = b.k.get(), value = b.v.get();
    auto rstd = [&](int t, int h) {
        double sum = 0;
        for (int c = 0; c < d; ++c) {
            double x = float(input[t * input_stride + h * d + c]);
            sum += x * x;
        }
        return 1.0 / std::sqrt(sum / d + 1.1920928955078125e-7);
    };
    std::vector<double> refq(q.size()), refk(k.size()), refv(value.size());
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h) {
            int fi = t * factor_stride + (paired ? h % 2 * d : 0);
            for (int kind = 0; kind < 2; ++kind) {
                int ih = h + kind * heads, xi = t * input_stride + ih * d;
                double scale = rstd(t, ih);
                for (int c = 0; c < d; ++c) {
                    double y = float(f1[fi + c]) * double(float(input[xi + c])) * scale +
                               float(f2[fi + c]) * double(float(input[xi + (c ^ 1)])) * scale;
                    if (offset && kind && t && c >= d / 2)
                        y = float(input[xi - input_stride + c]) * rstd(t - 1, ih);
                    (kind ? refk : refq)[(t * heads + h) * d + c] = float(bf16(float(y)));
                }
            }
            for (int c = 0; c < v; ++c)
                refv[(t * heads + h) * v + c] = float(bf16(
                    float(input[t * input_stride + 2 * heads * d + h * v + c]) +
                    (auxiliary ? float(aux[(t * heads + h) * 128 + c]) : 0.0f)));
        }
    auto floats = [](const std::vector<bf16> &x) { return std::vector<float>(x.begin(), x.end()); };
    check("transformed_q", floats(q), refq); check("transformed_k", floats(k), refk);
    check("transformed_v", floats(value), refv);
    // Check attention against the actual rounded transform boundary, then check
    // the transform's backward against the actual rounded attention gradients.
    in.q = q; in.k = k; in.value = value;
    Reference ref(in);
    check("transformed_attention_y", b.raw_y.get(), ref.y);
    check("transformed_attention_dq", b.raw_dq.get(), ref.dq);
    check("transformed_attention_dk", b.raw_dk.get(), ref.dk);
    check("transformed_attention_dv", b.raw_dv.get(), ref.dv);
    auto dq = b.dq.get(), dk = b.dk.get(), dv = b.dv.get();
    std::vector<double> ref_grad(raw.n);
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h) {
            int fi = t * factor_stride + (paired ? h % 2 * d : 0);
            int oi = (t * heads + h) * d;
            for (int kind = 0; kind < 2; ++kind) {
                int ih = h + kind * heads, xi = t * input_stride + ih * d;
                const auto &incoming = kind ? dk : dq;
                double scale = rstd(t, ih), correction = 0;
                std::vector<double> normalized(d), grad(d);
                for (int c = 0; c < d; ++c) {
                    normalized[c] = float(input[xi + c]) * scale;
                    grad[c] = double(float(f1[fi + c])) * float(incoming[oi + c]) +
                              double(float(f2[fi + (c ^ 1)])) * float(incoming[oi + (c ^ 1)]);
                    if (offset && kind && c >= d / 2) {
                        double next = t + 1 < tokens ? float(dk[oi + heads * d + c]) : 0.0;
                        grad[c] = t == 0 ? float(incoming[oi + c]) + next : next;
                    }
                    correction += normalized[c] * grad[c] / d;
                }
                for (int c = 0; c < d; ++c)
                    ref_grad[t * features + ih * d + c] = scale * (grad[c] - normalized[c] * correction);
            }
            for (int c = 0; c < v; ++c)
                ref_grad[t * features + 2 * heads * d + h * v + c] = float(dv[(t * heads + h) * v + c]);
        }
    auto actual = raw.get();
    check("qkv_gradient", actual, ref_grad);
    auto pg = packed.get(), pt = transposed.get();
    for (int t = 0; t < tokens; ++t)
        for (int f = 0; f < features; ++f) {
            int i = t * features + f;
            auto expected = fp8(float(bf16(actual[i])) / transform.grad_scale);
            if (pg[i].__x != expected.__x || pt[f * tokens + t].__x != pg[i].__x)
                throw std::runtime_error("QKV gradient quantization/layout mismatch");
        }
    launch(false);
    if (raw.get() != actual)
        throw std::runtime_error("QKV production reuse mismatch");
}

void validate(const Inputs &in, int workers) {
    printf("attention tokens=%d heads=%d qk=%d v=%d window=%d documents=%zu workers=%d\n",
           in.tokens, in.heads, in.d, in.v, in.window, in.seq.size() - 1, workers);
    Buffers b(in);
    Schedule s(b.a);
    Reference ref(in);
    b.poison();
    CHECK_CUDA(cudaMemset(s.mc.p, 0xff, s.mc.n * sizeof(bf16)));
    CHECK_CUDA(cudaMemset(s.audit.p, 0, s.audit.n * sizeof(int)));
    s.reset();
    megakernel<true, false, true><<<workers, threads>>>(s.graph(true));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    auto visits = s.audit.get();
    for (int i = 0; i < s.count; ++i)
        if (visits[i] != 1)
            throw std::runtime_error("task not executed exactly once");
    s.verify();
    auto y = b.raw_y.get(), dq = b.raw_dq.get(), dk = b.raw_dk.get(), dv = b.raw_dv.get();
    check("y", y, ref.y); check("dq", dq, ref.dq); check("dk", dk, ref.dk); check("dv", dv, ref.dv);
    check("lse", b.lse.get(), ref.lse); check("delta", b.delta.get(), ref.delta);
    check_rounded("y", b.y.get(), y); check_rounded("dq", b.dq.get(), dq);
    check_rounded("dk", b.dk.get(), dk); check_rounded("dv", b.dv.get(), dv);
    // Reuse poisoned storage and compare the unaudited worker with separate
    // launches. This catches readiness bugs masked by instrumentation.
    for (int repeat = 0; repeat < 2; ++repeat) {
        b.poison(); s.reset();
        megakernel<false, false, true><<<workers, threads>>>(s.graph(false));
        CHECK_CUDA(cudaGetLastError());
        if (b.raw_y.get() != y || b.raw_dq.get() != dq || b.raw_dk.get() != dk || b.raw_dv.get() != dv)
            throw std::runtime_error("production reuse mismatch");
        s.verify();
    }
    b.poison();
    dim3 grid((in.tokens + 3) / 4, in.heads);
    for (auto kind : {TaskKind::attention_forward, TaskKind::attention_dq, TaskKind::attention_dkv})
        staged_attention<<<grid, threads>>>(b.a, kind);
    CHECK_CUDA(cudaGetLastError());
    if (b.raw_y.get() != y || b.raw_dq.get() != dq || b.raw_dk.get() != dk || b.raw_dv.get() != dv)
        throw std::runtime_error("staged versus persistent mismatch");
}

void benchmark(int d, int v, int workers, bool profile = false) {
    std::vector<int> lengths(18, 896);
    lengths.push_back(16384 - 18 * 896);
    Inputs in(lengths, d, v, 384, false, 0);
    Buffers b(in);
    b.a.raw_y = b.a.raw_dq = b.a.raw_dk = b.a.raw_dv = nullptr;
    Schedule s(b.a);
    printf("timing scalar attention baseline tokens=%d qk=%d v=%d window=%d; no FA3 speed claim\n",
           in.tokens, d, v, in.window);
    dim3 grid((in.tokens + 3) / 4, in.heads);
    device_time("staged_forward_backward", [&] {
        for (auto kind : {TaskKind::attention_forward, TaskKind::attention_dq, TaskKind::attention_dkv})
            staged_attention<<<grid, threads>>>(b.a, kind);
    });
    device_time("persistent_forward_backward_plus_reset_and_sentinel", [&] {
        s.reset();
        megakernel<false, false, true><<<workers, threads>>>(s.graph(false));
    });
    CHECK_CUDA(cudaGetLastError());
    s.verify();
    if (profile) {
        s.reset();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaProfilerStart());
        megakernel<false, false, true><<<workers, threads>>>(s.graph(false));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaProfilerStop());
    }
}

int main(int argc, char **argv) {
    try {
        bool quick = argc > 1 && std::string(argv[1]) == "--quick";
        bool profile = argc > 1 && std::string(argv[1]) == "--profile";
        device_info();
        cudaDeviceProp p;
        CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
        int resident;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, megakernel<false, false, true>, threads, 0));
        printf("mixed worker resident_ctas_per_sm=%d\n", resident);
        if (profile) {
            benchmark(64, 128, p.multiProcessorCount * resident, true);
            return 0;
        }
        for (auto shape : {std::pair{64, 128}, std::pair{64, 64}, std::pair{128, 128}}) {
            validate(Inputs({1, 7, 13}, shape.first, shape.second, 8, false, 0), 1);
            validate(Inputs({1, 3, 19}, shape.first, shape.second, 5, true, 1), 7);
            validate_transform(shape.first, shape.second, shape.first == 64 && shape.second == 128,
                               shape.first == 128, 7);
            if (!quick) {
                validate(Inputs({1, 15, 65, 176}, shape.first, shape.second, 128, false, 0), p.multiProcessorCount);
                validate(Inputs({1, 3, 19}, shape.first, shape.second, 0, false, 0), 7);
                validate(Inputs({1, 127, 385}, shape.first, shape.second, 1536, true, 2), p.multiProcessorCount * 4);
                benchmark(shape.first, shape.second, p.multiProcessorCount * resident);
            }
        }
        validate_transform(64, 128, true, false, 1, false);
        puts("PASS: native attention math and mixed persistent scheduling; full model and FA3 parity remain unverified");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

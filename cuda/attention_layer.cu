#include "bench.cuh"
#include "attention_layer.cuh"
#include "attention_reference.cuh"
#include "attention_layer_reference.cuh"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <memory>
#include <random>

using namespace nano;

std::vector<int> layer_sequences(int tokens, bool paired, int limit) {
    std::vector<int> result{0};
    if (limit > 0) {
        while (result.back() < tokens) result.push_back(std::min(tokens, result.back() + limit));
    } else {
        result = {0, tokens / 4, tokens / 2, tokens};
    }
    if (paired) for (int &x : result) x *= 2;
    return result;
}

struct LayerStorage {
    int t, d, v, f, forwards;
    bool paired, shifted, auxiliary, xsa, gated, evaluation, separate;
    DeviceBuffer<bf16> input, weights, wo, dy, factors1, factors2, aux, alpha, gate;
    DeviceBuffer<bf16> projected, projected_v, q, k, value, attn_y, post_y, output;
    DeviceBuffer<bf16> dpost, attn_dy, dq, dk, dv, daux, dalpha, dgate, dx, dwq0, dwq, dwo0, dwo;
    DeviceBuffer<fp8> x8, xt8, w8, wt8, grad8, gradt8;
    DeviceBuffer<float> scalars, lse, delta, side, qpartial, opartial, gains;
    DeviceBuffer<float> raw_a_y, raw_a_dq, raw_a_dk, raw_a_dv, raw_post, raw_post_dy, raw_da, raw_dg, raw_qkv;
    std::vector<int> seq;
    DeviceBuffer<int> sequences;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> raw_fp8, raw_bf16;
    LayerStorage(int tokens, int qk, int vd, bool pair, bool offset, bool aux_enabled, bool alpha_enabled, bool gate_enabled,
                 int sequence_limit = 0, bool eval = false, bool diagnostics = true)
        : t(tokens), d(qk), v(vd), f(6 * (2 * d + v)), forwards((eval ? vd == 64 : pair) ? 2 : 1), paired(pair), shifted(offset),
          auxiliary(aux_enabled), xsa(alpha_enabled), gated(gate_enabled), evaluation(eval), separate(forwards == 2),
          input(size_t(t) * 768), weights(size_t(f) * 768),
          wo(size_t(768) * 6 * v), dy(eval ? 0 : input.n), factors1(size_t(t) * d * (pair ? 2 : 1)), factors2(factors1.n),
          aux(size_t(t) * 768), alpha(size_t(t) * 6), gate(alpha.n),
          projected(size_t(t) * (separate ? 12 * d : f)), projected_v(separate ? size_t(t) * 6 * v : 1),
          q(size_t(t) * 6 * d), k(q.n), value(size_t(t) * 6 * v), attn_y(value.n), post_y(value.n), output(input.n),
          dpost(eval ? 0 : value.n), attn_dy(dpost.n), dq(eval ? 0 : q.n), dk(dq.n), dv(dpost.n),
          daux(eval ? 0 : aux.n), dalpha(eval ? 0 : alpha.n), dgate(dalpha.n),
          dx(eval ? 0 : input.n), dwq0(eval ? 0 : weights.n), dwq(dwq0.n), dwo0(eval ? 0 : wo.n), dwo(dwo0.n),
          x8(eval ? 0 : input.n), xt8(x8.n), w8(eval ? 0 : weights.n), wt8(w8.n),
          grad8(eval ? 0 : size_t(t) * f), gradt8(grad8.n),
          scalars(6), lse(size_t(t) * 6), delta(eval ? 0 : lse.n), side(dpost.n),
          qpartial(eval ? 0 : (weights.n + projection_elements - 1) / projection_elements),
          opartial(eval ? 0 : (wo.n + projection_elements - 1) / projection_elements), gains(eval ? 0 : 3),
          raw_a_y(diagnostics ? value.n : 0), raw_a_dq(diagnostics ? dq.n : 0),
          raw_a_dk(raw_a_dq.n), raw_a_dv(diagnostics ? dv.n : 0),
          raw_post(diagnostics ? value.n : 0), raw_post_dy(diagnostics ? dpost.n : 0),
          raw_da(diagnostics ? dalpha.n : 0), raw_dg(raw_da.n), raw_qkv(diagnostics ? grad8.n : 0),
          seq(layer_sequences(t, pair, sequence_limit)), sequences(seq.size()) {
        sequences.put(seq);
        for (size_t n : {projected.n, separate ? projected_v.n : dwq0.n, separate ? dwq0.n : dx.n})
            raw_fp8.push_back(std::make_unique<DeviceBuffer<float>>(diagnostics && !eval ? n : 0));
        if (separate) raw_fp8.push_back(std::make_unique<DeviceBuffer<float>>(diagnostics && !eval ? dx.n : 0));
        for (size_t n : {output.n, dpost.n, dwo0.n, projected.n, projected_v.n})
            raw_bf16.push_back(std::make_unique<DeviceBuffer<float>>(diagnostics ? n : 0));
        std::mt19937 rng(711 + t + d + v);
        std::normal_distribution<float> normal;
        for (auto *b : {&input, &weights, &wo, &dy, &aux, &alpha, &gate}) {
            std::vector<bf16> values(b->n);
            float scale = b == &weights || b == &wo ? 0.04f : b == &dy ? 0.1f : 0.4f;
            for (auto &x : values) x = bf16(normal(rng) * scale);
            b->put(values);
        }
        std::vector<float> s = {0.015625f, 0, 0.00390625f, 0.75f, 0.875f, 1.125f};
        auto w = weights.get();
        for (auto x : w) s[1] = std::max(s[1], std::abs(float(x)));
        s[1] = std::max(s[1], 1e-12f) * (1.0f / 448.0f);
        auto quantize = [](const std::vector<bf16> &src, float scale, int rows, int cols,
                           DeviceBuffer<fp8> &row, DeviceBuffer<fp8> &transposed) {
            std::vector<fp8> a(src.size()), b(src.size());
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    b[c * rows + r] = a[r * cols + c] = fp8(float(src[r * cols + c]) / scale);
            row.put(a); transposed.put(b);
        };
        if (eval) {
            s[0] = s[1] = s[2] = NAN;
        } else {
            quantize(w, s[1], f, 768, w8, wt8);
            quantize(input.get(), s[0], t, 768, x8, xt8);
        }
        scalars.put(s);
        std::vector<bf16> f1(factors1.n), f2(factors2.n);
        int stride = d * (pair ? 2 : 1);
        for (int token = 0; token < t; ++token)
            for (int c = 0; c < stride; ++c) {
                int channel = c % d;
                double theta = (pair ? 2 * token + c / d : token) * std::pow(10000.0, -2.0 * (channel / 2) / d);
                f1[token * stride + c] = bf16(float(std::cos(theta)));
                f2[token * stride + c] = bf16(float(std::sin(theta) * (channel % 2 ? -1 : 1)));
            }
        factors1.put(f1); factors2.put(f2);
    }
    AttentionLayerBuffers buffers(int window = 31, bool diagnostics = true) {
        AttentionLayerBuffers b{};
        b.evaluation = evaluation; b.x_bf16 = evaluation ? input.p : nullptr;
        b.tokens = t; b.qk_dim = d; b.value_dim = v; b.documents = int(seq.size()) - 1; b.window = window;
        b.paired = paired; b.key_offset = shifted; b.attention_scale = d == 64 ? 0.13f : 0.085f;
        b.sequences = sequences.p; b.scalars = scalars.p;
        b.x = x8.p; b.x_transposed = xt8.p; b.weight = w8.p; b.weight_transposed = wt8.p;
        b.weight_bf16 = weights.p; b.output_weight = wo.p; b.dy = dy.p;
        b.factor1 = factors1.p; b.factor2 = factors2.p; b.aux = auxiliary ? aux.p : nullptr;
        b.alpha = xsa ? alpha.p : nullptr; b.gate = gated ? gate.p : nullptr;
        b.projected = projected.p; b.projected_v = projected_v.p; b.q = q.p; b.k = k.p; b.v = value.p;
        b.attention_y = attn_y.p; b.post_y = post_y.p; b.output = output.p;
        b.dpost = dpost.p; b.attention_dy = attn_dy.p; b.dq = dq.p; b.dk = dk.p; b.dv = dv.p;
        b.daux = auxiliary ? daux.p : nullptr; b.dalpha = dalpha.p; b.dgate = dgate.p; b.dx = dx.p;
        b.dw_qkv_unscaled = dwq0.p; b.dw_qkv = dwq.p; b.dw_o_unscaled = dwo0.p; b.dw_o = dwo.p;
        b.packed_gradient = grad8.p; b.packed_gradient_transposed = gradt8.p;
        b.lse = lse.p; b.delta = delta.p; b.value_side = side.p;
        b.qkv_partial = qpartial.p; b.o_partial = opartial.p; b.gain_gradients = gains.p;
        if (evaluation) {
            b.x = b.x_transposed = b.weight = b.weight_transposed = nullptr;
            b.dy = nullptr;
            b.dpost = b.attention_dy = b.dq = b.dk = b.dv = b.daux = b.dalpha = b.dgate = b.dx = nullptr;
            b.dw_qkv_unscaled = b.dw_qkv = b.dw_o_unscaled = b.dw_o = nullptr;
            b.packed_gradient = b.packed_gradient_transposed = nullptr;
            b.delta = b.value_side = b.qkv_partial = b.o_partial = b.gain_gradients = nullptr;
        }
        if (!diagnostics) return b;
        for (size_t i = 0; i < raw_fp8.size(); ++i) b.fp8_raw[i] = raw_fp8[i]->p;
        for (size_t i = 0; i < raw_bf16.size(); ++i) b.bf16_raw[i] = raw_bf16[i]->p;
        b.attention_raw[0] = raw_a_y.p; b.attention_raw[1] = raw_a_dq.p;
        b.attention_raw[2] = raw_a_dk.p; b.attention_raw[3] = raw_a_dv.p;
        b.post_raw[0] = raw_post.p; b.post_raw[1] = raw_post_dy.p;
        b.post_raw[2] = raw_da.p; b.post_raw[3] = raw_dg.p; b.qkv_raw_gradient = raw_qkv.p;
        return b;
    }
    std::vector<DeviceBuffer<bf16> *> outputs() {
        std::vector<DeviceBuffer<bf16> *> result = {&projected, &q, &k, &value, &attn_y, &post_y, &output};
        if (separate) result.push_back(&projected_v);
        if (!evaluation) {
            result.insert(result.end(), {&dpost, &attn_dy, &dq, &dk, &dv, &dalpha, &dgate, &dx, &dwq0, &dwq, &dwo0, &dwo});
            if (auxiliary) result.push_back(&daux);
        }
        return result;
    }
    void poison() {
        for (auto *b : outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : {&lse, &delta, &side, &qpartial, &opartial, &gains, &raw_a_y, &raw_a_dq, &raw_a_dk,
                        &raw_a_dv, &raw_post, &raw_post_dy, &raw_da, &raw_dg, &raw_qkv})
            if (b->n) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto &b : raw_fp8) if (b->n) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto &b : raw_bf16) if (b->n) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        if (grad8.n) CHECK_CUDA(cudaMemset(grad8.p, 0xff, grad8.n));
        if (gradt8.n) CHECK_CUDA(cudaMemset(gradt8.p, 0xff, gradt8.n));
    }
};

__global__ void reset_layer(Graph g, int groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g.task_count) g.queue[i] = i < g.root_count ? i + 1 : 0;
    if (i < groups) g.counters[i] = 0;
    if (i < 4) g.state[i] = i == 2 ? g.task_count : i == 3 ? 0 : g.root_count;
}

template <int MinimumBlocks>
__global__ __launch_bounds__(128, MinimumBlocks) void staged_layer(Graph g, int begin) {
    __shared__ __align__(1024) fp8 scratch[bf16_scratch_bytes];
    const Task task = g.tasks[begin + blockIdx.x];
#ifdef NANO_EVALUATION_BODY
    if (task.kind == TaskKind::evaluation_loss) {
        evaluation_loss(g.evaluation_heads[task.op], task.row);
    } else if (task.kind == TaskKind::embedding_read) {
        embedding_read(g.embeddings[task.op], task.row);
    } else if (task.kind == TaskKind::gate_transform) {
        gate_transform(g.gates[task.op], task.row);
    } else if (task.kind == TaskKind::residual_mix) {
        residual_mix_forward(g.residual_mix[task.op], task.row);
    } else if (task.kind == TaskKind::residual_norm) {
        residual_norm(g.residual_norm[task.op], task.row, false);
    } else
#endif
    if (task.kind == TaskKind::layer_setup) {
        const AttentionLayerSetup op = g.layer_setup[task.op]; setup_attention_layer(op);
    } else if (task.kind == TaskKind::attention_post) {
        const AttentionPost op = g.attention_post[task.op];
        execute_attention_post(op, AttentionPostStage(task.k_begin), task.row, task.col);
    } else if (task.kind == TaskKind::projection_gradient) {
        const ProjectionGradient op = g.projection_gradient[task.op];
        projection_gradient_tile(op, task.col != 0, task.row, reinterpret_cast<float *>(scratch));
    } else if (task.kind == TaskKind::bf16_matmul) {
        const BF16Matmul op = g.bf16_ops[task.op]; bf16_matmul_tile(op, task.row, task.col, scratch);
    } else if (task.kind == TaskKind::matmul) {
        const Matmul op = g.ops[task.op]; execute_tile<false>(op, task, scratch);
    } else if (task.kind == TaskKind::qkv_forward || task.kind == TaskKind::qkv_backward) {
        const QKVTransform op = g.qkv[task.op]; execute_qkv(op, task.kind == TaskKind::qkv_backward, task.row, task.col);
    } else {
        const Attention op = g.attention[task.op]; execute_attention(op, task.kind, task.row, task.col);
    }
}

struct LayerSchedule {
    AttentionLayerPlan plan;
    DeviceBuffer<Matmul> ops;
    DeviceBuffer<BF16Matmul> bf16_ops;
    DeviceBuffer<QKVTransform> qkv;
    DeviceBuffer<Attention> attention;
    DeviceBuffer<AttentionPost> post;
    DeviceBuffer<ProjectionGradient> gradient;
    DeviceBuffer<AttentionLayerSetup> setup;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    explicit LayerSchedule(const AttentionLayerBuffers &b) : plan(b), ops(plan.ops.size()), bf16_ops(plan.bf16_ops.size()),
        qkv(1), attention(1), post(1), gradient(plan.gradient.size()), setup(1), tasks(plan.tasks.size()), groups(plan.groups.size()),
        counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2) {
        ops.put(plan.ops); bf16_ops.put(plan.bf16_ops); qkv.put({plan.qkv}); attention.put({plan.attention});
        post.put({plan.post}); gradient.put(plan.gradient); setup.put({plan.setup(ops.p, bf16_ops.p, qkv.p)});
        tasks.put(plan.tasks); groups.put(plan.groups);
    }
    Graph graph(bool checked = false) {
        return {ops.p, tasks.p, groups.p, counters.p, queue.p, state.p, int(tasks.n), 1,
                checked ? audit.p : nullptr, attention.p, qkv.p, bf16_ops.p, nullptr, setup.p, post.p, gradient.p};
    }
    void run(int workers, bool checked, bool all_families = true) {
        reset_layer<<<(tasks.n + 127) / 128, 128>>>(graph(), int(groups.n));
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, true, true, true, true><<<workers, 128>>>(graph(true));
        } else if (all_families) {
            megakernel<false, false, true, true, true, true><<<workers, 128>>>(graph());
        } else {
            megakernel<false, false, true, true, false, true><<<workers, 128>>>(graph());
        }
        CHECK_CUDA(cudaGetLastError());
    }
    void staged() {
        for (const auto &stage : plan.stages)
            staged_layer<1><<<stage.second - stage.first, 128>>>(graph(), stage.first);
        CHECK_CUDA(cudaGetLastError());
    }
    void verify() {
        if (state.get()[2] != 0) throw std::runtime_error("unfinished layer tasks");
        auto visits = audit.get(), completed = counters.get();
        for (size_t i = 0; i < tasks.n; ++i)
            if (visits[i] != 1) throw std::runtime_error("layer task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i)
            if (completed[i] != plan.groups[i].expected) throw std::runtime_error("layer dependency mismatch");
    }
};

struct LayerGraphControl {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    template <class Schedule> explicit LayerGraphControl(Schedule &schedule, bool four_blocks = false) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t previous = nullptr;
        for (const auto &stage : schedule.plan.stages) {
            Graph descriptor = schedule.graph();
            int begin = stage.first;
            void *args[] = {&descriptor, &begin};
            cudaKernelNodeParams params{};
            params.func = four_blocks ? reinterpret_cast<void *>(staged_layer<4>) : reinterpret_cast<void *>(staged_layer<1>);
            params.gridDim = dim3(stage.second - stage.first); params.blockDim = dim3(128);
            params.kernelParams = args;
            cudaGraphNode_t node;
            CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, previous ? &previous : nullptr, previous ? 1 : 0, &params));
            previous = node;
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~LayerGraphControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

struct LayerResult {
    std::vector<std::vector<bf16>> values;
    std::vector<float> gains;
    std::vector<fp8> packed, transposed;
    explicit LayerResult(LayerStorage &b)
        : gains(b.evaluation ? std::vector<float>{} : b.gains.get()),
          packed(b.evaluation ? std::vector<fp8>{} : b.grad8.get()),
          transposed(b.evaluation ? std::vector<fp8>{} : b.gradt8.get()) {
        for (auto *buffer : b.outputs()) {
            values.push_back(buffer->get());
            for (auto x : values.back())
                if (!std::isfinite(float(x))) throw std::runtime_error("non-finite layer output");
        }
        for (float x : gains) if (!std::isfinite(x)) throw std::runtime_error("non-finite layer gain");
    }
    void compare(LayerStorage &b) const {
        auto outputs = b.outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto actual = outputs[i]->get();
            if (std::memcmp(actual.data(), values[i].data(), actual.size() * sizeof(bf16)))
                throw std::runtime_error("layer staged/production mismatch");
        }
        if (b.evaluation) return;
        auto got = b.gains.get();
        if (std::memcmp(got.data(), gains.data(), gains.size() * sizeof(float)))
            throw std::runtime_error("layer gain replay mismatch");
        auto gp = b.grad8.get(), gt = b.gradt8.get();
        if (std::memcmp(gp.data(), packed.data(), packed.size()) || std::memcmp(gt.data(), transposed.data(), transposed.size()))
            throw std::runtime_error("layer FP8 gradient replay mismatch");
    }
};

void check_layer_step(LayerStorage &b, LayerSchedule &schedule, LayerGraphControl &control,
                      LayerGraphControl &bounded_control, int workers) {
    b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify();
    check_layer_math(b, schedule);
    LayerResult expected(b);
    for (int mode = 0; mode < 5; ++mode) {
        b.poison();
        if (mode == 4) bounded_control.run();
        else if (mode == 3) control.run();
        else if (mode) schedule.run(workers, false, mode == 1);
        else schedule.staged();
        CHECK_CUDA(cudaDeviceSynchronize());
        expected.compare(b);
    }
    printf("LAYER PASS evaluation=%d tokens=%d qk=%d v=%d paired=%d xsa=%d gate=%d tasks=%zu",
           b.evaluation, b.t, b.d, b.v, b.paired, b.xsa, b.gated, schedule.tasks.n);
    if (!b.evaluation) printf(" gains=%g,%g,%g", expected.gains[0], expected.gains[1], expected.gains[2]);
    puts("");
}

void check_layer(LayerStorage &b, int workers, int steps, bool extra_output_gain = true) {
    LayerSchedule schedule(b.buffers());
    LayerGraphControl control(schedule);
    LayerGraphControl bounded_control(schedule, true);
    for (int step = 0; step < steps; ++step) {
        auto s = b.scalars.get();
        if (step == 1) { s[2] = 0.0078125f; s[3] = 1.25f; s[4] = 0; s[5] = 0.875f; }
        if (step == 2) { s[2] = 0.001953125f; s[3] = 0; s[4] = -0.5f; s[5] = 1.5f; }
        if (!extra_output_gain) s[5] = 1.0f;
        if (b.evaluation) s[0] = s[1] = s[2] = NAN;
        b.scalars.put(s);
        printf("LAYER step=%d grad_scale=%g qkv_gain=%g o_gain=%g extra_gain=%g\n", step, s[2], s[3], s[4], s[5]);
        check_layer_step(b, schedule, control, bounded_control, workers);
    }
}

__global__ void post_only(AttentionPost op, AttentionPostStage stage) {
    execute_attention_post(op, stage, blockIdx.x, blockIdx.y);
}

void check_post_floor(int width) {
    LayerStorage b(16, 64, width, false, false, true, true, true);
    auto desc = b.buffers();
    AttentionLayerPlan plan(desc);
    auto y = b.aux.get(), incoming = b.dy.get();
    y.resize(b.value.n); incoming.resize(b.value.n);
    b.attn_y.put(y); b.dpost.put(incoming);
    for (float magnitude : {0.0f, 1e-7f, 1e-4f}) {
        std::vector<bf16> v(b.value.n), alpha(b.alpha.n), gate(b.gate.n);
        for (size_t i = 0; i < v.size(); ++i) v[i] = bf16(magnitude * (int(i % 7) - 3));
        for (size_t i = 0; i < alpha.size(); ++i) {
            alpha[i] = bf16(i % 3 == 0 ? 0.75f : i % 3 == 1 ? -0.5f : 8.0f);
            gate[i] = bf16(i % 3 == 0 ? 1.25f : i % 3 == 1 ? 0.0f : -1.0f);
        }
        b.value.put(v); b.alpha.put(alpha); b.gate.put(gate);
        post_only<<<dim3(4, 6), 128>>>(plan.post, AttentionPostStage::forward);
        post_only<<<dim3(4, 6), 128>>>(plan.post, AttentionPostStage::backward);
        CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
        PostReference reference(y, v, incoming, &alpha, &gate, width);
        layer_near("post floor output", b.raw_post.get(), reference.output);
        layer_near("post floor dy", b.raw_post_dy.get(), reference.dy);
        layer_near("post floor dv", b.side.get(), reference.dv);
        layer_near("post floor alpha", b.raw_da.get(), reference.da);
        layer_near("post floor gate", b.raw_dg.get(), reference.dg);
        post_finite_difference(y, v, incoming, &alpha, &gate, width, reference);
        auto side = b.side.get();
        b.dv.put(incoming);
        CHECK_CUDA(cudaMemset(b.daux.p, 0xff, b.daux.n * sizeof(bf16)));
        post_only<<<dim3(4, 6), 128>>>(plan.post, AttentionPostStage::value_gradient);
        CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
        auto joined = b.dv.get(), aux = b.daux.get();
        for (size_t i = 0; i < joined.size(); ++i) {
            bf16 expected(float(incoming[i]) + side[i]);
            if (float(joined[i]) != float(expected) || float(aux[i / width * 128 + i % width]) != float(expected))
                throw std::runtime_error("post floor value join mismatch");
        }
        for (int row = 0; row < b.t * 6; ++row)
            for (int c = width; c < 128; ++c)
                if (float(aux[row * 128 + c]) != 0) throw std::runtime_error("post floor auxiliary padding mismatch");
        printf("POST FLOOR PASS width=%d magnitude=%g\n", width, magnitude);
    }
}

void benchmark_layer(int d, int v, bool paired, bool profile = false, bool evaluation = false, bool final_evaluation = false) {
    if (final_evaluation && !evaluation) throw std::runtime_error("final evaluation requires BF16 mode");
    int tokens = final_evaluation ? frontier::validation_local_tokens : 16384;
    int layer = d == 128 ? 3 : paired ? 2 : 1;
    const auto role = frontier::attention_role[layer];
    LayerStorage b(tokens, d, v, paired, d == 128, role.auxiliary, role.xsa, role.head_gate,
                   final_evaluation ? 4096 : 896, evaluation, false);
    int window = final_evaluation ? (d == 128 ? frontier::final_validation_long_window : frontier::final_validation_short_window) :
                                   (d == 128 ? 384 : 128);
    LayerSchedule schedule(b.buffers(window, false));
    LayerGraphControl control(schedule);
    LayerGraphControl bounded_control(schedule, true);
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    int workers = properties.multiProcessorCount * 4;
    printf("LAYER BENCH role=%d evaluation=%d tokens=%d qk=%d v=%d paired=%d window=%d documents=%zu workers=%d tasks=%zu stages=%zu\n",
           layer, evaluation, b.t, d, v, paired, window, b.seq.size() - 1, workers, schedule.tasks.n, schedule.plan.stages.size());
    if (profile) {
        for (int i = 0; i < 5; ++i) schedule.run(workers, false);
        reset_layer<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaProfilerStart());
        megakernel<false, false, true, true, true, true><<<workers, 128>>>(schedule.graph());
        CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaProfilerStop());
        return;
    }
    b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify();
    LayerResult expected(b);
    for (int mode = 0; mode < 4; ++mode) {
        b.poison();
        if (mode == 0) schedule.staged();
        else if (mode == 1) control.run();
        else if (mode == 3) bounded_control.run();
        else schedule.run(workers, false);
        CHECK_CUDA(cudaDeviceSynchronize()); expected.compare(b);
    }
    puts(evaluation ? "LAYER BF16 evaluation bitwise replay PASS; normalized input supplied externally" :
                      "LAYER training bitwise replay PASS; caches supplied externally, setup and queue reset timed");
    device_time("LAYER separate stages", [&] { schedule.staged(); });
    device_time("LAYER CUDA Graph", [&] { control.run(); });
    device_time("LAYER CUDA Graph four-block occupancy", [&] { bounded_control.run(); });
    for (int multiple : {1, 2, 4}) {
        int count = multiple * properties.multiProcessorCount;
        std::string name = "LAYER persistent+reset workers=" + std::to_string(count);
        device_time(name.c_str(), [&] { schedule.run(count, false); });
    }
}

#ifdef NANO_EVALUATION_ONLY
#include "mlp_evaluation_check.cuh"
#endif
#ifdef NANO_EVALUATION_BODY
#include "model_evaluation_check.cuh"
#endif

#ifndef NANO_EMBED_ATTENTION_DRIVER
int main(int argc, char **argv) {
    try {
        device_info();
#ifdef NANO_EVALUATION_ONLY
        constexpr bool evaluation = true;
#else
        constexpr bool evaluation = false;
#endif
        bool quick = false, profile = false, main_shapes = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--quick") quick = true;
            else if (arg == "--profile") profile = true;
            else if (arg == "--main-shapes") main_shapes = true;
            else if (arg.rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown argument: " + arg);
        }
#ifdef NANO_EVALUATION_BODY
        if (profile) return evaluation_body_profile();
        return evaluation_body_main(quick, main_shapes);
#endif
        if (profile) { benchmark_layer(128, 128, false, true, evaluation); return 0; }
        int steps = quick ? 2 : 3;
        // Layers 0 and 5 share the same attention-call configuration.
        for (int layer : {0, 1, 2, 3, 8, 10}) {
            const auto role = frontier::attention_role[layer];
            int d = frontier::qk_width[layer], v = frontier::value_width[layer];
            int tokens = layer == 1 ? 80 : layer == 0 || layer == 8 ? 16 : 32;
            int workers = layer == 1 ? 1 : layer == 10 ? 17 : 7;
            printf("LAYER CHECK reference_layer=%d\n", layer);
            LayerStorage b(tokens, d, v, role.paired, d == 128, role.auxiliary, role.xsa, role.head_gate, 0, evaluation);
            check_layer(b, workers, steps, role.extra_output_gain);
        }
        if (!evaluation) { check_post_floor(64); check_post_floor(128); }
#ifdef NANO_EVALUATION_ONLY
        check_mlp_evaluation(16, 1, steps);
        check_mlp_evaluation(80, 7, steps);
#endif
        puts("PASS: connected attention layer, independent stage references and exact replay");
        if (!quick) {
            benchmark_layer(64, 128, true, false, evaluation);
            benchmark_layer(64, 64, false, false, evaluation);
            benchmark_layer(128, 128, false, false, evaluation);
#ifdef NANO_EVALUATION_ONLY
            benchmark_mlp_evaluation(16384);
            benchmark_mlp_evaluation(frontier::validation_local_tokens);
            benchmark_layer(128, 128, false, false, true, true);
#endif
        }
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
#endif

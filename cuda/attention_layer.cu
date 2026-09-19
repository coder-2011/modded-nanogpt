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
    bool paired, shifted, auxiliary, xsa, gated;
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
                 int sequence_limit = 0)
        : t(tokens), d(qk), v(vd), f(6 * (2 * d + v)), forwards(pair ? 2 : 1), paired(pair), shifted(offset),
          auxiliary(aux_enabled), xsa(alpha_enabled), gated(gate_enabled), input(size_t(t) * 768), weights(size_t(f) * 768),
          wo(size_t(768) * 6 * v), dy(input.n), factors1(size_t(t) * d * (pair ? 2 : 1)), factors2(factors1.n),
          aux(size_t(t) * 768), alpha(size_t(t) * 6), gate(alpha.n),
          projected(size_t(t) * (pair ? 12 * d : f)), projected_v(pair ? size_t(t) * 6 * v : 1),
          q(size_t(t) * 6 * d), k(q.n), value(size_t(t) * 6 * v), attn_y(value.n), post_y(value.n), output(input.n),
          dpost(value.n), attn_dy(value.n), dq(q.n), dk(q.n), dv(value.n), daux(aux.n), dalpha(alpha.n), dgate(alpha.n),
          dx(input.n), dwq0(weights.n), dwq(weights.n), dwo0(wo.n), dwo(wo.n),
          x8(input.n), xt8(input.n), w8(weights.n), wt8(weights.n), grad8(size_t(t) * f), gradt8(grad8.n),
          scalars(6), lse(size_t(t) * 6), delta(lse.n), side(value.n),
          qpartial((weights.n + projection_elements - 1) / projection_elements),
          opartial((wo.n + projection_elements - 1) / projection_elements), gains(3),
          raw_a_y(value.n), raw_a_dq(q.n), raw_a_dk(q.n), raw_a_dv(value.n),
          raw_post(value.n), raw_post_dy(value.n), raw_da(alpha.n), raw_dg(alpha.n), raw_qkv(grad8.n),
          seq(layer_sequences(t, pair, sequence_limit)), sequences(seq.size()) {
        sequences.put(seq);
        for (size_t n : {projected.n, pair ? projected_v.n : dwq0.n, pair ? dwq0.n : dx.n})
            raw_fp8.push_back(std::make_unique<DeviceBuffer<float>>(n));
        if (pair) raw_fp8.push_back(std::make_unique<DeviceBuffer<float>>(dx.n));
        for (size_t n : {output.n, dpost.n, dwo0.n}) raw_bf16.push_back(std::make_unique<DeviceBuffer<float>>(n));
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
        scalars.put(s);
        auto quantize = [](const std::vector<bf16> &src, float scale, int rows, int cols,
                           DeviceBuffer<fp8> &row, DeviceBuffer<fp8> &transposed) {
            std::vector<fp8> a(src.size()), b(src.size());
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    b[c * rows + r] = a[r * cols + c] = fp8(float(src[r * cols + c]) / scale);
            row.put(a); transposed.put(b);
        };
        quantize(w, s[1], f, 768, w8, wt8);
        quantize(input.get(), s[0], t, 768, x8, xt8);
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
        std::vector<DeviceBuffer<bf16> *> result = {&projected, &q, &k, &value, &attn_y, &post_y, &output,
            &dpost, &attn_dy, &dq, &dk, &dv, &dalpha, &dgate, &dx, &dwq0, &dwq, &dwo0, &dwo};
        if (paired) result.push_back(&projected_v);
        if (auxiliary) result.push_back(&daux);
        return result;
    }
    void poison() {
        for (auto *b : outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : {&lse, &delta, &side, &qpartial, &opartial, &gains, &raw_a_y, &raw_a_dq, &raw_a_dk,
                        &raw_a_dv, &raw_post, &raw_post_dy, &raw_da, &raw_dg, &raw_qkv})
            CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto &b : raw_fp8) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto &b : raw_bf16) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        CHECK_CUDA(cudaMemset(grad8.p, 0xff, grad8.n)); CHECK_CUDA(cudaMemset(gradt8.p, 0xff, gradt8.n));
    }
};

__global__ void reset_layer(Graph g, int groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g.task_count) g.queue[i] = i == 0 ? 1 : 0;
    if (i < groups) g.counters[i] = 0;
    if (i < 4) g.state[i] = i == 2 ? g.task_count : i == 3 ? 0 : 1;
}

template <int MinimumBlocks>
__global__ __launch_bounds__(128, MinimumBlocks) void staged_layer(Graph g, int begin) {
    __shared__ __align__(1024) fp8 scratch[bf16_scratch_bytes];
    const Task task = g.tasks[begin + blockIdx.x];
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
        qkv(1), attention(1), post(1), gradient(2), setup(1), tasks(plan.tasks.size()), groups(plan.groups.size()),
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
    explicit LayerGraphControl(LayerSchedule &schedule, bool four_blocks = false) {
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
    explicit LayerResult(LayerStorage &b) : gains(b.gains.get()), packed(b.grad8.get()), transposed(b.gradt8.get()) {
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
    printf("LAYER PASS tokens=%d qk=%d v=%d paired=%d xsa=%d gate=%d tasks=%zu gains=%g,%g,%g\n",
           b.t, b.d, b.v, b.paired, b.xsa, b.gated, schedule.tasks.n,
           expected.gains[0], expected.gains[1], expected.gains[2]);
}

void check_layer(LayerStorage &b, int workers, int steps) {
    LayerSchedule schedule(b.buffers());
    LayerGraphControl control(schedule);
    LayerGraphControl bounded_control(schedule, true);
    for (int step = 0; step < steps; ++step) {
        auto s = b.scalars.get();
        if (step == 1) { s[2] = 0.0078125f; s[3] = 1.25f; s[4] = 0; s[5] = 0.875f; }
        if (step == 2) { s[2] = 0.001953125f; s[3] = 0; s[4] = -0.5f; s[5] = 1.5f; }
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

void benchmark_layer(int d, int v, bool paired, bool profile = false) {
    LayerStorage b(16384, d, v, paired, d == 128, true, !paired, d == 128, 896);
    int window = d == 128 ? 384 : 128;
    LayerSchedule schedule(b.buffers(window, false));
    LayerGraphControl control(schedule);
    LayerGraphControl bounded_control(schedule, true);
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    int workers = properties.multiProcessorCount * 4;
    printf("LAYER BENCH tokens=%d qk=%d v=%d paired=%d window=%d documents=%zu workers=%d tasks=%zu stages=%zu\n",
           b.t, d, v, paired, window, b.seq.size() - 1, workers, schedule.tasks.n, schedule.plan.stages.size());
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
    puts("LAYER main-shape bitwise replay PASS; caches supplied externally, setup and queue reset timed");
    device_time("LAYER separate stages", [&] { schedule.staged(); });
    device_time("LAYER CUDA Graph", [&] { control.run(); });
    device_time("LAYER CUDA Graph four-block occupancy", [&] { bounded_control.run(); });
    for (int multiple : {1, 2, 4}) {
        int count = multiple * properties.multiProcessorCount;
        std::string name = "LAYER persistent+reset workers=" + std::to_string(count);
        device_time(name.c_str(), [&] { schedule.run(count, false); });
    }
}

int main(int argc, char **argv) {
    try {
        device_info();
        bool quick = false, profile = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--quick") quick = true;
            else if (arg == "--profile") profile = true;
            else if (arg.rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown argument: " + arg);
        }
        if (profile) { benchmark_layer(128, 128, false, true); return 0; }
        int steps = quick ? 2 : 3;
        LayerStorage narrow(16, 64, 64, false, false, false, false, false);
        check_layer(narrow, 7, steps);
        LayerStorage paired(32, 64, 128, true, false, true, false, false);
        check_layer(paired, 3, steps);
        LayerStorage wide(32, 128, 128, false, true, true, true, true);
        check_layer(wide, 7, steps);
        LayerStorage half(80, 64, 64, false, false, true, true, false);
        check_layer(half, 1, steps);
        if (!quick) {
            LayerStorage last(32, 128, 128, false, true, false, false, true);
            check_layer(last, 17, steps);
        }
        check_post_floor(64); check_post_floor(128);
        puts("PASS: connected attention layer, independent stage references and exact replay");
        if (!quick) {
            benchmark_layer(64, 128, true);
            benchmark_layer(64, 64, false);
            benchmark_layer(128, 128, false);
        }
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}

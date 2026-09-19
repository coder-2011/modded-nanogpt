#define NANO_TRAINING_TAIL
#include "head.cu"
#include "attention_layer_reference.cuh"
#include "tail_reference.h"
#include <memory>

struct TailStorage {
    HeadStorage head;
    DeviceBuffer<bf16> x, w1, w2, bias, wg, pre, hidden, product_mu, product_group, mu, mug, mixed;
    DeviceBuffer<bf16> dmixed, dx_direct, dx_network, dx, dmu, dmug, dh_mu, dh_group, dpre, dw1, dw2, dbias, dwg;
    DeviceBuffer<float> raw_mix, raw_norm, raw_dnorm, raw_dpre, raw_hidden, raw_mu, raw_mug;
    std::array<std::unique_ptr<DeviceBuffer<bf16>>, 10> sources, dsources;
    std::array<std::unique_ptr<DeviceBuffer<float>>, 10> coefficient_rows;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> raw_products;
    explicit TailStorage(int t, int v) : head(t, v, 3), x(size_t(t) * 768), w1(64 * 768), w2(14 * 64), bias(14), wg(14 * 12 * 64),
        pre(size_t(t) * 64), hidden(pre.n), product_mu(size_t(t) * 10), product_group(size_t(t) * 120), mu(product_mu.n), mug(product_group.n), mixed(x.n),
        dmixed(x.n), dx_direct(x.n), dx_network(x.n), dx(x.n), dmu(mu.n), dmug(mug.n), dh_mu(pre.n), dh_group(pre.n), dpre(pre.n),
        dw1(w1.n), dw2(w2.n), dbias(bias.n), dwg(wg.n), raw_mix(x.n), raw_norm(x.n), raw_dnorm(x.n), raw_dpre(pre.n),
        raw_hidden(hidden.n), raw_mu(mu.n), raw_mug(mug.n) {
        for (int i = 0; i < 10; ++i) {
            sources[i] = std::make_unique<DeviceBuffer<bf16>>(x.n);
            dsources[i] = std::make_unique<DeviceBuffer<bf16>>(x.n);
            coefficient_rows[i] = std::make_unique<DeviceBuffer<float>>(size_t(t) * 12);
        }
        for (const auto &op : products(false)) raw_products.push_back(std::make_unique<DeviceBuffer<float>>(size_t(op.m) * op.n));
    }
    void change(int step) {
        head.change(step == 1 ? 0 : step);
        std::mt19937 rng(8237 + step); std::normal_distribution<float> normal;
        for (auto *buffer : {&x, &w1, &w2, &wg, &bias}) {
            std::vector<bf16> values(buffer->n);
            for (auto &v : values) v = bf16(normal(rng) * (buffer == &x ? 0.2f : 0.05f));
            if ((step == 1 && (buffer == &w2 || buffer == &wg)) || (step == 2 && buffer == &x)) std::fill(values.begin(), values.end(), bf16(0.0f));
            if (buffer == &bias) values[1] = bf16(-5.0f);
            buffer->put(values);
        }
        for (auto &buffer : sources) {
            std::vector<bf16> values(buffer->n);
            for (auto &v : values) v = bf16(step == 2 ? 0.0f : normal(rng) * 0.2f);
            buffer->put(values);
        }
    }
    std::vector<BF16Matmul> products(bool diagnostics = true) {
        int t = head.loss.tokens;
        std::vector<BF16Matmul> result{
            {x.p, w1.p, nullptr, pre.p, nullptr, t, 64, 768, 768, 1, 768, 1},
            {hidden.p, w2.p, nullptr, product_mu.p, nullptr, t, 10, 64, 64, 1, 64, 1},
            {hidden.p, wg.p, nullptr, product_group.p, nullptr, t, 120, 64, 64, 1, 64, 1},
            {dmu.p, w2.p, nullptr, dh_mu.p, nullptr, t, 64, 10, 10, 1, 1, 64},
            {dmug.p, wg.p, nullptr, dh_group.p, nullptr, t, 64, 120, 120, 1, 1, 64},
            {dpre.p, w1.p, nullptr, dx_network.p, nullptr, t, 768, 64, 64, 1, 1, 768},
            {dpre.p, x.p, nullptr, dw1.p, nullptr, 64, 768, t, 1, 64, 1, 768},
            {dmu.p, hidden.p, nullptr, dw2.p, nullptr, 10, 64, t, 1, 10, 1, 64},
            {dmug.p, hidden.p, nullptr, dwg.p, nullptr, 120, 64, t, 1, 120, 1, 64}};
        if (diagnostics) for (size_t i = 0; i < result.size(); ++i) result[i].raw = raw_products[i]->p;
        return result;
    }
    std::vector<GateTransform> gates(bool diagnostics = true) {
        int t = head.loss.tokens;
        std::vector<GateTransform> result{{GateKind::gelu, pre.p, nullptr, nullptr, hidden.p, t, 64, 64},
                {GateKind::affine, product_mu.p, bias.p, nullptr, mu.p, t, 10, 10, nullptr, 0.1f},
                {GateKind::affine, product_group.p, nullptr, nullptr, mug.p, t, 120, 120, nullptr, 0.1f}};
        if (diagnostics) { result[0].raw = raw_hidden.p; result[1].raw = raw_mu.p; result[2].raw = raw_mug.p; }
        return result;
    }
    ResidualMix mix(bool diagnostics = true) {
        ResidualMix op{}; op.tokens = head.loss.tokens; op.count = 11; op.output = mixed.p; op.dy = dmixed.p;
        op.raw = diagnostics ? raw_mix.p : nullptr;
        op.terms[0].value = x.p; op.terms[0].constant = 1; op.terms[0].dvalue = dx_direct.p;
        for (int i = 0; i < 10; ++i) {
            auto &term = op.terms[i + 1];
            term.value = sources[i]->p; term.coefficient = mu.p + i; term.coefficient_row_stride = 10;
            term.group_delta = mug.p + i * 12; term.delta_row_stride = 120; term.groups = 12;
            term.dvalue = dsources[i]->p; term.dcoefficient_rows = coefficient_rows[i]->p;
        }
        return op;
    }
    ResidualNorm norm(bool diagnostics = true) {
        return {mixed.p, head.dx.p, head.input.p, dmixed.p, head.loss.tokens, nullptr,
                diagnostics ? raw_norm.p : nullptr, diagnostics ? raw_dnorm.p : nullptr};
    }
    TailBackward backward(bool diagnostics = true) {
        TailBackward op{};
        for (int i = 0; i < 10; ++i) op.coefficient_rows[i] = coefficient_rows[i]->p;
        op.dmu = dmu.p; op.dmug = dmug.p; op.pre = pre.p; op.dh_mu = dh_mu.p; op.dh_group = dh_group.p;
        op.dpre = dpre.p; op.dbias = dbias.p; op.dw2 = dw2.p; op.dwg = dwg.p;
        op.dx_direct = dx_direct.p; op.dx_network = dx_network.p; op.dx = dx.p; op.tokens = head.loss.tokens;
        op.raw_dpre = diagnostics ? raw_dpre.p : nullptr; return op;
    }
    std::vector<DeviceBuffer<bf16> *> outputs() {
        std::vector<DeviceBuffer<bf16> *> result{&pre, &hidden, &product_mu, &product_group, &mu, &mug, &mixed, &dmixed, &dx_direct,
            &dx_network, &dx, &dmu, &dmug, &dh_mu, &dh_group, &dpre, &dw1, &dw2, &dbias, &dwg};
        for (auto &buffer : dsources) result.push_back(buffer.get());
        return result;
    }
    void poison() {
        head.poison();
        for (auto *buffer : outputs()) CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(bf16)));
        for (auto *buffer : {&raw_mix, &raw_norm, &raw_dnorm, &raw_dpre, &raw_hidden, &raw_mu, &raw_mug}) CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(float)));
        for (auto &buffer : raw_products) CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(float)));
        for (auto &buffer : coefficient_rows) CHECK_CUDA(cudaMemset(buffer->p, 0xff, buffer->n * sizeof(float)));
    }
    TailReferenceView reference() {
        TailReferenceView v{}; v.tokens = head.loss.tokens; v.x = x.p; v.w1 = w1.p; v.w2 = w2.p; v.bias = bias.p; v.wg = wg.p;
        v.dy = head.dx.p; v.output = head.input.p; v.dx = dx.p; v.dw1 = dw1.p; v.dw2 = dw2.p; v.dbias = dbias.p; v.dwg = dwg.p;
        v.mu = mu.p; v.mug = mug.p; v.mixed = mixed.p;
        for (int i = 0; i < 10; ++i) { v.sources[i] = sources[i]->p; v.dsources[i] = dsources[i]->p; }
        return v;
    }
};

struct TailPlan {
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    std::vector<int> predecessor;
    explicit TailPlan(TailStorage &b) {
        int t = b.head.loss.tokens, rows = (t + 3) / 4;
        auto phase = [&](TaskKind kind, int op, int nr, int nc = 1, int backward = 0, int column = -1) {
            int begin = int(tasks.size());
            for (int r = 0; r < nr; ++r) for (int c = 0; c < nc; ++c)
                tasks.push_back({op, r, column < 0 ? c : column, {-1, -1}, backward, 0, kind});
            predecessor.push_back(int(stages.size()) - 1); stages.push_back({begin, int(tasks.size())});
        };
        auto matrix = [&](int id) { auto op = b.products(false)[id]; phase(TaskKind::bf16_matmul, id, (op.m + 63) / 64, (op.n + 63) / 64); };
        matrix(0); phase(TaskKind::gate_transform, 0, rows);
        matrix(1); phase(TaskKind::gate_transform, 1, rows);
        matrix(2); phase(TaskKind::gate_transform, 2, rows);
        phase(TaskKind::residual_mix, 0, rows); phase(TaskKind::residual_norm, 0, rows);
        auto connect = [&](int from, int begin, int end) {
            int id = int(groups.size()); auto range = stages[from];
            groups.push_back({range.second - range.first, begin, end});
            for (int i = range.first; i < range.second; ++i) tasks[i].signal[0] = id;
        };
        for (int i = 0; i < 7; ++i) connect(i, stages[i + 1].first, stages[i + 1].second);
        HeadPlan head(b.head); int base = int(tasks.size());
        connect(7, base, base + 1); int group_base = int(groups.size());
        for (auto task : head.tasks) {
            for (int &signal : task.signal) if (signal >= 0) signal += group_base;
            tasks.push_back(task);
        }
        for (auto group : head.groups) { group.begin += base; group.end += base; groups.push_back(group); }
        for (auto [begin, end] : head.stages) { predecessor.push_back(int(stages.size()) - 1); stages.push_back({base + begin, base + end}); }
        predecessor[15] = 13;
        phase(TaskKind::residual_norm, 0, rows, 1, 1); predecessor[16] = 14;
        phase(TaskKind::residual_mix, 0, rows, 11, 1);
        phase(TaskKind::tail_backward, 0, rows, 1, 0, int(TailStep::coefficients));
        matrix(3); matrix(4);
        phase(TaskKind::tail_backward, 0, rows, 1, 0, int(TailStep::gelu));
        matrix(5); matrix(6); matrix(7); matrix(8);
        phase(TaskKind::tail_backward, 0, 1, 1, 0, int(TailStep::bias));
        phase(TaskKind::tail_backward, 0, rows, 1, 0, int(TailStep::input_sum));
        connect(14, stages[16].first, stages[16].second);
        for (int i = 16; i + 1 < int(stages.size()); ++i) connect(i, stages[i + 1].first, stages[i + 1].second);
    }
};

struct TailSchedule {
    HeadSchedule head;
    TailPlan plan;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    DeviceBuffer<BF16Matmul> products;
    DeviceBuffer<GateTransform> gates;
    DeviceBuffer<ResidualMix> mixes;
    DeviceBuffer<ResidualNorm> norms;
    DeviceBuffer<TailBackward> backward;
    explicit TailSchedule(TailStorage &b) : head(b.head), plan(b), tasks(plan.tasks.size()), groups(plan.groups.size()), counters(groups.n), queue(tasks.n),
        state(4), audit(tasks.n + 2), products(9), gates(3), mixes(1), norms(1), backward(1) {
        tasks.put(plan.tasks); groups.put(plan.groups); configure(b, true);
    }
    void configure(TailStorage &b, bool diagnostics) {
        head.products.put(b.head.products(diagnostics)); head.losses.put({b.head.loss_descriptor(diagnostics)});
        products.put(b.products(diagnostics)); gates.put(b.gates(diagnostics)); mixes.put({b.mix(diagnostics)});
        norms.put({b.norm(diagnostics)}); backward.put({b.backward(diagnostics)});
    }
    Graph graph(bool checked = false) {
        Graph g = head.graph(); g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p; g.queue = queue.p;
        g.state = state.p; g.task_count = int(tasks.n); g.root_count = plan.stages[0].second;
        g.audit = checked ? audit.p : nullptr; g.bf16_ops = products.p; g.gates = gates.p;
        g.residual_mix = mixes.p; g.residual_norm = norms.p; g.tail_backward_ops = backward.p; return g;
    }
    void run(int workers, bool checked = false) {
        reset_loss<<<(tasks.n + 127) / 128, 128>>>(graph(), int(groups.n));
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, false, true, false, false, true, true><<<workers, 128>>>(graph(true));
        } else megakernel<false, false, false, true, false, false, true, true><<<workers, 128>>>(graph());
        CHECK_CUDA(cudaGetLastError());
    }
    void verify() {
        auto visits = audit.get(), completed = counters.get();
        if (state.get()[2]) throw std::runtime_error("unfinished tail tasks");
        for (size_t i = 0; i < tasks.n; ++i) if (visits[i] != 1) throw std::runtime_error("tail task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i) if (completed[i] != plan.groups[i].expected) throw std::runtime_error("tail dependency mismatch");
    }
};

template <int MinimumBlocks>
__global__ __launch_bounds__(128, MinimumBlocks) void tail_stage(Graph g, int begin) {
    __shared__ __align__(1024) fp8 scratch[bf16_scratch_bytes];
    Task task = g.tasks[begin + blockIdx.x];
    if (task.kind == TaskKind::tail_backward) tail_backward(g.tail_backward_ops[task.op], task.row, TailStep(task.col));
    else if (task.kind == TaskKind::bf16_matmul) bf16_matmul_tile(g.bf16_ops[task.op], task.row, task.col, scratch);
    else if (task.kind == TaskKind::gate_transform) gate_transform(g.gates[task.op], task.row);
    else if (task.kind == TaskKind::residual_norm) residual_norm(g.residual_norm[task.op], task.row, task.k_begin != 0);
    else if (task.kind == TaskKind::residual_mix) {
        if (task.k_begin) residual_mix_backward(g.residual_mix[task.op], task.row, task.col);
        else residual_mix_forward(g.residual_mix[task.op], task.row);
    } else if (task.kind == TaskKind::head_setup) head_setup(g.head_setups[task.op]);
    else if (task.kind == TaskKind::head_input) head_input(g.head_inputs[task.op], task.row, task.col, scratch);
    else if (task.kind == TaskKind::loss_partial) training_loss_partial(g.training_losses[task.op], task.row, task.col);
    else if (task.kind == TaskKind::loss_reduce) training_loss_reduce(g.training_losses[task.op], task.row);
    else if (task.kind == TaskKind::loss_gradient) training_loss_gradient(g.training_losses[task.op], task.row, task.col);
    else execute_tile<false, true>(g.ops[task.op], task, scratch);
}

struct TailControl {
    cudaGraph_t graph; cudaGraphExec_t exec;
    TailControl(TailSchedule &schedule, bool bounded) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0)); std::vector<cudaGraphNode_t> nodes;
        for (size_t i = 0; i < schedule.plan.stages.size(); ++i) {
            auto [begin, end] = schedule.plan.stages[i]; Graph g = schedule.graph(); void *args[] = {&g, &begin};
            cudaKernelNodeParams p{}; p.func = bounded ? reinterpret_cast<void *>(tail_stage<4>) : reinterpret_cast<void *>(tail_stage<1>);
            p.gridDim = dim3(end - begin); p.blockDim = dim3(128); p.kernelParams = args;
            int previous = schedule.plan.predecessor[i]; cudaGraphNode_t node;
            CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, previous >= 0 ? &nodes[previous] : nullptr, previous >= 0 ? 1 : 0, &p)); nodes.push_back(node);
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~TailControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

void check_tail_math(TailStorage &b, TailSchedule &schedule) {
    check_head_math(b.head, schedule.head);
    LayerBlasReference blas;
    for (auto op : schedule.products.get()) blas.check_product(op);
    for (const auto &op : b.gates()) {
        auto input = layer_read(op.input, size_t(op.tokens) * op.columns);
        auto bias = op.bias ? layer_read(op.bias, op.columns) : std::vector<bf16>{};
        auto raw = layer_read(op.raw, input.size()); auto out = layer_read(op.output, input.size());
        std::vector<double> expected(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            double x = float(input[i]);
            expected[i] = op.kind == GateKind::gelu ? 0.5 * x * (1 + std::erf(x / std::sqrt(2.0))) :
                (x + (bias.empty() ? 0.0 : double(float(bias[i % op.columns])))) * double(op.scale);
            if (float(out[i]) != float(bf16(raw[i]))) throw std::runtime_error("tail forward gate rounding mismatch");
        }
        layer_near("tail forward gate", raw, expected);
    }
    auto x = b.x.get(), mu = b.mu.get(), mug = b.mug.get(), mixed = b.mixed.get(), dm = b.dmixed.get();
    auto normalized = b.head.input.get(), dy = b.head.dx.get();
    std::vector<double> expected_mix(x.size()), expected_norm(x.size()), expected_dnorm(x.size());
    for (size_t i = 0; i < x.size(); ++i) expected_mix[i] = float(x[i]);
    for (int term = 0; term < 10; ++term) {
        auto source = b.sources[term]->get(), ds = b.dsources[term]->get();
        std::vector<double> dc(size_t(b.head.loss.tokens) * 12, 0);
        for (int row = 0; row < b.head.loss.tokens; ++row) for (int c = 0; c < 768; ++c) {
            int group = c / 64; size_t i = size_t(row) * 768 + c;
            double weight = float(mu[row * 10 + term]) + double(float(mug[row * 120 + term * 12 + group]));
            expected_mix[i] += weight * float(source[i]); dc[row * 12 + group] += double(float(dm[i])) * float(source[i]);
            if (float(ds[i]) != float(bf16(float(weight) * float(dm[i])))) throw std::runtime_error("tail source gradient mismatch");
        }
        layer_near("tail coefficient adjoints", b.coefficient_rows[term]->get(), dc);
    }
    for (int row = 0; row < b.head.loss.tokens; ++row) {
        double squares = 0, dot = 0;
        for (int c = 0; c < 768; ++c) { size_t i = size_t(row) * 768 + c; squares += double(float(mixed[i])) * float(mixed[i]); dot += double(float(mixed[i])) * float(dy[i]); }
        double rstd = 1 / std::sqrt(squares / 768 + 0x1p-23);
        for (int c = 0; c < 768; ++c) {
            size_t i = size_t(row) * 768 + c;
            expected_norm[i] = float(mixed[i]) * rstd;
            expected_dnorm[i] = rstd * (float(dy[i]) - float(mixed[i]) * dot / 768 * rstd * rstd);
        }
    }
    layer_near("tail mixing", b.raw_mix.get(), expected_mix); layer_near("tail RMS", b.raw_norm.get(), expected_norm);
    layer_near("tail RMS backward", b.raw_dnorm.get(), expected_dnorm);
    auto pre = b.pre.get(), h1 = b.dh_mu.get(), h2 = b.dh_group.get(); auto raw = b.raw_dpre.get(); auto dpre = b.dpre.get();
    std::vector<double> expected_pre(pre.size());
    for (size_t i = 0; i < pre.size(); ++i) {
        double z = float(pre[i]), dh = float(bf16(float(h1[i]) + float(h2[i])));
        expected_pre[i] = dh * (0.5 * (1 + std::erf(z / std::sqrt(2.0))) + z * std::exp(-z * z / 2) / std::sqrt(2 * 3.141592653589793));
        if (float(dpre[i]) != float(bf16(raw[i]))) throw std::runtime_error("tail GELU gradient rounding mismatch");
    }
    layer_near("tail shared GELU gradient", raw, expected_pre);
    auto dmu = b.dmu.get(), dmug = b.dmug.get(), dbias = b.dbias.get(), dw2 = b.dw2.get(), dwg = b.dwg.get();
    for (int term = 0; term < 10; ++term) {
        auto dc = b.coefficient_rows[term]->get(); double bias_sum = 0;
        for (int row = 0; row < b.head.loss.tokens; ++row) {
            double sum = 0;
            for (int group = 0; group < 12; ++group) {
                float rounded = float(bf16(dc[row * 12 + group])); sum += rounded;
                if (float(dmug[row * 120 + term * 12 + group]) != float(bf16(rounded * 0.1f))) throw std::runtime_error("tail group gradient cast mismatch");
            }
            if (float(dmu[row * 10 + term]) != float(bf16(float(bf16(float(sum))) * 0.1f))) throw std::runtime_error("tail scalar gradient reduction mismatch");
            bias_sum += float(dmu[row * 10 + term]);
        }
        if (float(dbias[term]) != float(bf16(float(bias_sum)))) throw std::runtime_error("tail bias gradient mismatch");
    }
    for (size_t i = 10; i < dbias.size(); ++i) if (float(dbias[i]) != 0) throw std::runtime_error("tail unused bias gradient is not zero");
    for (size_t i = 10 * 64; i < dw2.size(); ++i) if (float(dw2[i]) != 0) throw std::runtime_error("tail unused weight gradient is not zero");
    for (size_t i = 120 * 64; i < dwg.size(); ++i) if (float(dwg[i]) != 0) throw std::runtime_error("tail unused group weight gradient is not zero");
    auto direct = b.dx_direct.get(), network = b.dx_network.get(), dx = b.dx.get();
    for (size_t i = 0; i < dx.size(); ++i)
        if (float(direct[i]) != float(dm[i]) || float(dx[i]) != float(bf16(float(direct[i]) + float(network[i])))) throw std::runtime_error("tail input gradient fan-in mismatch");
}

void check_tail(int tokens, int vocabulary, int workers, int steps, bool timing = false, bool aten = true) {
    TailStorage b(tokens, vocabulary); TailSchedule schedule(b); TailControl control(schedule, false), bounded(schedule, true);
    for (int step = 0; step < steps; ++step) {
        b.change(step); b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify(); check_tail_math(b, schedule);
        if (aten) check_tail_aten(b.reference());
        std::vector<std::vector<bf16>> expected;
        for (auto *buffer : b.outputs()) expected.push_back(buffer->get());
        auto loss = b.head.loss.loss.get(); auto head_dw = b.head.dw.get();
        for (int mode = 0; mode < 4; ++mode) {
            b.poison(); schedule.configure(b, mode != 3);
            if (!mode) control.run(); else if (mode == 1) bounded.run(); else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize()); size_t index = 0;
            for (auto *buffer : b.outputs()) {
                auto actual = buffer->get(); auto &ref = expected[index++];
                if (std::memcmp(actual.data(), ref.data(), actual.size() * sizeof(bf16))) throw std::runtime_error("tail output replay mismatch");
            }
            auto actual_loss = b.head.loss.loss.get(); auto actual_dw = b.head.dw.get();
            if (std::memcmp(actual_loss.data(), loss.data(), loss.size() * sizeof(float)) || std::memcmp(actual_dw.data(), head_dw.data(), head_dw.size() * sizeof(bf16)))
                throw std::runtime_error("tail head replay mismatch");
        }
        schedule.configure(b, true);
        printf("TAIL PASS tokens=%d vocabulary=%d workers=%d step=%d tasks=%zu phases=%zu\n", tokens, vocabulary, workers, step, schedule.tasks.n, schedule.plan.stages.size());
    }
    if (timing) {
        schedule.configure(b, false);
        device_time("TAIL CUDA Graph", [&] { control.run(); });
        device_time("TAIL CUDA Graph four-block occupancy", [&] { bounded.run(); });
        device_time("TAIL persistent+reset", [&] { schedule.run(workers); });
    }
}

int main(int argc, char **argv) {
    try {
        device_info(); bool quick = false, profile = false, aten = true;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--quick") quick = true;
            else if (std::string(argv[i]) == "--profile") profile = true;
            else if (std::string(argv[i]) == "--native-only") aten = false;
            else if (std::string(argv[i]).rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown tail argument");
        }
        cudaDeviceProp properties{}; CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        if (profile) {
            TailStorage b(256, 50304); b.change(0); TailSchedule schedule(b); schedule.configure(b, false);
            for (int i = 0; i < 5; ++i) schedule.run(properties.multiProcessorCount * 4);
            reset_loss<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, false, true, false, false, true, true><<<properties.multiProcessorCount * 4, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop()); return 0;
        }
        if (!aten) puts("Native-only checks: ATen diagnostic excluded, independent arithmetic and replay gates remain enabled");
        check_tail(16, 2048, 17, 3, false, aten);
        if (!quick) {
            check_tail(65, 2048, 17, 1, false, aten); check_tail(16, 50304, 17, 1, false, aten);
            check_tail(256, 50304, properties.multiProcessorCount * 4, 1, true, aten);
        }
        puts("PASS: connected post-loop MUDD, normalization and head backward; eager ATen comparisons are diagnostic, compiled parity and full training remain"); return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

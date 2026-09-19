#define NANO_TRAINING_LAST_LAYER
#include "suffix.cu"
#define NANO_EMBED_ATTENTION_DRIVER
#include "attention_layer.cu"

struct LastLayerStorage {
    SuffixStorage suffix;
    LayerStorage attention;
    DeviceBuffer<bf16> w1, w2, bias, pre, hidden, product, mu, before, bigram;
    DeviceBuffer<bf16> dmu, dh, dpre, dx_network, dw1, dw2, dbias, shared_dy, shared_dx;
    std::vector<std::unique_ptr<DeviceBuffer<bf16>>> partial, total;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> coefficient, raw_products, raw_mix;
    DeviceBuffer<float> raw_norm, raw_dnorm, raw_hidden, raw_mu, raw_dpre;
    LastLayerStorage(int t, int v) : suffix(t, v), attention(t, 128, 128, false, true, true, false, true),
        w1(64 * 768), w2(14 * 64), bias(14), pre(size_t(t) * 64), hidden(pre.n), product(size_t(t) * 14), mu(product.n),
        before(size_t(t) * 768), bigram(before.n), dmu(mu.n), dh(pre.n), dpre(pre.n), dx_network(before.n),
        dw1(w1.n), dw2(w2.n), dbias(14), shared_dy(before.n), shared_dx(before.n), raw_norm(before.n), raw_dnorm(before.n),
        raw_hidden(hidden.n), raw_mu(mu.n), raw_dpre(pre.n) {
        for (int i = 0; i < 11; ++i) {
            partial.push_back(std::make_unique<DeviceBuffer<bf16>>(before.n));
            coefficient.push_back(std::make_unique<DeviceBuffer<float>>(size_t(t) * (i == 6 ? 2 : 1)));
        }
        for (int i = 0; i < 4; ++i) total.push_back(std::make_unique<DeviceBuffer<bf16>>(before.n));
        for (int i = 0; i < 3; ++i) raw_mix.push_back(std::make_unique<DeviceBuffer<float>>(before.n));
        for (auto op : products(false)) raw_products.push_back(std::make_unique<DeviceBuffer<float>>(size_t(op.m) * op.n));
    }
    int tokens() const { return attention.t; }
    bf16 *source(int i) { return suffix.tail.sources[i]->p; }
    void change(int step) {
        suffix.change(step); std::mt19937 rng(10731 + step); std::normal_distribution<float> normal;
        for (auto *b : {&w1, &w2, &bias, &bigram, &attention.gate}) {
            std::vector<bf16> values(b->n);
            for (auto &v : values) v = bf16(normal(rng) * (b == &bigram ? 0.2f : 0.05f));
            if (step == 1 && b == &w2) std::fill(values.begin(), values.end(), bf16(0));
            if (b == &bias) { values[8] = bf16(10); values[9] = bf16(5); values[12] = bf16(10); values[13] = bf16(5); }
            b->put(values);
        }
        auto scales = attention.scalars.get(); scales[0] = 0.0625f; scales[2] = 0.015625f;
        scales[3] = step == 2 ? 0 : step == 3 ? -0.5f : 0.75f;
        scales[4] = step == 1 ? 0 : step == 3 ? -0.875f : 0.875f; scales[5] = 1;
        attention.scalars.put(scales);
    }
    std::vector<BF16Matmul> products(bool diagnostics) {
        int t = tokens();
        std::vector<BF16Matmul> ops{
            {source(2), w1.p, nullptr, pre.p, nullptr, t, 64, 768, 768, 1, 768, 1},
            {hidden.p, w2.p, nullptr, product.p, nullptr, t, 14, 64, 64, 1, 64, 1},
            {dmu.p, w2.p, nullptr, dh.p, nullptr, t, 64, 14, 14, 1, 1, 64},
            {dpre.p, w1.p, nullptr, dx_network.p, nullptr, t, 768, 64, 64, 1, 1, 768},
            {dpre.p, source(2), nullptr, dw1.p, nullptr, 64, 768, t, 1, 64, 1, 768},
            {dmu.p, hidden.p, nullptr, dw2.p, nullptr, 14, 64, t, 1, 14, 1, 64}};
        if (diagnostics) for (size_t i = 0; i < ops.size(); ++i) ops[i].raw = raw_products[i]->p;
        return ops;
    }
    std::vector<GateTransform> gates(bool diagnostics) {
        return {{GateKind::gelu, pre.p, nullptr, nullptr, hidden.p, tokens(), 64, 64, nullptr, 1, false, diagnostics ? raw_hidden.p : nullptr},
                {GateKind::affine, product.p, bias.p, nullptr, mu.p, tokens(), 14, 14, nullptr, 0.1f, false, diagnostics ? raw_mu.p : nullptr},
                {GateKind::copy, mu.p + 12, nullptr, nullptr, suffix.coefficients.p, tokens(), 2, 14}};
    }
    std::vector<ResidualMix> mixes(bool diagnostics) {
        std::vector<ResidualMix> ops(3);
        const bf16 *values[11] = {source(2), source(0), source(1), source(0), source(1), source(2), source(6), before.p, attention.output.p, source(0), bigram.p};
        int slots[11] = {5, 3, 4, 0, 1, 2, 6, 8, 9, 10, 11};
        for (int i = 0; i < 3; ++i) {
            auto &op = ops[i]; op.tokens = tokens(); op.count = i ? 4 : 3;
            op.output = i == 0 ? before.p : i == 1 ? attention.aux.p : suffix.before.p;
            op.dy = i == 0 ? partial[7]->p : i == 1 ? attention.daux.p : suffix.input_gradient.p;
            op.raw = diagnostics ? raw_mix[i]->p : nullptr;
            for (int j = 0; j < op.count; ++j) {
                int id = (i == 0 ? 0 : i == 1 ? 3 : 7) + j; auto &term = op.terms[j];
                term.value = values[id]; term.coefficient = mu.p + slots[id]; term.coefficient_row_stride = 14;
                term.constant = id == 0 ? 1 : 0; term.groups = id == 6 ? 2 : 1;
                term.coefficient_group_stride = id == 6 ? 1 : 0;
                term.dvalue = id == 8 ? attention.dy.p : partial[id]->p;
                term.dcoefficient_rows = coefficient[id]->p;
            }
        }
        return ops;
    }
    ResidualNorm norm(bool diagnostics) {
        return {source(1), shared_dy.p, source(9), shared_dx.p, tokens(), nullptr, raw_norm.p, diagnostics ? raw_dnorm.p : nullptr};
    }
    std::vector<GradientCast> casts() {
        int ids[12] = {3, 4, 5, 1, 2, 0, 6, 6, 7, 8, 9, 10}; std::vector<GradientCast> ops;
        for (int i = 0; i < 14; ++i) {
            GradientCast op{}; op.output = dmu.p + i; op.elements = tokens(); op.output_stride = 14; op.scale = 0.1f;
            if (i < 12) {
                op.input = coefficient[ids[i]]->p + (i == 7 ? 1 : 0); op.input_stride = i == 6 || i == 7 ? 2 : 1; op.round_input = true;
            } else { op.rounded_input = suffix.dcoeff.p + i - 12; op.input_stride = 2; }
            ops.push_back(op);
        }
        return ops;
    }
    std::vector<GradientSum> sums() {
        auto &d = suffix.tail.dsources;
        return {{{attention.dx.p, d[9]->p}, shared_dy.p, int(before.n), 2},
                {{d[0]->p, partial[1]->p, partial[3]->p, partial[9]->p}, total[0]->p, int(before.n), 4},
                {{d[1]->p, partial[2]->p, partial[4]->p, shared_dx.p}, total[1]->p, int(before.n), 4},
                {{d[2]->p, partial[0]->p, partial[5]->p, dx_network.p}, total[2]->p, int(before.n), 4},
                {{d[6]->p, partial[6]->p}, total[3]->p, int(before.n), 2}};
    }
    std::vector<DeviceBuffer<bf16> *> outputs() {
        auto out = suffix.outputs(), attn = attention.outputs(); out.insert(out.end(), attn.begin(), attn.end());
        out.insert(out.end(), {&pre, &hidden, &product, &mu, &before, &dmu, &dh, &dpre, &dx_network, &dw1, &dw2, &dbias,
                              &shared_dy, &shared_dx, &suffix.before, &suffix.coefficients, &attention.aux, &attention.dy, suffix.tail.sources[9].get()});
        for (size_t i = 0; i < partial.size(); ++i) if (i != 8) out.push_back(partial[i].get());
        for (auto &b : total) out.push_back(b.get()); return out;
    }
    std::vector<DeviceBuffer<fp8> *> packed_outputs() {
        auto out = suffix.packed_outputs(); out.insert(out.end(), {&attention.x8, &attention.xt8, &attention.grad8, &attention.gradt8}); return out;
    }
    void poison() {
        suffix.poison(); attention.poison();
        for (auto *b : outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : packed_outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n));
        for (auto *b : {&raw_norm, &raw_dnorm, &raw_hidden, &raw_mu, &raw_dpre}) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto *list : {&coefficient, &raw_products, &raw_mix}) for (auto &b : *list) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
    }
};

struct LastLayerPlan {
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    std::vector<std::vector<int>> predecessor;
    LastLayerPlan() = default;
    int phase(TaskKind kind, int op, int nr, int nc = 1, int backward = 0, int column = -1) {
        int id = int(stages.size()), begin = int(tasks.size());
        for (int r = 0; r < nr; ++r) for (int c = 0; c < nc; ++c)
            tasks.push_back({op, r, column < 0 ? c : column, {-1, -1}, backward, 0, kind});
        stages.push_back({begin, int(tasks.size())}); predecessor.emplace_back(); return id;
    }
    void release(std::vector<int> from, int first, int last = -1) {
        if (last < 0) last = first;
        int expected = 0, id = int(groups.size());
        for (int p : from) {
            auto [begin, end] = stages[p]; expected += end - begin;
            for (int i = begin; i < end; ++i) {
                int slot = tasks[i].signal[0] < 0 ? 0 : 1;
                if (tasks[i].signal[slot] >= 0) throw std::runtime_error("last-layer dependency slots exhausted");
                tasks[i].signal[slot] = id;
            }
        }
        groups.push_back({expected, stages[first].first, stages[last].second});
        for (int i = first; i <= last; ++i) predecessor[i] = from;
    }
    int layer_stage(const AttentionLayerPlan &p, int index) {
        int id = int(stages.size()), begin = int(tasks.size()); auto range = p.stages[index];
        for (int i = range.first; i < range.second; ++i) {
            Task task = p.tasks[i]; task.signal[0] = task.signal[1] = -1;
            if (task.kind == TaskKind::matmul || task.kind == TaskKind::bf16_matmul) task.op += 9;
            tasks.push_back(task);
        }
        stages.push_back({begin, int(tasks.size())}); predecessor.emplace_back(); return id;
    }
    LastLayerPlan(LastLayerStorage &b, const AttentionLayerPlan &attention, const SuffixPlan &suffix) {
        int t = b.tokens(), rows = (t + 3) / 4;
        auto matrix = [&](int id) { auto op = b.products(false)[id]; return phase(TaskKind::bf16_matmul, id + 12, (op.m + 63) / 64, (op.n + 63) / 64); };
        phase(TaskKind::residual_norm, 2, rows); phase(TaskKind::activation_pack, 2, (t + 63) / 64, 12);
        matrix(0); phase(TaskKind::gate_transform, 3, rows); matrix(1); phase(TaskKind::gate_transform, 4, rows);
        phase(TaskKind::gate_transform, 5, rows); phase(TaskKind::residual_mix, 2, rows); phase(TaskKind::residual_mix, 3, rows);
        for (int i = 0; i < 6; ++i) layer_stage(attention, i);
        phase(TaskKind::residual_mix, 4, rows);
        for (int i = 1; i < int(stages.size()); ++i) release({i - 1}, i);
        int base = int(tasks.size()), stage_base = int(stages.size()), group_base = int(groups.size());
        for (auto task : suffix.tasks) { for (int &signal : task.signal) if (signal >= 0) signal += group_base; tasks.push_back(task); }
        for (auto group : suffix.groups) { group.begin += base; group.end += base; groups.push_back(group); }
        for (size_t i = 0; i < suffix.stages.size(); ++i) {
            auto range = suffix.stages[i]; stages.push_back({range.first + base, range.second + base});
            predecessor.push_back({suffix.predecessor[i] < 0 ? stage_base - 1 : stage_base + suffix.predecessor[i]});
        }
        release({stage_base - 1}, stage_base);
        int after = phase(TaskKind::residual_mix, 4, rows, 4, 1); release({stage_base + 42}, after);
        int pre = phase(TaskKind::residual_mix, 2, rows, 3, 1);
        int attn = layer_stage(attention, 6); release({after}, pre, attn);
        for (int i = 7; i < 15; ++i) { int next = layer_stage(attention, i); release({next - 1}, next); }
        int attn_done = int(stages.size()) - 1;
        int aux = phase(TaskKind::residual_mix, 3, rows, 4, 1);
        int norm_sum = phase(TaskKind::gradient_sum, 0, (t * 768 + 127) / 128); release({attn_done}, aux, norm_sum);
        int norm_back = phase(TaskKind::residual_norm, 2, rows, 1, 1); release({norm_sum}, norm_back);
        int cast = int(stages.size());
        // One phase owns all fourteen coefficient columns, including the two MLP consumers.
        int begin = int(tasks.size());
        for (int id = 0; id < 14; ++id) for (int r = 0; r < (t + 127) / 128; ++r)
            tasks.push_back({id + 2, r, 0, {-1, -1}, 0, 0, TaskKind::gradient_cast});
        stages.push_back({begin, int(tasks.size())}); predecessor.emplace_back();
        release({aux, pre, stage_base + 43, stage_base + 44}, cast);
        int dh = matrix(2); matrix(5); int bias = phase(TaskKind::network_backward, 0, 1, 1, 0, 1); release({cast}, dh, bias);
        int gelu = phase(TaskKind::network_backward, 0, rows); release({dh}, gelu);
        int dx = matrix(3), dw = matrix(4); release({gelu}, dx, dw);
        begin = int(tasks.size()); int sum = int(stages.size());
        for (int id = 1; id < 5; ++id) for (int r = 0; r < (t * 768 + 127) / 128; ++r)
            tasks.push_back({id, r, 0, {-1, -1}, 0, 0, TaskKind::gradient_sum});
        stages.push_back({begin, int(tasks.size())}); predecessor.emplace_back();
        release({dx, norm_back}, sum);
    }
};

struct LastLayerSchedule {
    SuffixSchedule suffix;
    LayerSchedule attention;
    LastLayerPlan plan;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    DeviceBuffer<Matmul> products;
    DeviceBuffer<BF16Matmul> bf16_products;
    DeviceBuffer<HeadSetup> head_setup;
    DeviceBuffer<MLPSetup> mlp_setup;
    DeviceBuffer<AttentionLayerSetup> layer_setup;
    DeviceBuffer<ActivationPack> packs;
    DeviceBuffer<GateTransform> gates;
    DeviceBuffer<ResidualNorm> norms;
    DeviceBuffer<ResidualMix> mixes;
    DeviceBuffer<GradientCast> casts;
    DeviceBuffer<GradientSum> sums;
    DeviceBuffer<NetworkBackward> network;
    explicit LastLayerSchedule(LastLayerStorage &b) : suffix(b.suffix), attention(b.attention.buffers()), plan(b, attention.plan, suffix.plan),
        tasks(plan.tasks.size()), groups(plan.groups.size()), counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2),
        products(12), bf16_products(18), head_setup(1), mlp_setup(1), layer_setup(1), packs(3), gates(6), norms(3), mixes(5), casts(16), sums(5), network(1) {
        tasks.put(plan.tasks); groups.put(plan.groups); configure(b, true);
    }
    void configure(LastLayerStorage &b, bool diagnostics) {
        suffix.configure(b.suffix, diagnostics); AttentionLayerPlan ap(b.attention.buffers(31, diagnostics));
        auto p = suffix.products.get(); p.insert(p.end(), ap.ops.begin(), ap.ops.end()); products.put(p);
        auto bp = b.suffix.tail.products(diagnostics), net = b.products(diagnostics);
        bp.insert(bp.end(), ap.bf16_ops.begin(), ap.bf16_ops.end()); bp.insert(bp.end(), net.begin(), net.end()); bf16_products.put(bp);
        attention.qkv.put({ap.qkv}); attention.attention.put({ap.attention}); attention.post.put({ap.post}); attention.gradient.put(ap.gradient);
        layer_setup.put({ap.setup(products.p + 9, bf16_products.p + 9, attention.qkv.p)});
        head_setup.put({{products.p, b.suffix.tail.head.scales.p, b.suffix.tail.head.loss.parameters.p}});
        mlp_setup.put({{products.p + 3, b.suffix.scales.p, b.suffix.amax.p}});
        auto pack = suffix.packs.get(); pack.push_back({b.source(9), b.attention.x8.p, b.attention.xt8.p, b.attention.scalars.p, b.tokens(), 768, false, b.raw_norm.p}); packs.put(pack);
        auto gate = b.suffix.tail.gates(diagnostics), ng = b.gates(diagnostics); gate.insert(gate.end(), ng.begin(), ng.end()); gates.put(gate);
        norms.put({b.suffix.tail.norm(diagnostics), b.suffix.norm(diagnostics), b.norm(diagnostics)});
        auto mix = suffix.mixes.get(), nm = b.mixes(diagnostics); mix.insert(mix.end(), nm.begin(), nm.end()); mixes.put(mix);
        auto cast = suffix.casts.get(), nc = b.casts(); cast.insert(cast.end(), nc.begin(), nc.end()); casts.put(cast); sums.put(b.sums());
        network.put({{b.pre.p, b.dh.p, b.dmu.p, b.dpre.p, b.dbias.p, b.tokens(), 14, diagnostics ? b.raw_dpre.p : nullptr}});
    }
    Graph graph(bool checked = false) {
        Graph g = suffix.graph(); g.ops = products.p; g.bf16_ops = bf16_products.p; g.tasks = tasks.p; g.groups = groups.p;
        g.counters = counters.p; g.queue = queue.p; g.state = state.p; g.task_count = int(tasks.n); g.root_count = plan.stages[0].second;
        g.audit = checked ? audit.p : nullptr; g.head_setups = head_setup.p; g.mlp_setups = mlp_setup.p; g.layer_setup = layer_setup.p;
        g.attention = attention.attention.p; g.qkv = attention.qkv.p; g.attention_post = attention.post.p; g.projection_gradient = attention.gradient.p;
        g.activation_packs = packs.p; g.gates = gates.p; g.residual_norm = norms.p; g.residual_mix = mixes.p;
        g.gradient_casts = casts.p; g.gradient_sums = sums.p; g.network_backwards = network.p; return g;
    }
    void run(int workers, bool checked = false) {
        reset_loss<<<(tasks.n + 127) / 128, 128>>>(graph(), int(groups.n));
        if (checked) {
            CHECK_CUDA(cudaMemset(audit.p, 0, audit.n * sizeof(int)));
            megakernel<true, false, true, true, false, true, true, true><<<workers, 128>>>(graph(true));
        } else megakernel<false, false, true, true, false, true, true, true><<<workers, 128>>>(graph());
        CHECK_CUDA(cudaGetLastError());
    }
    void verify() {
        auto visits = audit.get(), completed = counters.get();
        if (state.get()[2]) throw std::runtime_error("unfinished last-layer tasks");
        for (size_t i = 0; i < tasks.n; ++i) if (visits[i] != 1) throw std::runtime_error("last-layer task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i) if (completed[i] != plan.groups[i].expected) throw std::runtime_error("last-layer dependency mismatch");
    }
};

template <int MinimumBlocks>
__global__ __launch_bounds__(128, MinimumBlocks) void last_layer_stage(Graph g, int begin) {
    __shared__ __align__(1024) fp8 scratch[bf16_scratch_bytes];
    Task task = g.tasks[begin + blockIdx.x];
    if (task.kind == TaskKind::gradient_sum) gradient_sum(g.gradient_sums[task.op], task.row);
    else if (task.kind == TaskKind::network_backward) network_backward(g.network_backwards[task.op], task.row, task.col != 0);
    else if (task.kind == TaskKind::layer_setup) setup_attention_layer(g.layer_setup[task.op]);
    else if (task.kind == TaskKind::attention_post) execute_attention_post(g.attention_post[task.op], AttentionPostStage(task.k_begin), task.row, task.col);
    else if (task.kind == TaskKind::projection_gradient) projection_gradient_tile(g.projection_gradient[task.op], task.col != 0, task.row, reinterpret_cast<float *>(scratch));
    else if (task.kind == TaskKind::qkv_forward || task.kind == TaskKind::qkv_backward) execute_qkv(g.qkv[task.op], task.kind == TaskKind::qkv_backward, task.row, task.col);
    else if (task.kind == TaskKind::attention_forward || task.kind == TaskKind::attention_dq || task.kind == TaskKind::attention_dkv) execute_attention(g.attention[task.op], task.kind, task.row, task.col);
    else if (task.kind == TaskKind::activation_pack) activation_pack(g.activation_packs[task.op], task.row, task.col, scratch);
    else if (task.kind == TaskKind::mlp_setup) mlp_setup(g.mlp_setups[task.op]);
    else if (task.kind == TaskKind::gradient_cast) gradient_cast(g.gradient_casts[task.op], task.row);
    else if (task.kind == TaskKind::tail_backward) tail_backward(g.tail_backward_ops[task.op], task.row, TailStep(task.col));
    else if (task.kind == TaskKind::bf16_matmul) bf16_matmul_tile(g.bf16_ops[task.op], task.row, task.col, scratch);
    else if (task.kind == TaskKind::gate_transform) gate_transform(g.gates[task.op], task.row);
    else if (task.kind == TaskKind::residual_norm) residual_norm(g.residual_norm[task.op], task.row, task.k_begin != 0);
    else if (task.kind == TaskKind::residual_mix) {
        if (task.k_begin) residual_mix_backward(g.residual_mix[task.op], task.row, task.col);
        else residual_mix_forward(g.residual_mix[task.op], task.row);
    } else if (task.kind == TaskKind::head_setup) nano::head_setup(g.head_setups[task.op]);
    else if (task.kind == TaskKind::head_input) head_input(g.head_inputs[task.op], task.row, task.col, scratch);
    else if (task.kind == TaskKind::loss_partial) training_loss_partial(g.training_losses[task.op], task.row, task.col);
    else if (task.kind == TaskKind::loss_reduce) training_loss_reduce(g.training_losses[task.op], task.row);
    else if (task.kind == TaskKind::loss_gradient) training_loss_gradient(g.training_losses[task.op], task.row, task.col);
    else execute_tile<false, true>(g.ops[task.op], task, scratch);
}

struct LastLayerControl {
    cudaGraph_t graph; cudaGraphExec_t exec;
    template <class Schedule> LastLayerControl(Schedule &schedule, bool bounded) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0)); std::vector<cudaGraphNode_t> nodes;
        for (size_t i = 0; i < schedule.plan.stages.size(); ++i) {
            auto [begin, end] = schedule.plan.stages[i]; Graph g = schedule.graph(); void *args[] = {&g, &begin};
            cudaKernelNodeParams p{}; p.func = bounded ? reinterpret_cast<void *>(last_layer_stage<4>) : reinterpret_cast<void *>(last_layer_stage<1>);
            p.gridDim = dim3(end - begin); p.blockDim = dim3(128); p.kernelParams = args;
            std::vector<cudaGraphNode_t> deps; for (int previous : schedule.plan.predecessor[i]) deps.push_back(nodes[previous]);
            cudaGraphNode_t node; CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, deps.data(), deps.size(), &p)); nodes.push_back(node);
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~LastLayerControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

void check_last_layer_math(LastLayerStorage &b, LastLayerSchedule &schedule) {
    CHECK_CUDA(cudaMemcpy(schedule.suffix.products.p, schedule.products.p, 9 * sizeof(Matmul), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(schedule.attention.ops.p, schedule.products.p + 9, 3 * sizeof(Matmul), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(schedule.attention.bf16_ops.p, schedule.bf16_products.p + 9, 3 * sizeof(BF16Matmul), cudaMemcpyDeviceToDevice));
    check_suffix_math(b.suffix, schedule.suffix); check_layer_math(b.attention, schedule.attention);
    LayerBlasReference blas; for (auto op : b.products(true)) blas.check_product(op);
    auto pre = b.pre.get(), hidden = b.hidden.get(), product = b.product.get(), bias = b.bias.get(), mu = b.mu.get();
    auto dh = b.dh.get(), dpre = b.dpre.get(), dmu = b.dmu.get(), dbias = b.dbias.get();
    auto raw_hidden = b.raw_hidden.get(), raw_mu = b.raw_mu.get(), raw_dpre = b.raw_dpre.get();
    std::vector<double> rh(pre.size()), rm(mu.size()), rd(pre.size());
    for (size_t i = 0; i < pre.size(); ++i) {
        double x = float(pre[i]), cdf = 0.5 * (1 + std::erf(x / std::sqrt(2.0)));
        rh[i] = x * cdf; rd[i] = float(dh[i]) * (cdf + x * std::exp(-0.5 * x * x) / std::sqrt(2 * std::acos(-1.0)));
        if (float(hidden[i]) != float(bf16(raw_hidden[i])) || float(dpre[i]) != float(bf16(raw_dpre[i]))) throw std::runtime_error("last-layer GELU rounding mismatch");
    }
    for (size_t i = 0; i < mu.size(); ++i) {
        rm[i] = (double(float(product[i])) + float(bias[i % 14])) * double(0.1f);
        if (float(mu[i]) != float(bf16(raw_mu[i]))) throw std::runtime_error("last-layer coefficient rounding mismatch");
    }
    layer_near("last-layer GELU", raw_hidden, rh); layer_near("last-layer coefficient affine", raw_mu, rm); layer_near("last-layer GELU backward", raw_dpre, rd);
    for (int c = 0; c < 14; ++c) {
        double sum = 0; for (int r = 0; r < b.tokens(); ++r) sum += float(dmu[r * 14 + c]);
        if (float(dbias[c]) != float(bf16(float(sum)))) throw std::runtime_error("last-layer bias gradient mismatch");
    }
    // References name the source roles independently of the graph's term descriptors.
    auto c0 = b.suffix.tail.sources[0]->get(), c7 = b.suffix.tail.sources[1]->get(), c9 = b.suffix.tail.sources[2]->get();
    auto ve = b.suffix.tail.sources[6]->get(), before = b.before.get(), ay = b.attention.output.get(), bigram = b.bigram.get();
    auto ady = b.suffix.input_gradient.get(), pdy = b.partial[7]->get(), vdy = b.attention.daux.get();
    std::vector<std::vector<bf16>> values{c9, c0, c7, c0, c7, c9, ve, before, ay, c0, bigram};
    std::vector<std::vector<double>> mixes(3, std::vector<double>(before.size()));
    int slots[11] = {5, 3, 4, 0, 1, 2, 6, 8, 9, 10, 11};
    for (int id = 0; id < 11; ++id) {
        int phase = id < 3 ? 0 : id < 7 ? 1 : 2, groups = id == 6 ? 2 : 1;
        auto &dy = phase == 0 ? pdy : phase == 1 ? vdy : ady;
        auto actual = id == 8 ? b.attention.dy.get() : b.partial[id]->get();
        std::vector<double> dc(size_t(b.tokens()) * groups);
        for (int row = 0; row < b.tokens(); ++row) for (int col = 0; col < 768; ++col) {
            size_t index = size_t(row) * 768 + col; int group = col / (768 / groups);
            double gain = float(mu[row * 14 + slots[id] + group]) + (id == 0 ? 1 : 0);
            mixes[phase][index] += gain * float(values[id][index]); dc[row * groups + group] += double(float(dy[index])) * float(values[id][index]);
            if (float(actual[index]) != float(bf16(float(gain * float(dy[index]))))) throw std::runtime_error("last-layer residual source adjoint mismatch");
        }
        layer_near("last-layer coefficient partial", b.coefficient[id]->get(), dc);
    }
    for (int i = 0; i < 3; ++i) {
        auto raw = b.raw_mix[i]->get(); layer_near("last-layer residual mix", raw, mixes[i]);
        auto output = i == 0 ? b.before.get() : i == 1 ? b.attention.aux.get() : b.suffix.before.get();
        for (size_t j = 0; j < raw.size(); ++j) if (float(output[j]) != float(bf16(raw[j]))) throw std::runtime_error("last-layer residual cast mismatch");
    }
    std::vector<std::vector<float>> partial; for (auto &buffer : b.coefficient) partial.push_back(buffer->get());
    auto mlp_coefficient = b.suffix.dcoeff.get(), copied = b.suffix.coefficients.get();
    int ids[12] = {3, 4, 5, 1, 2, 0, 6, 6, 7, 8, 9, 10};
    for (int row = 0; row < b.tokens(); ++row) for (int c = 0; c < 14; ++c) {
        float x = c >= 12 ? float(mlp_coefficient[row * 2 + c - 12]) :
            float(bf16(partial[ids[c]][row * (c == 6 || c == 7 ? 2 : 1) + (c == 7 ? 1 : 0)]));
        if (float(dmu[row * 14 + c]) != float(bf16(x * 0.1f))) throw std::runtime_error("last-layer coefficient gradient mapping mismatch");
        if (c >= 12 && float(copied[row * 2 + c - 12]) != float(mu[row * 14 + c])) throw std::runtime_error("last-layer MLP coefficient copy mismatch");
    }
    for (const auto &op : b.sums()) {
        std::vector<float> expected(op.elements, 0); auto actual = layer_read(op.output, op.elements);
        for (int i = 0; i < op.count; ++i) { auto input = layer_read(op.input[i], op.elements); for (int j = 0; j < op.elements; ++j) expected[j] += float(input[j]); }
        for (int i = 0; i < op.elements; ++i) if (float(actual[i]) != float(bf16(expected[i]))) throw std::runtime_error("last-layer shared-source gradient sum mismatch");
    }
    auto raw_norm = b.raw_norm.get(), raw_dnorm = b.raw_dnorm.get(); auto normalized = b.suffix.tail.sources[9]->get();
    auto norm_dy = b.shared_dy.get(), norm_dx = b.shared_dx.get(); std::vector<double> rn(c7.size()), rdn(c7.size());
    for (int row = 0; row < b.tokens(); ++row) {
        double square = 0, dot = 0;
        for (int c = 0; c < 768; ++c) { size_t i = size_t(row) * 768 + c; square += double(float(c7[i])) * float(c7[i]); dot += double(float(c7[i])) * float(norm_dy[i]); }
        double rstd = 1 / std::sqrt(square / 768 + 0x1p-23);
        for (int c = 0; c < 768; ++c) {
            size_t i = size_t(row) * 768 + c; rn[i] = float(c7[i]) * rstd;
            rdn[i] = rstd * (float(norm_dy[i]) - float(c7[i]) * dot / 768 * rstd * rstd);
            if (float(normalized[i]) != float(bf16(raw_norm[i])) || float(norm_dx[i]) != float(bf16(raw_dnorm[i]))) throw std::runtime_error("last-layer shared RMS cast mismatch");
        }
    }
    layer_near("last-layer shared RMS", raw_norm, rn); layer_near("last-layer shared RMS backward", raw_dnorm, rdn);
    auto packed = b.attention.x8.get(), transposed = b.attention.xt8.get(); float inverse = 1.0f / b.attention.scalars.get()[0];
    for (size_t i = 0; i < packed.size(); ++i) {
        fp8 expected(std::min(448.0f, std::max(-448.0f, raw_norm[i] * inverse)));
        if (packed[i].__x != expected.__x || transposed[(i % 768) * b.tokens() + i / 768].__x != expected.__x) throw std::runtime_error("last-layer shared FP8 cache mismatch");
    }
}

void check_last_layer(int tokens, int vocabulary, int workers, int steps, bool timing, bool aten) {
    LastLayerStorage b(tokens, vocabulary); LastLayerSchedule schedule(b); LastLayerControl control(schedule, false), bounded(schedule, true);
    for (int step = 0; step < steps; ++step) {
        b.change(step); b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify(); check_last_layer_math(b, schedule);
        if (aten) check_tail_aten(b.suffix.tail.reference());
        std::vector<std::vector<bf16>> expected; for (auto *buffer : b.outputs()) expected.push_back(buffer->get());
        std::vector<std::vector<fp8>> packed; for (auto *buffer : b.packed_outputs()) packed.push_back(buffer->get());
        auto loss = b.suffix.tail.head.loss.loss.get(), maximum = b.suffix.amax.get(), gains = b.attention.gains.get(); auto head_dw = b.suffix.tail.head.dw.get();
        for (int mode = 0; mode < 4; ++mode) {
            b.poison(); schedule.configure(b, mode != 3);
            if (!mode) control.run(); else if (mode == 1) bounded.run(); else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize()); size_t index = 0;
            auto same = [](const auto &a, const auto &b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])); };
            for (auto *buffer : b.outputs()) {
                if (!same(buffer->get(), expected[index])) throw std::runtime_error("last-layer BF16 replay mismatch at buffer " + std::to_string(index));
                ++index;
            }
            index = 0; for (auto *buffer : b.packed_outputs()) if (!same(buffer->get(), packed[index++])) throw std::runtime_error("last-layer FP8 replay mismatch");
            if (!same(b.suffix.tail.head.loss.loss.get(), loss) || !same(b.suffix.tail.head.dw.get(), head_dw) || !same(b.suffix.amax.get(), maximum) ||
                !same(b.attention.gains.get(), gains)) throw std::runtime_error("last-layer loss/weight-gradient/maximum/gain replay mismatch");
        }
        schedule.configure(b, true);
        printf("LAST_LAYER PASS tokens=%d vocabulary=%d workers=%d step=%d tasks=%zu phases=%zu\n", tokens, vocabulary, workers, step, schedule.tasks.n, schedule.plan.stages.size());
    }
    if (timing) {
        schedule.configure(b, false);
        device_time("LAST_LAYER CUDA Graph", [&] { control.run(); });
        device_time("LAST_LAYER CUDA Graph four-block occupancy", [&] { bounded.run(); });
        device_time("LAST_LAYER persistent+reset", [&] { schedule.run(workers); });
    }
}

#ifndef NANO_TRAINING_LAYER_NINE
int main(int argc, char **argv) {
    try {
        device_info(); bool quick = false, profile = false, aten = true;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--quick") quick = true; else if (arg == "--profile") profile = true; else if (arg == "--native-only") aten = false;
            else if (arg.rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown last-layer argument");
        }
        cudaDeviceProp properties{}; CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        if (profile) {
            LastLayerStorage b(256, 50304); b.change(0); LastLayerSchedule schedule(b); schedule.configure(b, false);
            for (int i = 0; i < 5; ++i) schedule.run(properties.multiProcessorCount * 4);
            reset_loss<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, true, true, false, true, true, true><<<properties.multiProcessorCount * 4, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop()); return 0;
        }
        if (!aten) puts("Native-only checks: ATen tail diagnostic excluded; cuBLAS, FP64 and replay gates remain enabled");
        check_last_layer(16, 2048, 17, 4, false, aten);
        if (!quick) {
            check_last_layer(80, 2048, 1, 1, false, aten); check_last_layer(16, 50304, 17, 1, false, aten);
            check_last_layer(256, 50304, properties.multiProcessorCount * 4, 1, true, aten);
        }
        puts("PASS: last-layer attention and MUDD network connected to MLP/loss/backward; earlier body and compiled parity remain"); return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
#endif

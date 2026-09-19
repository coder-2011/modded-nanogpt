#define NANO_TRAINING_SUFFIX
#include "tail.cu"
#include "mlp_training.cuh"

struct SuffixStorage {
    TailStorage tail;
    DeviceBuffer<bf16> before, coefficients, output, dy, dx, dw1, dw2, direct, norm_dy, norm_dx, input_gradient, dcoeff;
    DeviceBuffer<fp8> w1, w1t, w2, w2t, x, xt, g, gt, post, postt, dpre, dpret;
    DeviceBuffer<float> scales, amax, raw_norm, raw_dnorm, raw_mix, dr, dp;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> raw_products;
    SuffixStorage(int t, int v) : tail(t, v), before(size_t(t) * 768), coefficients(size_t(t) * 2),
        output(before.n), dy(before.n), dx(before.n), dw1(2816 * 768), dw2(dw1.n), direct(before.n), norm_dy(before.n), norm_dx(before.n),
        input_gradient(before.n), dcoeff(coefficients.n), w1(dw1.n), w1t(dw1.n), w2(dw2.n), w2t(dw2.n),
        x(before.n), xt(before.n), g(before.n), gt(before.n), post(size_t(t) * 2816), postt(post.n), dpre(post.n), dpret(post.n),
        scales(6), amax(2), raw_norm(before.n), raw_dnorm(before.n), raw_mix(before.n), dr(t), dp(t) {
        for (size_t n : {post.n, output.n, dpre.n, dx.n, dw1.n, dw2.n}) raw_products.push_back(std::make_unique<DeviceBuffer<float>>(n));
    }
    void change(int step) {
        tail.change(step); std::mt19937 rng(9351 + step); std::normal_distribution<float> random;
        std::vector<bf16> input(before.n), gains(coefficients.n);
        for (auto &v : input) v = bf16(step == 2 ? 0 : 0.2f * random(rng));
        for (size_t i = 0; i < gains.size(); i += 2) {
            gains[i] = bf16(1.05f + 0.05f * random(rng));
            gains[i + 1] = bf16(step == 3 ? 0.0f : 0.8f * random(rng));
        }
        before.put(input); coefficients.put(gains);
        std::vector<float> s{0.0625f, 0.0005f, 0.0004f, 0.015625f, 0.01f, 0.00002f};
        if (step == 3) { s[1] *= 2; s[2] *= 2; s[4] *= 2; s[5] *= 2; }
        scales.put(s);
        for (int index = 0; index < 2; ++index) {
            std::vector<fp8> row(w1.n), transposed(w1.n);
            for (int r = 0; r < 2816; ++r) for (int c = 0; c < 768; ++c) {
                float weight = step == 1 && index == 1 ? 0.0f : float(bf16(0.015f * random(rng)));
                row[r * 768 + c] = fp8(weight / s[index + 1]); transposed[c * 2816 + r] = row[r * 768 + c];
            }
            (index ? w2 : w1).put(row); (index ? w2t : w1t).put(transposed);
        }
    }
    std::vector<Matmul> products(bool diagnostics = true) {
        MLPTrainingBuffers b{tail.head.loss.tokens, w1.p, w1t.p, w2.p, w2t.p, x.p, xt.p, g.p, gt.p, post.p, postt.p,
                              dpre.p, dpret.p, output.p, dx.p, dw1.p, dw2.p, amax.p};
        if (diagnostics) for (int i = 0; i < 6; ++i) b.raw[i] = raw_products[i]->p;
        return mlp_training_products(b);
    }
    ResidualNorm norm(bool diagnostics) {
        // Keep the FP32 normalization result for the fused producer-to-FP8 edge.
        return {before.p, norm_dy.p, tail.sources[8]->p, norm_dx.p, tail.head.loss.tokens, nullptr,
                raw_norm.p, diagnostics ? raw_dnorm.p : nullptr};
    }
    ResidualMix mix(bool diagnostics) {
        ResidualMix op{}; op.tokens = tail.head.loss.tokens; op.output = tail.x.p; op.dy = tail.dx.p; op.count = 2;
        op.raw = diagnostics ? raw_mix.p : nullptr;
        op.terms[0].value = before.p; op.terms[0].coefficient = coefficients.p; op.terms[0].coefficient_row_stride = 2;
        op.terms[0].dvalue = direct.p; op.terms[0].dcoefficient_rows = dr.p;
        op.terms[1].value = output.p; op.terms[1].coefficient = coefficients.p + 1; op.terms[1].coefficient_row_stride = 2;
        op.terms[1].dvalue = dy.p; op.terms[1].dcoefficient_rows = dp.p;
        return op;
    }
    std::vector<DeviceBuffer<bf16> *> outputs() {
        auto buffers = tail.outputs();
        buffers.insert(buffers.end(), {&output, &dy, &dx, &dw1, &dw2, &direct, &norm_dy, &norm_dx, &input_gradient, &dcoeff});
        buffers.push_back(&tail.x); buffers.push_back(tail.sources[8].get()); return buffers;
    }
    std::vector<DeviceBuffer<fp8> *> packed_outputs() { return {&x, &xt, &g, &gt, &post, &postt, &dpre, &dpret}; }
    void poison() {
        tail.poison();
        for (auto *b : outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : packed_outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n));
        for (auto &b : raw_products) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto *b : {&amax, &raw_norm, &raw_dnorm, &raw_mix, &dr, &dp}) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
    }
};

struct SuffixPlan {
    std::vector<Task> tasks;
    std::vector<Group> groups;
    std::vector<std::pair<int, int>> stages;
    std::vector<int> predecessor;
    explicit SuffixPlan(SuffixStorage &b) {
        int t = b.tail.head.loss.tokens, rows = (t + 3) / 4;
        auto phase = [&](TaskKind kind, int op, int nr, int nc = 1, int backward = 0, int column = -1) {
            int begin = int(tasks.size());
            for (int r = 0; r < nr; ++r) for (int c = 0; c < nc; ++c)
                tasks.push_back({op, r, column < 0 ? c : column, {-1, -1}, backward, 0, kind});
            predecessor.push_back(int(stages.size()) - 1); stages.push_back({begin, int(tasks.size())});
        };
        auto matrix = [&](int id) { auto op = b.products(false)[id]; phase(TaskKind::matmul, id + 3, (op.m + 63) / 64, (op.n + 63) / 64); };
        phase(TaskKind::residual_norm, 1, rows);
        phase(TaskKind::activation_pack, 0, (t + 63) / 64, 12);
        phase(TaskKind::mlp_setup, 0, 1); matrix(0); matrix(1);
        phase(TaskKind::residual_mix, 1, rows);
        auto connect = [&](int from, int begin, int end) {
            int id = int(groups.size()); auto range = stages[from]; groups.push_back({range.second - range.first, begin, end});
            for (int i = range.first; i < range.second; ++i) {
                int slot = tasks[i].signal[0] < 0 ? 0 : 1;
                if (tasks[i].signal[slot] >= 0) throw std::runtime_error("suffix dependency slot occupied");
                tasks[i].signal[slot] = id;
            }
        };
        for (int i = 0; i < 5; ++i) connect(i, stages[i + 1].first, stages[i + 1].second);
        TailPlan tail(b.tail); int base = int(tasks.size()), phase_base = int(stages.size());
        connect(5, base, base + tail.stages[0].second); int group_base = int(groups.size());
        for (auto task : tail.tasks) { for (int &signal : task.signal) if (signal >= 0) signal += group_base; tasks.push_back(task); }
        for (auto group : tail.groups) { group.begin += base; group.end += base; groups.push_back(group); }
        for (size_t i = 0; i < tail.stages.size(); ++i) {
            auto range = tail.stages[i]; stages.push_back({range.first + base, range.second + base});
            predecessor.push_back(tail.predecessor[i] < 0 ? phase_base - 1 : phase_base + tail.predecessor[i]);
        }
        int backward_begin = int(stages.size());
        phase(TaskKind::residual_mix, 1, rows, 2, 1);
        phase(TaskKind::activation_pack, 1, (t + 63) / 64, 12);
        matrix(2); matrix(3); matrix(4); matrix(5);
        phase(TaskKind::tail_backward, 1, rows, 1, 0, int(TailStep::input_sum));
        phase(TaskKind::residual_norm, 1, rows, 1, 1);
        phase(TaskKind::tail_backward, 2, rows, 1, 0, int(TailStep::input_sum));
        phase(TaskKind::gradient_cast, 0, (t + 127) / 128);
        phase(TaskKind::gradient_cast, 1, (t + 127) / 128);
        auto edge = [&](int from, int to) { connect(from, stages[to].first, stages[to].second); predecessor[to] = from; };
        int residual = backward_begin, pack = residual + 1, dpre = residual + 2, dx = residual + 3;
        int dw1 = residual + 4, dw2 = residual + 5, sum = residual + 6;
        edge(residual - 1, residual); edge(residual, pack); edge(pack, dpre); edge(pack, dw2);
        connect(dpre, stages[dx].first, stages[dw1].second); predecessor[dx] = predecessor[dw1] = dpre;
        edge(dx, sum); edge(sum, sum + 1); edge(sum + 1, sum + 2);
        connect(residual, stages[sum + 3].first, stages[sum + 4].second);
        predecessor[sum + 3] = predecessor[sum + 4] = residual;
    }
};

struct SuffixSchedule {
    TailSchedule tail;
    SuffixPlan plan;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    DeviceBuffer<Matmul> products;
    DeviceBuffer<HeadSetup> head_setup;
    DeviceBuffer<MLPSetup> mlp_setup;
    DeviceBuffer<ActivationPack> packs;
    DeviceBuffer<ResidualNorm> norms;
    DeviceBuffer<ResidualMix> mixes;
    DeviceBuffer<TailBackward> sums;
    DeviceBuffer<GradientCast> casts;
    explicit SuffixSchedule(SuffixStorage &b) : tail(b.tail), plan(b), tasks(plan.tasks.size()), groups(plan.groups.size()), counters(groups.n),
        queue(tasks.n), state(4), audit(tasks.n + 2), products(9), head_setup(1), mlp_setup(1), packs(2), norms(2), mixes(2), sums(3), casts(2) {
        tasks.put(plan.tasks); groups.put(plan.groups); configure(b, true);
    }
    void configure(SuffixStorage &b, bool diagnostics) {
        tail.configure(b.tail, diagnostics);
        auto ops = b.tail.head.products(diagnostics), mlp = b.products(diagnostics); ops.insert(ops.end(), mlp.begin(), mlp.end()); products.put(ops);
        head_setup.put({{products.p, b.tail.head.scales.p, b.tail.head.loss.parameters.p}});
        mlp_setup.put({{products.p + 3, b.scales.p, b.amax.p}});
        packs.put({{b.tail.sources[8]->p, b.x.p, b.xt.p, b.scales.p, b.tail.head.loss.tokens, 768, false, b.raw_norm.p},
                   {b.dy.p, b.g.p, b.gt.p, b.scales.p + 3, b.tail.head.loss.tokens, 768, true}});
        norms.put({b.tail.norm(diagnostics), b.norm(diagnostics)}); mixes.put({b.tail.mix(diagnostics), b.mix(diagnostics)});
        TailBackward a{}, c{}; a.tokens = c.tokens = b.tail.head.loss.tokens;
        a.dx_direct = b.dx.p; a.dx_network = b.tail.dsources[8]->p; a.dx = b.norm_dy.p;
        c.dx_direct = b.direct.p; c.dx_network = b.norm_dx.p; c.dx = b.input_gradient.p;
        sums.put({b.tail.backward(diagnostics), a, c});
        casts.put({{b.dr.p, b.dcoeff.p, b.tail.head.loss.tokens, 2}, {b.dp.p, b.dcoeff.p + 1, b.tail.head.loss.tokens, 2}});
    }
    Graph graph(bool checked = false) {
        Graph g = tail.graph(); g.ops = products.p; g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p; g.queue = queue.p;
        g.state = state.p; g.task_count = int(tasks.n); g.root_count = plan.stages[0].second; g.audit = checked ? audit.p : nullptr;
        g.head_setups = head_setup.p; g.mlp_setups = mlp_setup.p; g.activation_packs = packs.p; g.residual_norm = norms.p;
        g.residual_mix = mixes.p; g.tail_backward_ops = sums.p; g.gradient_casts = casts.p; return g;
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
        if (state.get()[2]) throw std::runtime_error("unfinished suffix tasks");
        for (size_t i = 0; i < tasks.n; ++i) if (visits[i] != 1) throw std::runtime_error("suffix task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i) if (completed[i] != plan.groups[i].expected) throw std::runtime_error("suffix dependency mismatch");
    }
};

template <int MinimumBlocks>
__global__ __launch_bounds__(128, MinimumBlocks) void suffix_stage(Graph g, int begin) {
    __shared__ __align__(1024) fp8 scratch[bf16_scratch_bytes];
    Task task = g.tasks[begin + blockIdx.x];
    if (task.kind == TaskKind::activation_pack) activation_pack(g.activation_packs[task.op], task.row, task.col, scratch);
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

struct SuffixControl {
    cudaGraph_t graph; cudaGraphExec_t exec;
    SuffixControl(SuffixSchedule &schedule, bool bounded) {
        CHECK_CUDA(cudaGraphCreate(&graph, 0)); std::vector<cudaGraphNode_t> nodes;
        for (size_t i = 0; i < schedule.plan.stages.size(); ++i) {
            auto [begin, end] = schedule.plan.stages[i]; Graph g = schedule.graph(); void *args[] = {&g, &begin};
            cudaKernelNodeParams p{}; p.func = bounded ? reinterpret_cast<void *>(suffix_stage<4>) : reinterpret_cast<void *>(suffix_stage<1>);
            p.gridDim = dim3(end - begin); p.blockDim = dim3(128); p.kernelParams = args;
            int previous = schedule.plan.predecessor[i]; cudaGraphNode_t node;
            CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, previous >= 0 ? &nodes[previous] : nullptr, previous >= 0 ? 1 : 0, &p)); nodes.push_back(node);
        }
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~SuffixControl() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run() { CHECK_CUDA(cudaGraphLaunch(exec, nullptr)); }
};

void check_suffix_math(SuffixStorage &b, SuffixSchedule &schedule) {
    CHECK_CUDA(cudaMemcpy(schedule.tail.head.products.p, schedule.products.p, 3 * sizeof(Matmul), cudaMemcpyDeviceToDevice));
    check_tail_math(b.tail, schedule.tail);
    auto ops = schedule.products.get(); LayerBlasReference blas;
    for (int id = 0; id < 6; ++id) {
        const auto &op = ops[id + 3];
        auto operand = [&](const fp8 *pointer, int rows, int cols, int stride_row, int stride_col, bool e5) {
            auto storage = layer_read(pointer, int64_t(rows - 1) * stride_row + int64_t(cols - 1) * stride_col + 1);
            std::vector<float> result(size_t(rows) * cols);
            for (int row = 0; row < rows; ++row) for (int col = 0; col < cols; ++col)
                result[row * cols + col] = decode_fp8(storage[row * stride_row + col * stride_col], e5);
            return result;
        };
        auto ref = blas.multiply(operand(op.a, op.m, op.k, op.ar, op.ak, op.a_e5), operand(op.b, op.n, op.k, op.br, op.bk, op.b_e5), op.m, op.n, op.k);
        for (auto &v : ref) v *= op.scale;
        auto raw = layer_read(op.raw, ref.size()); layer_near("suffix FP8 MLP product", raw, ref);
        auto saved = op.post ? layer_read(op.post, ref.size()) : std::vector<fp8>{};
        auto quantized = op.quantized ? layer_read(op.quantized, ref.size()) : std::vector<fp8>{};
        auto transposed = op.quantized_t ? layer_read(op.quantized_t, ref.size()) : std::vector<fp8>{};
        auto output = op.output ? layer_read(op.output, ref.size()) : std::vector<bf16>{};
        float maximum = 0;
        for (size_t i = 0; i < raw.size(); ++i) {
            float value = raw[i];
            if (op.epilogue == Epilogue::relu_square) { value = std::max(float(bf16(value)), 0.0f); value = float(bf16(value * value)); }
            else if (op.epilogue == Epilogue::relu_backward) value = float(bf16(2.0f * value * std::sqrt(float(saved[i]) * op.post_scale)));
            maximum = std::max(maximum, std::abs(value));
            if (!output.empty() && float(output[i]) != float(bf16(value))) throw std::runtime_error("suffix MLP BF16 output mismatch");
            if (!quantized.empty()) {
                auto expected = encode_fp8(value / op.output_scale, op.quantized_e5);
                if (quantized[i].__x != expected.__x || transposed[(i % op.n) * op.m + i / op.n].__x != expected.__x)
                    throw std::runtime_error("suffix MLP FP8 epilogue/transpose mismatch");
            }
        }
        if (op.amax && layer_read(op.amax, 1)[0] != maximum) throw std::runtime_error("suffix MLP activation maximum mismatch");
    }
    for (const auto &op : schedule.packs.get()) {
        auto input = op.raw_input ? layer_read(op.raw_input, size_t(op.tokens) * op.columns) : std::vector<float>{};
        auto bf = layer_read(op.input, size_t(op.tokens) * op.columns);
        auto row = layer_read(op.row, bf.size()), transposed = layer_read(op.transposed, bf.size());
        float inverse = 1.0f / layer_read(op.scale, 1)[0], limit = op.e5 ? 57344.0f : 448.0f;
        for (size_t i = 0; i < bf.size(); ++i) {
            float x = (input.empty() ? float(bf[i]) : input[i]) * inverse;
            auto expected = encode_fp8(std::min(limit, std::max(-limit, x)), op.e5);
            if (row[i].__x != expected.__x || transposed[(i % op.columns) * op.tokens + i / op.columns].__x != expected.__x)
                throw std::runtime_error("suffix activation packing mismatch");
        }
    }
    auto before = b.before.get(), normed = b.tail.sources[8]->get(), gains = b.coefficients.get(), output = b.output.get();
    auto incoming = b.tail.dx.get(), mlp_dy = b.dy.get(), direct = b.direct.get(), mlp_dx = b.dx.get(), tail_dx = b.tail.dsources[8]->get();
    auto norm_dy = b.norm_dy.get(), norm_dx = b.norm_dx.get(), result = b.input_gradient.get(), dcoeff = b.dcoeff.get();
    auto raw_norm = b.raw_norm.get(), raw_dnorm = b.raw_dnorm.get(), raw_mix = b.raw_mix.get(), dr = b.dr.get(), dp = b.dp.get();
    std::vector<double> rn(before.size()), rd(before.size()), rm(before.size()), cr(dr.size()), cp(dp.size());
    for (int row = 0; row < b.tail.head.loss.tokens; ++row) {
        double squares = 0, dot = 0;
        for (int c = 0; c < 768; ++c) {
            size_t i = size_t(row) * 768 + c;
            if (float(norm_dy[i]) != float(bf16(float(mlp_dx[i]) + float(tail_dx[i])))) throw std::runtime_error("shared normalized-input gradient missing a consumer");
            squares += double(float(before[i])) * float(before[i]); dot += double(float(before[i])) * float(norm_dy[i]);
            rm[i] = double(float(gains[row * 2])) * float(before[i]) + double(float(gains[row * 2 + 1])) * float(output[i]);
            cr[row] += double(float(incoming[i])) * float(before[i]); cp[row] += double(float(incoming[i])) * float(output[i]);
            if (float(direct[i]) != float(bf16(float(incoming[i]) * float(gains[row * 2]))) ||
                float(mlp_dy[i]) != float(bf16(float(incoming[i]) * float(gains[row * 2 + 1])))) throw std::runtime_error("suffix residual adjoint mismatch");
            if (float(result[i]) != float(bf16(float(direct[i]) + float(norm_dx[i])))) throw std::runtime_error("suffix input gradient mismatch");
        }
        double rstd = 1 / std::sqrt(squares / 768 + 0x1p-23);
        for (int c = 0; c < 768; ++c) {
            size_t i = size_t(row) * 768 + c;
            rn[i] = float(before[i]) * rstd; rd[i] = rstd * (float(norm_dy[i]) - float(before[i]) * dot / 768 * rstd * rstd);
            if (float(normed[i]) != float(bf16(raw_norm[i])) || float(norm_dx[i]) != float(bf16(raw_dnorm[i]))) throw std::runtime_error("suffix norm cast mismatch");
        }
        if (float(dcoeff[row * 2]) != float(bf16(dr[row])) || float(dcoeff[row * 2 + 1]) != float(bf16(dp[row]))) throw std::runtime_error("suffix coefficient gradient cast mismatch");
    }
    layer_near("suffix RMS", raw_norm, rn); layer_near("suffix RMS backward", raw_dnorm, rd);
    layer_near("suffix residual", raw_mix, rm); layer_near("suffix residual coefficient", dr, cr); layer_near("suffix MLP coefficient", dp, cp);
}

void check_suffix(int tokens, int vocabulary, int workers, int steps, bool timing = false, bool aten = true) {
    SuffixStorage b(tokens, vocabulary); SuffixSchedule schedule(b); SuffixControl control(schedule, false), bounded(schedule, true);
    for (int step = 0; step < steps; ++step) {
        b.change(step); b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify(); check_suffix_math(b, schedule);
        if (aten) check_tail_aten(b.tail.reference());
        std::vector<std::vector<bf16>> expected; for (auto *buffer : b.outputs()) expected.push_back(buffer->get());
        std::vector<std::vector<fp8>> packed; for (auto *buffer : b.packed_outputs()) packed.push_back(buffer->get());
        auto loss = b.tail.head.loss.loss.get(); auto head_dw = b.tail.head.dw.get(); auto maximum = b.amax.get();
        for (int mode = 0; mode < 4; ++mode) {
            b.poison(); schedule.configure(b, mode != 3);
            if (!mode) control.run(); else if (mode == 1) bounded.run(); else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize()); size_t index = 0;
            auto same = [](const auto &a, const auto &b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])); };
            for (auto *buffer : b.outputs()) if (!same(buffer->get(), expected[index++])) throw std::runtime_error("suffix BF16 replay mismatch");
            index = 0;
            for (auto *buffer : b.packed_outputs()) if (!same(buffer->get(), packed[index++])) throw std::runtime_error("suffix FP8 replay mismatch");
            if (!same(b.tail.head.loss.loss.get(), loss) || !same(b.tail.head.dw.get(), head_dw) || !same(b.amax.get(), maximum)) throw std::runtime_error("suffix loss/weight-gradient/maximum replay mismatch");
        }
        schedule.configure(b, true);
        printf("SUFFIX PASS tokens=%d vocabulary=%d workers=%d step=%d tasks=%zu phases=%zu\n", tokens, vocabulary, workers, step, schedule.tasks.n, schedule.plan.stages.size());
    }
    if (timing) {
        schedule.configure(b, false);
        device_time("SUFFIX CUDA Graph", [&] { control.run(); });
        device_time("SUFFIX CUDA Graph four-block occupancy", [&] { bounded.run(); });
        device_time("SUFFIX persistent+reset", [&] { schedule.run(workers); });
    }
}

int main(int argc, char **argv) {
    try {
        device_info(); bool quick = false, profile = false, aten = true;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--quick") quick = true;
            else if (arg == "--profile") profile = true;
            else if (arg == "--native-only") aten = false;
            else if (arg.rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown suffix argument");
        }
        cudaDeviceProp properties{}; CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        if (profile) {
            SuffixStorage b(256, 50304); b.change(0); SuffixSchedule schedule(b); schedule.configure(b, false);
            for (int i = 0; i < 5; ++i) schedule.run(properties.multiProcessorCount * 4);
            reset_loss<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, false, true, false, false, true, true><<<properties.multiProcessorCount * 4, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop()); return 0;
        }
        if (!aten) puts("Native-only checks: ATen diagnostic excluded, independent arithmetic and replay gates remain enabled");
        check_suffix(16, 2048, 17, 4, false, aten);
        if (!quick) {
            check_suffix(80, 2048, 1, 1, false, aten); check_suffix(16, 50304, 17, 1, false, aten);
            check_suffix(256, 50304, properties.multiProcessorCount * 4, 1, true, aten);
        }
        puts("PASS: final FP8 MLP, shared normalized-input adjoints, MUDD and head connected; full body and compiled parity remain"); return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

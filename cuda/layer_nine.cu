#define NANO_TRAINING_LAYER_NINE
#include "last_layer.cu"

struct LayerNineStorage {
    LastLayerStorage last;
    DeviceBuffer<bf16> input, bigram_gate, parameters, after_attention, normalized, output;
    DeviceBuffer<bf16> dx, dw1, dw2_unscaled, dw2, original_w2, direct, norm_dx, after_dy, input_gradient, bigram_gradient, total_bigram_gradient;
    DeviceBuffer<bf16> parameter_gradient, gate_gradient;
    DeviceBuffer<fp8> w1, w1t, w2, w2t, x, xt, g, gt, post, postt, dpre, dpret;
    DeviceBuffer<float> scales, amax, raw_norm, raw_dnorm, raw_before, raw_output, gate_partial, raw_parameters;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> raw_products, scalar_partial;
    LayerNineStorage(int t, int v) : last(t, v), input(size_t(t) * 768), bigram_gate(t), parameters(3), after_attention(input.n), normalized(input.n), output(input.n),
        dx(input.n), dw1(2816 * 768), dw2_unscaled(dw1.n), dw2(dw1.n), original_w2(dw1.n), direct(input.n), norm_dx(input.n), after_dy(input.n),
        input_gradient(input.n), bigram_gradient(input.n), total_bigram_gradient(input.n), parameter_gradient(3), gate_gradient(t),
        w1(dw1.n), w1t(dw1.n), w2(dw1.n), w2t(dw1.n), x(input.n), xt(input.n), g(input.n), gt(input.n), post(size_t(t) * 2816), postt(post.n),
        dpre(post.n), dpret(post.n), scales(6), amax(2), raw_norm(input.n), raw_dnorm(input.n), raw_before(input.n), raw_output(input.n), gate_partial(t), raw_parameters(3) {
        for (size_t n : {input.n, input.n, dw1.n}) scalar_partial.push_back(std::make_unique<DeviceBuffer<float>>((n + projection_elements - 1) / projection_elements));
        for (auto op : products(false)) raw_products.push_back(std::make_unique<DeviceBuffer<float>>(size_t(op.m) * op.n));
    }
    int tokens() const { return last.tokens(); }
    void change(int step) {
        last.change(step); std::mt19937 rng(11621 + step); std::normal_distribution<float> random;
        std::vector<bf16> in(input.n), bg(bigram_gate.n);
        for (auto &v : in) v = bf16(step == 2 ? 0 : 0.2f * random(rng));
        for (auto &v : bg) v = bf16(0.1f * random(rng));
        input.put(in); bigram_gate.put(bg);
        if (step == 2) CHECK_CUDA(cudaMemset(last.bigram.p, 0, last.bigram.n * sizeof(bf16)));
        parameters.put({bf16(step == 3 ? -0.75f : 1.125f), bf16(step == 3 ? -0.875f : 0.9375f), bf16(step == 1 ? 0 : step == 3 ? -0.6875f : 1.1875f)});
        std::vector<float> s{0.0625f, 0.0005f, 0.0004f, 0.015625f, 0.01f, 0.00002f};
        if (step == 3) { s[1] *= 2; s[2] *= 2; s[4] *= 2; s[5] *= 2; } scales.put(s);
        for (int id = 0; id < 2; ++id) {
            std::vector<bf16> original(dw1.n); std::vector<fp8> row(dw1.n), transposed(dw1.n);
            for (int r = 0; r < 2816; ++r) for (int c = 0; c < 768; ++c) {
                int i = r * 768 + c; original[i] = bf16(0.015f * random(rng));
                row[i] = fp8(float(original[i]) / s[id + 1]); transposed[c * 2816 + r] = row[i];
            }
            (id ? w2 : w1).put(row); (id ? w2t : w1t).put(transposed);
            if (id) original_w2.put(original);
        }
    }
    std::vector<Matmul> products(bool diagnostics) {
        MLPTrainingBuffers b{tokens(), w1.p, w1t.p, w2.p, w2t.p, x.p, xt.p, g.p, gt.p, post.p, postt.p, dpre.p, dpret.p,
                             output.p, dx.p, dw1.p, dw2_unscaled.p, amax.p};
        if (diagnostics) for (int i = 0; i < 6; ++i) b.raw[i] = raw_products[i]->p;
        return mlp_training_products(b);
    }
    std::vector<ResidualMix> mixes(bool diagnostics) {
        ResidualMix before{}; before.tokens = tokens(); before.count = 2; before.output = after_attention.p; before.dy = after_dy.p;
        before.raw = diagnostics ? raw_before.p : nullptr;
        before.terms[0].value = input.p; before.terms[0].coefficient = parameters.p;
        before.terms[1].value = last.bigram.p; before.terms[1].coefficient = bigram_gate.p; before.terms[1].coefficient_row_stride = 1;
        before.terms[1].dvalue = bigram_gradient.p; before.terms[1].dcoefficient_rows = gate_partial.p;
        ResidualMix after{}; after.tokens = tokens(); after.count = 2; after.output = last.source(2); after.raw = diagnostics ? raw_output.p : nullptr;
        after.terms[0].value = after_attention.p; after.terms[0].coefficient = parameters.p + 1;
        after.terms[1].value = output.p; after.terms[1].constant = 1;
        return {before, after};
    }
    ResidualNorm norm(bool diagnostics) { return {after_attention.p, dx.p, normalized.p, norm_dx.p, tokens(), nullptr, raw_norm.p, diagnostics ? raw_dnorm.p : nullptr}; }
    std::vector<ActivationPack> packs() {
        return {{normalized.p, x.p, xt.p, scales.p, tokens(), 768, false, raw_norm.p},
                {last.total[2]->p, g.p, gt.p, scales.p + 3, tokens(), 768, true}};
    }
    std::vector<ProjectionGradient> gradients() {
        const bf16 *value[3] = {input.p, after_attention.p, original_w2.p};
        const bf16 *dy[3] = {after_dy.p, last.total[2]->p, dw2_unscaled.p};
        bf16 *out[3] = {input_gradient.p, direct.p, dw2.p}; std::vector<ProjectionGradient> ops;
        for (int i = 0; i < 3; ++i) ops.push_back({value[i], dy[i], out[i], nullptr, scalar_partial[i]->p, raw_parameters.p + i, nullptr,
            int(i == 2 ? dw1.n : input.n), false, parameters.p + i, parameter_gradient.p + i});
        return ops;
    }
    std::vector<GradientSum> sums() {
        return {{{direct.p, norm_dx.p}, after_dy.p, int(input.n), 2},
                {{last.partial[10]->p, bigram_gradient.p}, total_bigram_gradient.p, int(input.n), 2}};
    }
    std::vector<DeviceBuffer<bf16> *> outputs() {
        auto out = last.outputs(); out.insert(out.end(), {&after_attention, &normalized, &output, &dx, &dw1, &dw2_unscaled, &dw2, &direct, &norm_dx,
            &after_dy, &input_gradient, &bigram_gradient, &total_bigram_gradient, &parameter_gradient, &gate_gradient, last.suffix.tail.sources[2].get()}); return out;
    }
    std::vector<DeviceBuffer<fp8> *> packed_outputs() {
        auto out = last.packed_outputs(); out.insert(out.end(), {&x, &xt, &g, &gt, &post, &postt, &dpre, &dpret}); return out;
    }
    void poison() {
        last.poison();
        for (auto *b : outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(bf16)));
        for (auto *b : packed_outputs()) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n));
        for (auto *b : {&amax, &raw_norm, &raw_dnorm, &raw_before, &raw_output, &gate_partial, &raw_parameters}) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
        for (auto *list : {&raw_products, &scalar_partial}) for (auto &b : *list) CHECK_CUDA(cudaMemset(b->p, 0xff, b->n * sizeof(float)));
    }
};

struct LayerNinePlan : LastLayerPlan {
    LayerNinePlan(LayerNineStorage &b, const LastLayerPlan &last) {
        int t = b.tokens(), rows = (t + 3) / 4;
        auto matrix = [&](int id) { auto op = b.products(false)[id]; return phase(TaskKind::matmul, id + 12, (op.m + 63) / 64, (op.n + 63) / 64); };
        auto gradient = [&](int id, bool reduce) { return phase(TaskKind::projection_gradient, id + 2,
            reduce ? 1 : int(b.scalar_partial[id]->n), 1, 0, int(reduce)); };
        phase(TaskKind::residual_mix, 5, rows); phase(TaskKind::residual_norm, 3, rows); phase(TaskKind::activation_pack, 3, (t + 63) / 64, 12);
        phase(TaskKind::mlp_setup, 1, 1); matrix(0); matrix(1); phase(TaskKind::residual_mix, 6, rows);
        for (int i = 1; i < int(stages.size()); ++i) release({i - 1}, i);
        int base = int(tasks.size()), phase_base = int(stages.size()), group_base = int(groups.size());
        for (auto task : last.tasks) { for (int &signal : task.signal) if (signal >= 0) signal += group_base; tasks.push_back(task); }
        for (auto group : last.groups) { group.begin += base; group.end += base; groups.push_back(group); }
        for (size_t i = 0; i < last.stages.size(); ++i) {
            auto range = last.stages[i]; stages.push_back({range.first + base, range.second + base});
            std::vector<int> p; for (int source : last.predecessor[i]) p.push_back(source + phase_base); predecessor.push_back(p);
        }
        release({phase_base - 1}, phase_base);
        int tail_done = int(stages.size()) - 1;
        int pack = phase(TaskKind::activation_pack, 4, (t + 63) / 64, 12), rm = gradient(1, false); release({tail_done}, pack, rm);
        int rm_reduce = gradient(1, true); release({rm}, rm_reduce);
        int dpre = matrix(2), dw2 = matrix(5); release({pack}, dpre, dw2);
        int dx = matrix(3), dw1 = matrix(4); release({dpre}, dx, dw1);
        int norm = phase(TaskKind::residual_norm, 3, rows, 1, 1); release({dx}, norm);
        int sum = phase(TaskKind::gradient_sum, 5, (t * 768 + 127) / 128); release({norm, rm}, sum);
        int ra = gradient(0, false), bg = phase(TaskKind::residual_mix, 5, rows, 1, 1, 1); release({sum}, ra, bg);
        int ra_reduce = gradient(0, true); release({ra}, ra_reduce);
        int cast = phase(TaskKind::gradient_cast, 16, (t + 127) / 128), bg_sum = phase(TaskKind::gradient_sum, 6, (t * 768 + 127) / 128); release({bg}, cast, bg_sum);
        int p = gradient(2, false), p_reduce = gradient(2, true); release({dw2}, p); release({p}, p_reduce);
    }
};

struct LayerNineSchedule {
    LastLayerSchedule last;
    LayerNinePlan plan;
    DeviceBuffer<Task> tasks;
    DeviceBuffer<Group> groups;
    DeviceBuffer<int> counters, queue, state, audit;
    DeviceBuffer<Matmul> products;
    DeviceBuffer<MLPSetup> mlp_setup;
    DeviceBuffer<HeadSetup> head_setup;
    DeviceBuffer<AttentionLayerSetup> layer_setup;
    DeviceBuffer<ActivationPack> packs;
    DeviceBuffer<ResidualNorm> norms;
    DeviceBuffer<ResidualMix> mixes;
    DeviceBuffer<ProjectionGradient> gradients;
    DeviceBuffer<GradientCast> casts;
    DeviceBuffer<GradientSum> sums;
    explicit LayerNineSchedule(LayerNineStorage &b) : last(b.last), plan(b, last.plan), tasks(plan.tasks.size()), groups(plan.groups.size()),
        counters(groups.n), queue(tasks.n), state(4), audit(tasks.n + 2), products(18), mlp_setup(2), head_setup(1), layer_setup(1), packs(5), norms(4), mixes(7), gradients(5), casts(17), sums(7) {
        tasks.put(plan.tasks); groups.put(plan.groups); configure(b, true);
    }
    void configure(LayerNineStorage &b, bool diagnostics) {
        last.configure(b.last, diagnostics);
        auto p = last.products.get(), extra = b.products(diagnostics); p.insert(p.end(), extra.begin(), extra.end()); products.put(p);
        head_setup.put({{products.p, b.last.suffix.tail.head.scales.p, b.last.suffix.tail.head.loss.parameters.p}});
        mlp_setup.put({{products.p + 3, b.last.suffix.scales.p, b.last.suffix.amax.p}, {products.p + 12, b.scales.p, b.amax.p, b.parameters.p + 2}});
        auto ls = last.layer_setup.get(); ls[0].ops = products.p + 9; layer_setup.put(ls);
        auto pack = last.packs.get(), ep = b.packs(); pack.insert(pack.end(), ep.begin(), ep.end()); packs.put(pack);
        auto norm = last.norms.get(); norm.push_back(b.norm(diagnostics)); norms.put(norm);
        auto mix = last.mixes.get(), em = b.mixes(diagnostics); mix.insert(mix.end(), em.begin(), em.end()); mixes.put(mix);
        auto grad = last.attention.gradient.get(), eg = b.gradients(); grad.insert(grad.end(), eg.begin(), eg.end()); gradients.put(grad);
        auto cast = last.casts.get(); cast.push_back({b.gate_partial.p, b.gate_gradient.p, b.tokens(), 1}); casts.put(cast);
        auto sum = last.sums.get(), es = b.sums(); sum.insert(sum.end(), es.begin(), es.end()); sums.put(sum);
    }
    Graph graph(bool checked = false) {
        Graph g = last.graph(); g.ops = products.p; g.tasks = tasks.p; g.groups = groups.p; g.counters = counters.p; g.queue = queue.p;
        g.state = state.p; g.task_count = int(tasks.n); g.root_count = plan.stages[0].second; g.audit = checked ? audit.p : nullptr;
        g.head_setups = head_setup.p; g.mlp_setups = mlp_setup.p; g.layer_setup = layer_setup.p; g.activation_packs = packs.p; g.residual_norm = norms.p;
        g.residual_mix = mixes.p; g.projection_gradient = gradients.p; g.gradient_casts = casts.p; g.gradient_sums = sums.p; return g;
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
        if (state.get()[2]) throw std::runtime_error("unfinished layer-nine tasks");
        for (size_t i = 0; i < tasks.n; ++i) if (visits[i] != 1) throw std::runtime_error("layer-nine task visit mismatch");
        for (size_t i = 0; i < groups.n; ++i) if (completed[i] != plan.groups[i].expected) throw std::runtime_error("layer-nine dependency mismatch");
    }
};

void check_layer_nine_math(LayerNineStorage &b, LayerNineSchedule &schedule) {
    CHECK_CUDA(cudaMemcpy(schedule.last.products.p, schedule.products.p, 12 * sizeof(Matmul), cudaMemcpyDeviceToDevice));
    check_last_layer_math(b.last, schedule.last);
    LayerBlasReference blas; auto ops = schedule.products.get(); auto scales = b.scales.get(); auto parameters = b.parameters.get();
    if (ops[13].scale != scales[4] * (scales[2] * float(parameters[2])) ||
        ops[14].scale != (scales[2] * scales[3]) * float(parameters[2]) || ops[14].post_scale != scales[4] || ops[17].scale != scales[4] * scales[3])
        throw std::runtime_error("layer-nine folded scale or unscaled weight gradient mismatch");
    for (int id = 0; id < 6; ++id) {
        const auto &op = ops[id + 12];
        auto operand = [&](const fp8 *pointer, int rows, int cols, int sr, int sc, bool e5) {
            auto storage = layer_read(pointer, int64_t(rows - 1) * sr + int64_t(cols - 1) * sc + 1); std::vector<float> out(size_t(rows) * cols);
            for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) out[r * cols + c] = decode_fp8(storage[r * sr + c * sc], e5); return out;
        };
        auto expected = blas.multiply(operand(op.a, op.m, op.k, op.ar, op.ak, op.a_e5), operand(op.b, op.n, op.k, op.br, op.bk, op.b_e5), op.m, op.n, op.k);
        for (auto &x : expected) x *= op.scale;
        auto raw = layer_read(op.raw, expected.size()); layer_near("layer-nine FP8 product", raw, expected);
        auto saved = op.post ? layer_read(op.post, raw.size()) : std::vector<fp8>{};
        auto quantized = op.quantized ? layer_read(op.quantized, raw.size()) : std::vector<fp8>{};
        auto transposed = op.quantized_t ? layer_read(op.quantized_t, raw.size()) : std::vector<fp8>{};
        auto out = op.output ? layer_read(op.output, raw.size()) : std::vector<bf16>{}; float maximum = 0;
        for (size_t i = 0; i < raw.size(); ++i) {
            float value = raw[i];
            if (op.epilogue == Epilogue::relu_square) { value = std::max(float(bf16(value)), 0.0f); value = float(bf16(value * value)); }
            else if (op.epilogue == Epilogue::relu_backward) value = float(bf16(2.0f * value * std::sqrt(float(saved[i]) * op.post_scale)));
            maximum = std::max(maximum, std::abs(value));
            if (!out.empty() && float(out[i]) != float(bf16(value))) throw std::runtime_error("layer-nine MLP BF16 epilogue mismatch");
            if (!quantized.empty()) {
                auto q = encode_fp8(value / op.output_scale, op.quantized_e5);
                if (quantized[i].__x != q.__x || transposed[(i % op.n) * op.m + i / op.n].__x != q.__x) throw std::runtime_error("layer-nine MLP FP8 epilogue/transpose mismatch");
            }
        }
        if (op.amax && layer_read(op.amax, 1)[0] != maximum) throw std::runtime_error("layer-nine activation maximum mismatch");
    }
    for (auto op : b.packs()) {
        auto input = layer_read(op.input, size_t(op.tokens) * op.columns);
        auto raw = op.raw_input ? layer_read(op.raw_input, input.size()) : std::vector<float>{};
        auto row = layer_read(op.row, input.size()), transposed = layer_read(op.transposed, input.size()); float inverse = 1 / layer_read(op.scale, 1)[0];
        float limit = op.e5 ? 57344 : 448;
        for (size_t i = 0; i < input.size(); ++i) {
            auto expected = encode_fp8(std::min(limit, std::max(-limit, (raw.empty() ? float(input[i]) : raw[i]) * inverse)), op.e5);
            if (row[i].__x != expected.__x || transposed[(i % op.columns) * op.tokens + i / op.columns].__x != expected.__x) throw std::runtime_error("layer-nine input/gradient packing mismatch");
        }
    }
    auto in = b.input.get(), bg = b.last.bigram.get(), gate = b.bigram_gate.get(), before = b.after_attention.get(), mlp = b.output.get(), result = b.last.suffix.tail.sources[2]->get();
    auto normed = b.normalized.get(), dx = b.dx.get(), ndx = b.norm_dx.get(), direct = b.direct.get(), dy = b.after_dy.get();
    auto raw_before = b.raw_before.get(), raw_out = b.raw_output.get(), raw_norm = b.raw_norm.get(), raw_dnorm = b.raw_dnorm.get();
    auto dbg = b.bigram_gradient.get(), total_bg = b.total_bigram_gradient.get(), last_dbg = b.last.partial[10]->get();
    auto dg = b.gate_gradient.get(); auto gate_partial = b.gate_partial.get();
    std::vector<double> rb(in.size()), ro(in.size()), rn(in.size()), rd(in.size()), rg(b.tokens());
    for (int row = 0; row < b.tokens(); ++row) {
        double square = 0, dot = 0;
        for (int c = 0; c < 768; ++c) {
            size_t i = size_t(row) * 768 + c;
            rb[i] = double(float(parameters[0])) * float(in[i]) + double(float(gate[row])) * float(bg[i]);
            ro[i] = double(float(parameters[1])) * float(before[i]) + float(mlp[i]);
            square += double(float(before[i])) * float(before[i]); dot += double(float(before[i])) * float(dx[i]);
            rg[row] += double(float(bg[i])) * float(dy[i]);
            if (float(dy[i]) != float(bf16(float(direct[i]) + float(ndx[i])))) throw std::runtime_error("layer-nine residual/MLP gradient join mismatch");
            if (float(dbg[i]) != float(bf16(float(gate[row]) * float(dy[i])))) throw std::runtime_error("layer-nine bigram source gradient mismatch");
            if (float(total_bg[i]) != float(bf16(float(last_dbg[i]) + float(dbg[i])))) throw std::runtime_error("layer-nine shared bigram gradient missing a consumer");
            if (float(before[i]) != float(bf16(raw_before[i])) || float(result[i]) != float(bf16(raw_out[i]))) throw std::runtime_error("layer-nine residual output rounding mismatch");
        }
        double rstd = 1 / std::sqrt(square / 768 + 0x1p-23);
        for (int c = 0; c < 768; ++c) {
            size_t i = size_t(row) * 768 + c; rn[i] = float(before[i]) * rstd; rd[i] = rstd * (float(dx[i]) - float(before[i]) * dot / 768 * rstd * rstd);
            if (float(normed[i]) != float(bf16(raw_norm[i])) || float(ndx[i]) != float(bf16(raw_dnorm[i]))) throw std::runtime_error("layer-nine norm rounding mismatch");
        }
        if (float(dg[row]) != float(bf16(gate_partial[row]))) throw std::runtime_error("layer-nine bigram gate gradient cast mismatch");
    }
    layer_near("layer-nine pre-MLP residual", raw_before, rb); layer_near("layer-nine post-MLP residual", raw_out, ro);
    layer_near("layer-nine RMS", raw_norm, rn); layer_near("layer-nine RMS backward", raw_dnorm, rd); layer_near("layer-nine bigram gate", gate_partial, rg);
    std::vector<double> scalar(3, 0); auto pg = b.parameter_gradient.get(); auto raw_pg = b.raw_parameters.get();
    // The folded multiplier's derivative uses the original BF16 W2 and the unscaled BF16 dW2, even when p is zero.
    std::vector<std::vector<bf16>> values{in, before, b.original_w2.get()}, incoming{dy, b.last.total[2]->get(), b.dw2_unscaled.get()};
    std::vector<std::vector<bf16>> gradients{b.input_gradient.get(), direct, b.dw2.get()};
    for (int id = 0; id < 3; ++id) {
        for (size_t i = 0; i < values[id].size(); ++i) {
            scalar[id] += double(float(values[id][i])) * float(incoming[id][i]);
            if (float(gradients[id][i]) != float(bf16(float(incoming[id][i]) * float(parameters[id])))) throw std::runtime_error("layer-nine scalar-scaled adjoint mismatch");
        }
        if (float(pg[id]) != float(bf16(raw_pg[id]))) throw std::runtime_error("layer-nine scalar gradient cast mismatch");
    }
    layer_near("layer-nine scalar gradients", raw_pg, scalar);
    if (float(parameters[2]) == 0 && (std::abs(scalar[2]) < 1e-8 || float(pg[2]) == 0)) throw std::runtime_error("zero-fold fixture failed to exercise nonzero multiplier gradient");
}

void check_layer_nine(int tokens, int vocabulary, int workers, int steps, bool timing, bool aten) {
    LayerNineStorage b(tokens, vocabulary); LayerNineSchedule schedule(b); LastLayerControl control(schedule, false), bounded(schedule, true);
    for (int step = 0; step < steps; ++step) {
        b.change(step); b.poison(); schedule.run(workers, true); CHECK_CUDA(cudaDeviceSynchronize()); schedule.verify(); check_layer_nine_math(b, schedule);
        if (aten) check_tail_aten(b.last.suffix.tail.reference());
        std::vector<std::vector<bf16>> expected; for (auto *buffer : b.outputs()) expected.push_back(buffer->get());
        std::vector<std::vector<fp8>> packed; for (auto *buffer : b.packed_outputs()) packed.push_back(buffer->get());
        auto loss = b.last.suffix.tail.head.loss.loss.get(), maximum = b.amax.get(), last_maximum = b.last.suffix.amax.get(), gains = b.last.attention.gains.get();
        auto head_dw = b.last.suffix.tail.head.dw.get(); auto scalar = b.raw_parameters.get();
        for (int mode = 0; mode < 4; ++mode) {
            b.poison(); schedule.configure(b, mode != 3);
            if (!mode) control.run(); else if (mode == 1) bounded.run(); else schedule.run(workers);
            CHECK_CUDA(cudaDeviceSynchronize()); size_t index = 0;
            auto same = [](const auto &a, const auto &b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])); };
            for (auto *buffer : b.outputs()) { if (!same(buffer->get(), expected[index])) throw std::runtime_error("layer-nine BF16 replay mismatch at buffer " + std::to_string(index)); ++index; }
            index = 0; for (auto *buffer : b.packed_outputs()) if (!same(buffer->get(), packed[index++])) throw std::runtime_error("layer-nine FP8 replay mismatch");
            if (!same(b.last.suffix.tail.head.loss.loss.get(), loss) || !same(b.last.suffix.tail.head.dw.get(), head_dw) || !same(b.amax.get(), maximum) ||
                !same(b.last.suffix.amax.get(), last_maximum) || !same(b.last.attention.gains.get(), gains) || !same(b.raw_parameters.get(), scalar))
                throw std::runtime_error("layer-nine loss/maximum/scalar replay mismatch");
        }
        schedule.configure(b, true);
        printf("LAYER_NINE PASS tokens=%d vocabulary=%d workers=%d step=%d tasks=%zu phases=%zu\n", tokens, vocabulary, workers, step, schedule.tasks.n, schedule.plan.stages.size());
    }
    if (timing) {
        schedule.configure(b, false);
        device_time("LAYER_NINE CUDA Graph", [&] { control.run(); });
        device_time("LAYER_NINE CUDA Graph four-block occupancy", [&] { bounded.run(); });
        device_time("LAYER_NINE persistent+reset", [&] { schedule.run(workers); });
    }
}

int main(int argc, char **argv) {
    try {
        device_info(); bool quick = false, profile = false, aten = true;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--quick") quick = true; else if (arg == "--profile") profile = true; else if (arg == "--native-only") aten = false;
            else if (arg.rfind("--gradient-chunk=", 0) != 0) throw std::runtime_error("unknown layer-nine argument");
        }
        cudaDeviceProp properties{}; CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        if (profile) {
            LayerNineStorage b(256, 50304); b.change(0); LayerNineSchedule schedule(b); schedule.configure(b, false);
            for (int i = 0; i < 5; ++i) schedule.run(properties.multiProcessorCount * 4);
            reset_loss<<<(schedule.tasks.n + 127) / 128, 128>>>(schedule.graph(), int(schedule.groups.n));
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStart());
            megakernel<false, false, true, true, false, true, true, true><<<properties.multiProcessorCount * 4, 128>>>(schedule.graph());
            CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaProfilerStop()); return 0;
        }
        if (!aten) puts("Native-only checks: ATen tail diagnostic excluded; native arithmetic and replay gates remain enabled");
        check_layer_nine(16, 2048, 17, 4, false, aten);
        if (!quick) {
            check_layer_nine(80, 2048, 1, 1, false, aten); check_layer_nine(16, 50304, 17, 1, false, aten);
            check_layer_nine(256, 50304, properties.multiProcessorCount * 4, 1, true, aten);
        }
        puts("PASS: folded layer-nine MLP and scalar/bigram adjoints connected through last-layer loss; earlier body and compiled parity remain"); return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

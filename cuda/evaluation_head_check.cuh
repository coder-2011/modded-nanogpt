#pragma once
#include "evaluation_head.cuh"

struct EvaluationHeadStorage {
    int t;
    DeviceBuffer<bf16> weight, logits, capped;
    DeviceBuffer<int32_t> targets;
    DeviceBuffer<float> raw, partial, target_logit, loss;
    explicit EvaluationHeadStorage(int tokens) : t(tokens), weight(size_t(768) * 50304),
        logits(tokens <= 80 ? size_t(tokens) * 50304 : 0), capped(logits.n), targets(tokens),
        raw(logits.n), partial(size_t(tokens) * 786), target_logit(tokens), loss(tokens) {}
    void change(int step) {
        std::vector<bf16> weights(weight.n);
        for (size_t i = 0; i < weights.size(); ++i) {
            uint32_t h = uint32_t(i) * 2654435761u + uint32_t(step) * 2246822519u;
            h ^= h >> 13;
            float value = float(int(h % 255) - 127) / 4096.0f;
            if (i % 50304 == 0) value = 4.0f;
            if (i % 50304 == 50303) value = -4.0f;
            weights[i] = bf16(step == 2 ? 0.0f : value);
        }
        weight.put(weights);
        std::vector<int32_t> ids(t);
        const int special[] = {0, 63, 64, 50256, 50303};
        for (int i = 0; i < t; ++i) ids[i] = i % 7 < 5 ? special[(i + step) % 5] : (i * 997 + step * 131) % 50304;
        targets.put(ids);
    }
    EvaluationHead buffers() { return {targets.p, partial.p, target_logit.p, loss.p, t, 50304, capped.p}; }
};

void check_evaluation_head(const BF16Matmul &matrix, const EvaluationHead &head) {
    auto targets = layer_read(head.targets, head.tokens);
    auto losses = layer_read(head.loss, head.tokens);
    LayerBlasReference reference;
    if (matrix.raw) {
        reference.check_product(matrix);
        auto raw = layer_read(matrix.raw, size_t(matrix.m) * matrix.n);
        auto capped = layer_read(head.softcapped, raw.size());
        std::vector<double> partial(size_t(head.tokens) * 786), target(head.tokens), expected(head.tokens);
        for (int row = 0; row < head.tokens; ++row) {
            double sum = 0;
            for (int col = 0; col < head.vocabulary; ++col) {
                size_t i = size_t(row) * head.vocabulary + col;
                double z = 23.0 / (1.0 + std::exp(-(double(float(bf16(raw[i]))) + 5.0) / 7.5));
                // Bound FP32 transcendental error at a BF16 rounding midpoint;
                // this is the existing 2e-5 absolute arithmetic tolerance.
                float low = float(bf16(float(z - 2e-5))), high = float(bf16(float(z + 2e-5)));
                if (float(capped[i]) < low || float(capped[i]) > high)
                    throw std::runtime_error("head softcap/rounding mismatch");
                double term = std::exp(double(float(capped[i])) - 23.0);
                partial[size_t(row) * 786 + col / 64] += term; sum += term;
            }
            target[row] = float(capped[size_t(row) * head.vocabulary + targets[row]]);
            expected[row] = std::log(sum) + 23.0 - target[row];
        }
        layer_near("head per-tile exponential sums", layer_read(head.partial, partial.size()), partial);
        layer_near("head selected full-vocabulary targets", layer_read(head.target_logit, target.size()), target);
        layer_near("head full-vocabulary reduction", losses, expected);
    }
    auto input = reference.operand(matrix.a, matrix.m, matrix.k, matrix.ar, matrix.ak);
    auto weight = reference.operand(matrix.b, matrix.n, matrix.k, matrix.br, matrix.bk);
    DeviceBuffer<float> dx(input.size()), dw(weight.size()), logits(size_t(64) * matrix.n);
    dx.put(input); dw.put(weight);
    std::vector<double> expected(head.tokens);
    float one = 1, zero = 0;
    for (int begin = 0; begin < head.tokens; begin += 64) {
        int rows = std::min(64, head.tokens - begin);
        LayerBlasReference::check(cublasSgemm(reference.handle, CUBLAS_OP_T, CUBLAS_OP_N,
            matrix.n, rows, matrix.k, &one, dw.p, matrix.k, dx.p + int64_t(begin) * matrix.k,
            matrix.k, &zero, logits.p, matrix.n));
        auto product = layer_read(logits.p, size_t(rows) * matrix.n);
        for (int row = 0; row < rows; ++row) {
            double sum = 0, target = 0;
            for (int col = 0; col < matrix.n; ++col) {
                double logit = float(bf16(product[size_t(row) * matrix.n + col]));
                double cap = float(bf16(float(23.0 / (1.0 + std::exp(-(logit + 5.0) / 7.5)))));
                sum += std::exp(cap - 23.0);
                if (col == targets[begin + row]) target = cap;
            }
            expected[begin + row] = std::log(sum) + 23.0 - target;
        }
    }
    layer_near("head pedantic-cuBLAS and FP64 full-vocabulary CE", losses, expected);
    printf("HEAD PASS tokens=%d vocabulary=50304 tiles_per_row=786 materialized_logits=%d\n", head.tokens, matrix.raw != nullptr);
}

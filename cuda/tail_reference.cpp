#include "tail_reference.h"
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

void check_tail_aten(const TailReferenceView &v) {
    auto tensor = [](const void *pointer, at::IntArrayRef shape, bool gradient = true) {
        auto t = at::from_blob(const_cast<void *>(pointer), shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
        return t.set_requires_grad(gradient);
    };
    auto x = tensor(v.x, {v.tokens, 768});
    std::vector<at::Tensor> leaves{x};
    for (auto pointer : v.sources) leaves.push_back(tensor(pointer, {v.tokens, 768}));
    auto w1 = tensor(v.w1, {64, 768}), w2 = tensor(v.w2, {14, 64});
    auto bias = tensor(v.bias, {14}), wg = tensor(v.wg, {14, 12, 64});
    leaves.insert(leaves.end(), {w1, w2, bias, wg});
    auto hidden = at::gelu(at::linear(x, w1), "none");
    auto mu = (at::linear(hidden, w2.narrow(0, 0, 10)) + bias.narrow(0, 0, 10)) * 0.1;
    auto mug = (at::linear(hidden, wg.narrow(0, 0, 10).reshape({120, 64})) * 0.1).reshape({v.tokens, 10, 12});
    auto mixed = x.reshape({v.tokens, 12, 64});
    for (int i = 0; i < 10; ++i) {
        auto coefficient = mu.select(1, i).unsqueeze(-1) + mug.select(1, i);
        mixed = mixed + coefficient.unsqueeze(-1) * leaves[i + 1].reshape({v.tokens, 12, 64});
    }
    auto output = at::rms_norm(mixed.reshape({v.tokens, 768}), {768}, std::nullopt, std::nullopt);
    auto gradients = torch::autograd::grad({output}, leaves, {tensor(v.dy, {v.tokens, 768}, false)});
    auto compare = [&](const char *name, const void *actual, const at::Tensor &expected) {
        auto a = tensor(actual, expected.sizes(), false).to(at::kFloat).cpu().contiguous();
        auto b = expected.detach().to(at::kFloat).cpu().contiguous();
        const float *pa = a.const_data_ptr<float>(), *pb = b.const_data_ptr<float>();
        double error = 0, norm = 0, maximum = 0;
        int64_t changed = 0;
        for (int64_t i = 0; i < a.numel(); ++i) {
            if (!std::isfinite(pa[i]) || !std::isfinite(pb[i])) throw std::runtime_error("non-finite tail ATen comparison");
            double d = double(pa[i]) - pb[i]; error += d * d; norm += double(pb[i]) * pb[i];
            maximum = std::max(maximum, std::abs(d)); changed += pa[i] != pb[i];
        }
        printf("TAIL ATen eager diagnostic %s rel_l2=%.9g max_abs=%.9g changed=%lld/%lld\n",
               name, std::sqrt(error / std::max(norm, 1e-30)), maximum, (long long)changed, (long long)a.numel());
    };
    compare("mu", v.mu, mu); compare("mug", v.mug, mug);
    compare("mixed", v.mixed, mixed.reshape({v.tokens, 768}));
    compare("RMS from native mixed", v.output,
            at::rms_norm(tensor(v.mixed, {v.tokens, 768}, false), {768}, std::nullopt, std::nullopt));
    compare("output", v.output, output); compare("dx", v.dx, gradients[0]);
    for (int i = 0; i < 10; ++i) {
        char name[32]; std::snprintf(name, sizeof(name), "source_%d", i);
        compare(name, v.dsources[i], gradients[i + 1]);
    }
    compare("dw1", v.dw1, gradients[11]); compare("dw2", v.dw2, gradients[12]);
    compare("dbias", v.dbias, gradients[13]); compare("dwg", v.dwg, gradients[14]);
}

#ifdef NANO_TAIL_ATEN_PROBE
int main() {
    try {
        auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
        auto x = at::ones({16, 768}, options).set_requires_grad(true);
        auto w = at::ones({64, 768}, options).set_requires_grad(true);
        puts("ATen-only probe: no native megakernel, forward linear/GELU and autograd"); std::fflush(stdout);
        auto y = at::gelu(at::linear(x, w), "none");
        auto gradients = torch::autograd::grad({y}, {x, w}, {at::ones_like(y)});
        if (!(y == 768).all().item<bool>() || !(gradients[0] == 64).all().item<bool>() ||
            !(gradients[1] == 16).all().item<bool>()) throw std::runtime_error("ATen-only probe arithmetic mismatch");
        puts("PASS: ATen-only probe"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
#endif

#pragma once
#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <vector>

namespace nano {

struct AttentionReference {
    std::vector<double> y, dq, dk, dv, lse, delta;
    template <class Inputs> explicit AttentionReference(const Inputs &in, const std::vector<__nv_bfloat16> *saved_y = nullptr)
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
                        delta[si] += double(float(saved_y ? (*saved_y)[vi + c] : __nv_bfloat16(float(y[vi + c])))) * float(in.dy[vi + c]);
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

} // namespace nano

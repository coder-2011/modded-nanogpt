#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        auto e = (call);                                                                           \
        if (e != cudaSuccess)                                                                      \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e));           \
    } while (0)

template <class T> struct DeviceBuffer {
    T *p = nullptr;
    size_t n;
    int device;
    explicit DeviceBuffer(size_t count) : n(count) {
        CHECK_CUDA(cudaGetDevice(&device));
        if (n) CHECK_CUDA(cudaMalloc(&p, n * sizeof(T)));
    }
    ~DeviceBuffer() {
        int current;
        cudaGetDevice(&current);
        cudaSetDevice(device);
        cudaFree(p);
        cudaSetDevice(current);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    void put(const std::vector<T> &v) {
        if (v.size() != n)
            throw std::runtime_error("size mismatch");
        if (n) CHECK_CUDA(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(n);
        if (n) CHECK_CUDA(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

inline double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return (x[(x.size() - 1) / 2] + x[x.size() / 2]) * 0.5;
}
template <class F> double device_time(const char *name, F f) {
    cudaEvent_t a, b;
    CHECK_CUDA(cudaEventCreate(&a));
    CHECK_CUDA(cudaEventCreate(&b));
    std::vector<double> samples;
    for (int i = 0; i < 15; ++i) {
        CHECK_CUDA(cudaEventRecord(a));
        f();
        CHECK_CUDA(cudaEventRecord(b));
        CHECK_CUDA(cudaEventSynchronize(b));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
        if (i >= 5)
            samples.push_back(ms);
    }
    double result = median(samples);
    printf("%s median_ms=%.6f samples_ms=", name, result);
    for (double x : samples)
        printf(" %.6f", x);
    puts("");
    CHECK_CUDA(cudaEventDestroy(a));
    CHECK_CUDA(cudaEventDestroy(b));
    return result;
}
inline void device_info() {
    int count;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        CHECK_CUDA(cudaGetDeviceProperties(&p, i));
        printf("GPU[%d]=%s SMs=%d runtime=%d\n", i, p.name, p.multiProcessorCount, CUDART_VERSION);
    }
}

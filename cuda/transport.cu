#include "bench.cuh"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cub/device/device_scan.cuh>
#include <cuda_bf16.h>
#include <nccl.h>
#include <random>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#define NCCL(call)                                                                                 \
    do {                                                                                           \
        auto e = (call);                                                                           \
        if (e != ncclSuccess)                                                                      \
            throw std::runtime_error(ncclGetErrorString(e));                                       \
    } while (0)

struct Packed {
    uint64_t *planes;
    uint8_t *payload;
    uint32_t *offsets;
    int *base;
    int n;
};
__global__ void exponent_histogram(const uint16_t *input, int n, unsigned *histogram) {
    __shared__ unsigned local[256];
    local[threadIdx.x] = 0;
    __syncthreads();
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        atomicAdd(local + ((input[i] >> 7) & 255), 1);
    __syncthreads();
    atomicAdd(histogram + threadIdx.x, local[threadIdx.x]);
}
__global__ void choose_exponents(const unsigned *histogram, int *base) {
    uint64_t score = 0;
    if (threadIdx.x <= 249) {
        for (int j = 0; j < 7; ++j)
            score += histogram[threadIdx.x + j];
        score = score * 256 + (255 - threadIdx.x);
    }
    for (int d = 16; d; d /= 2)
        score = max(score, __shfl_down_sync(~0u, score, d));
    __shared__ uint64_t best[8];
    if (threadIdx.x % 32 == 0)
        best[threadIdx.x / 32] = score;
    __syncthreads();
    if (threadIdx.x < 32) {
        score = threadIdx.x < 8 ? best[threadIdx.x] : 0;
        for (int d = 16; d; d /= 2)
            score = max(score, __shfl_down_sync(~0u, score, d));
        if (threadIdx.x == 0)
            *base = 254 - int(score & 255);
    }
}
__global__ void tile_sizes(const uint16_t *input, Packed p, uint32_t *sizes) {
    int index = blockIdx.x * 64 + threadIdx.x;
    int code = index < p.n ? int((input[index] >> 7) & 255) - *p.base : 0;
    code = code >= 1 && code <= 7 ? code : 0;
    __shared__ uint32_t planes[6];
    for (int bit = 0; bit < 3; ++bit) {
        uint32_t mask = __ballot_sync(~0u, (code >> bit) & 1);
        if (threadIdx.x % 32 == 0)
            planes[2 * bit + threadIdx.x / 32] = mask;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t any = 0;
        for (int bit = 0; bit < 3; ++bit) {
            uint64_t word = uint64_t(planes[2 * bit]) | (uint64_t(planes[2 * bit + 1]) << 32);
            p.planes[blockIdx.x * 3 + bit] = word;
            any |= word;
        }
        int valid = min(64, p.n - int(blockIdx.x) * 64);
        sizes[blockIdx.x] = 2 * valid - __popcll(any);
        if (blockIdx.x == gridDim.x - 1)
            sizes[gridDim.x] = 0;
    }
}
__global__ void pack_values(const uint16_t *input, Packed p) {
    int i = blockIdx.x * 64 + threadIdx.x;
    if (i >= p.n)
        return;
    uint64_t mask =
        p.planes[blockIdx.x * 3] | p.planes[blockIdx.x * 3 + 1] | p.planes[blockIdx.x * 3 + 2];
    uint64_t before = threadIdx.x ? (uint64_t(1) << threadIdx.x) - 1 : 0;
    unsigned rank = __popcll(mask & before), offset = p.offsets[blockIdx.x];
    uint16_t value = input[i];
    if ((mask >> threadIdx.x) & 1)
        p.payload[offset + rank] = uint8_t((value & 127) | ((value >> 8) & 128));
    else {
        offset += __popcll(mask) + 2 * (threadIdx.x - rank);
        p.payload[offset] = uint8_t(value);
        p.payload[offset + 1] = uint8_t(value >> 8);
    }
}
__device__ uint16_t decode(Packed p, int i) {
    int tile = i / 64, lane = i % 64;
    uint64_t b0 = p.planes[tile * 3], b1 = p.planes[tile * 3 + 1], b2 = p.planes[tile * 3 + 2];
    uint64_t mask = b0 | b1 | b2, before = lane ? (uint64_t(1) << lane) - 1 : 0;
    unsigned rank = __popcll(mask & before), offset = p.offsets[tile];
    int code = ((b0 >> lane) & 1) | (((b1 >> lane) & 1) << 1) | (((b2 >> lane) & 1) << 2);
    if (code) {
        uint16_t v = p.payload[offset + rank];
        return ((v & 128) << 8) | ((*p.base + code) << 7) | (v & 127);
    }
    offset += __popcll(mask) + 2 * (lane - rank);
    return uint16_t(p.payload[offset]) | (uint16_t(p.payload[offset + 1]) << 8);
}
__global__ void decode_add(Packed p, const uint16_t *other, uint16_t *output) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < p.n) {
        uint16_t v = decode(p, i);
        output[i] = other ? __bfloat16_as_ushort(__float2bfloat16_rn(
                                __bfloat162float(__ushort_as_bfloat16(v)) +
                                __bfloat162float(__ushort_as_bfloat16(other[i]))))
                          : v;
    }
}
__global__ void dense_add(const uint16_t *a, const uint16_t *b, uint16_t *out, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        out[i] =
            __bfloat16_as_ushort(__float2bfloat16_rn(__bfloat162float(__ushort_as_bfloat16(a[i])) +
                                                     __bfloat162float(__ushort_as_bfloat16(b[i]))));
}
struct Codec {
    DeviceBuffer<uint64_t> planes;
    DeviceBuffer<uint8_t> payload;
    DeviceBuffer<uint32_t> offsets, sizes, histogram;
    DeviceBuffer<int> base;
    void *temp = nullptr;
    size_t temp_bytes = 0;
    int n, tiles, device;
    explicit Codec(int count)
        : planes(size_t((count + 63) / 64) * 3), payload(size_t(count) * 2),
          offsets((count + 63) / 64 + 1), sizes(offsets.n), histogram(256), base(1), n(count),
          tiles((count + 63) / 64) {
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(
            cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, sizes.p, offsets.p, tiles + 1));
        CHECK_CUDA(cudaMalloc(&temp, temp_bytes));
    }
    ~Codec() {
        int current;
        cudaGetDevice(&current);
        cudaSetDevice(device);
        cudaFree(temp);
        cudaSetDevice(current);
    }
    Packed view() { return {planes.p, payload.p, offsets.p, base.p, n}; }
    void encode(const uint16_t *input) {
        CHECK_CUDA(cudaMemsetAsync(histogram.p, 0, 256 * sizeof(unsigned)));
        exponent_histogram<<<std::min((n + 255) / 256, 1024), 256>>>(input, n, histogram.p);
        choose_exponents<<<1, 256>>>(histogram.p, base.p);
        tile_sizes<<<tiles, 64>>>(input, view(), sizes.p);
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, sizes.p, offsets.p, tiles + 1));
        pack_values<<<tiles, 64>>>(input, view());
    }
    uint32_t bytes() {
        uint32_t count;
        CHECK_CUDA(cudaMemcpy(&count, offsets.p + tiles, 4, cudaMemcpyDeviceToHost));
        return count;
    }
};
void exact(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, const char *name) {
    if (a != b)
        throw std::runtime_error(std::string(name) + " bit mismatch");
}
void codec_check() {
    for (int n : {1, 63, 64, 65, 65543}) {
        DeviceBuffer<uint16_t> input(n), out(n);
        std::vector<uint16_t> host(n);
        for (int i = 0; i < n; ++i)
            host[i] = uint16_t(i);
        input.put(host);
        Codec codec(n);
        codec.encode(input.p);
        CHECK_CUDA(cudaMemset(out.p, 0xff, n * 2));
        decode_add<<<(n + 255) / 256, 256>>>(codec.view(), nullptr, out.p);
        CHECK_CUDA(cudaGetLastError());
        exact(host, out.get(), "lossless codec");
        printf("codec exact n=%d payload=%u (all BF16 bit patterns covered at n=65543)\n", n,
               codec.bytes());
    }
}
template <class F> double host_time(const char *name, F f) {
    std::vector<double> samples;
    for (int i = 0; i < 15; ++i) {
        auto begin = std::chrono::steady_clock::now();
        f();
        auto end = std::chrono::steady_clock::now();
        if (i >= 5)
            samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    double result = median(samples);
    printf("%s median_ms=%.6f samples_ms=", name, result);
    for (double ms : samples)
        printf(" %.6f", ms);
    puts("");
    return result;
}
void transfer_case(int n, bool broad, ncclComm_t *comm) {
    CHECK_CUDA(cudaSetDevice(0));
    DeviceBuffer<uint16_t> a(n);
    Codec codec(n);
    std::mt19937 rng(1337);
    std::normal_distribution<float> normal(0, 0.02);
    std::vector<uint16_t> ha(n), hb(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = broad ? uint16_t(((rng() % 224 + 16) << 7) | (rng() & 0x807f))
                      : __bfloat16_as_ushort(__float2bfloat16_rn(normal(rng)));
        hb[i] = __bfloat16_as_ushort(__float2bfloat16_rn(normal(rng)));
    }
    a.put(ha);
    codec.encode(a.p);
    uint32_t payload_bytes = codec.bytes();
    size_t packed_bytes = payload_bytes + codec.planes.n * 8 + codec.offsets.n * 4 + 4;
    CHECK_CUDA(cudaSetDevice(1));
    DeviceBuffer<uint16_t> b(n), dense(n), out(n), reference(n);
    DeviceBuffer<uint64_t> masks(codec.planes.n);
    DeviceBuffer<uint8_t> payload(codec.payload.n);
    DeviceBuffer<uint32_t> offsets(codec.offsets.n);
    DeviceBuffer<int> base(1);
    Packed received{masks.p, payload.p, offsets.p, base.p, n};
    b.put(hb);
    auto dense_path = [&] {
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMemcpyPeerAsync(dense.p, 1, a.p, 0, size_t(n) * 2));
        dense_add<<<(n + 255) / 256, 256>>>(dense.p, b.p, reference.p, n);
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    auto compressed_path = [&](bool recompress) {
        if (recompress) {
            CHECK_CUDA(cudaSetDevice(0));
            codec.encode(a.p);
            payload_bytes = codec.bytes();
        }
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMemcpyPeerAsync(masks.p, 1, codec.planes.p, 0, codec.planes.n * 8));
        CHECK_CUDA(cudaMemcpyPeerAsync(offsets.p, 1, codec.offsets.p, 0, codec.offsets.n * 4));
        CHECK_CUDA(cudaMemcpyPeerAsync(base.p, 1, codec.base.p, 0, 4));
        CHECK_CUDA(cudaMemcpyPeerAsync(payload.p, 1, codec.payload.p, 0, payload_bytes));
        decode_add<<<(n + 255) / 256, 256>>>(received, b.p, out.p);
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    auto nccl_path = [&] {
        NCCL(ncclGroupStart());
        CHECK_CUDA(cudaSetDevice(0));
        NCCL(ncclReduce(a.p, a.p, n, ncclBfloat16, ncclSum, 1, comm[0], 0));
        CHECK_CUDA(cudaSetDevice(1));
        NCCL(ncclReduce(b.p, out.p, n, ncclBfloat16, ncclSum, 1, comm[1], 0));
        NCCL(ncclGroupEnd());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    dense_path();
    auto expected = reference.get();
    compressed_path(false);
    exact(expected, out.get(), "compressed reduce");
    nccl_path();
    CHECK_CUDA(cudaSetDevice(1));
    exact(expected, out.get(), "NCCL reduce");
    printf("transport n=%d broad_exponents=%d dense_bytes=%zu wire_bytes=%zu ratio=%.6f\n", n,
           broad, size_t(n) * 2, packed_bytes, double(packed_bytes) / (size_t(n) * 2));
    host_time("dense_peer_reduce", dense_path);
    host_time("nccl_reduce", nccl_path);
    host_time("precompressed_transfer_reduce", [&] { compressed_path(false); });
    host_time("compress_transfer_reduce", [&] { compressed_path(true); });
    CHECK_CUDA(cudaSetDevice(1));
    exact(expected, out.get(), "recompressed reduce");
}

struct Message {
    cudaIpcMemHandle_t memory;
    cudaIpcEventHandle_t ready;
    int n;
};
void exchange(int fd, void *data, size_t bytes, bool send) {
    char *p = static_cast<char *>(data);
    while (bytes) {
        ssize_t n = send ? ::write(fd, p, bytes) : ::read(fd, p, bytes);
        if (n <= 0)
            throw std::runtime_error("IPC socket closed");
        p += n;
        bytes -= n;
    }
}
__global__ void check_pattern(const uint16_t *values, int n, unsigned *errors) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        if (values[i] != uint16_t(uint32_t(i) * 1664525u + 1013904223u))
            atomicAdd(errors, 1u);
}
int ipc_client(int fd) {
    CHECK_CUDA(cudaSetDevice(0));
    Message message;
    exchange(fd, &message, sizeof(message), false);
    cudaEvent_t ready;
    CHECK_CUDA(cudaIpcOpenEventHandle(&ready, message.ready));
    DeviceBuffer<unsigned> errors(1);
    DeviceBuffer<uint16_t> copied(message.n);
    uint16_t *host = nullptr;
    CHECK_CUDA(cudaMallocHost(&host, size_t(message.n) * 2));
    for (int i = 0; i < message.n; ++i)
        host[i] = uint16_t(uint32_t(i) * 1664525u + 1013904223u);
    auto verify = [&](const uint16_t *p) {
        CHECK_CUDA(cudaMemsetAsync(errors.p, 0, 4));
        check_pattern<<<std::min((message.n + 255) / 256, 4096), 256>>>(p, message.n, errors.p);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (errors.get()[0])
            throw std::runtime_error("IPC contents mismatch");
    };
    auto map = [&] {
        uint16_t *p;
        CHECK_CUDA(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&p), message.memory,
                                        cudaIpcMemLazyEnablePeerAccess));
        CHECK_CUDA(cudaStreamWaitEvent(0, ready));
        verify(p);
        CHECK_CUDA(cudaIpcCloseMemHandle(p));
    };
    printf("IPC bytes=%zu (owner allocation stays alive; client is a separate exec process)\n",
           size_t(message.n) * 2);
    auto start = std::chrono::steady_clock::now();
    map();
    printf("IPC first_map_and_read_ms=%.6f\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
               .count());
    host_time("ipc_remap_read", map);
    host_time("pinned_h2d_read", [&] {
        CHECK_CUDA(cudaMemcpyAsync(copied.p, host, size_t(message.n) * 2, cudaMemcpyHostToDevice));
        verify(copied.p);
    });
    uint16_t *persistent;
    CHECK_CUDA(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&persistent), message.memory,
                                    cudaIpcMemLazyEnablePeerAccess));
    CHECK_CUDA(cudaStreamWaitEvent(0, ready));
    host_time("ipc_reused_map_read", [&] { verify(persistent); });
    CHECK_CUDA(cudaIpcCloseMemHandle(persistent));
    CHECK_CUDA(cudaEventDestroy(ready));
    CHECK_CUDA(cudaFreeHost(host));
    int done = 1;
    exchange(fd, &done, sizeof(done), true);
    close(fd);
    return 0;
}
void ipc_case(const char *executable, int n) {
    CHECK_CUDA(cudaSetDevice(0));
    DeviceBuffer<uint16_t> data(n);
    std::vector<uint16_t> host(n);
    for (int i = 0; i < n; ++i)
        host[i] = uint16_t(uint32_t(i) * 1664525u + 1013904223u);
    auto begin = std::chrono::steady_clock::now();
    data.put(host);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("IPC owner_populate_ms=%.6f\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
               .count());
    cudaEvent_t ready;
    CHECK_CUDA(cudaEventCreateWithFlags(&ready, cudaEventInterprocess | cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(ready));
    Message message{};
    message.n = n;
    CHECK_CUDA(cudaIpcGetMemHandle(&message.memory, data.p));
    CHECK_CUDA(cudaIpcGetEventHandle(&message.ready, ready));
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets))
        throw std::runtime_error("socketpair");
    std::string descriptor = std::to_string(sockets[1]);
    char *args[] = {const_cast<char *>(executable), const_cast<char *>("--ipc-client"),
                    descriptor.data(), nullptr};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, sockets[0]);
    pid_t pid;
    fflush(stdout);
    int error = posix_spawn(&pid, executable, &actions, nullptr, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (error)
        throw std::runtime_error("posix_spawn");
    close(sockets[1]);
    exchange(sockets[0], &message, sizeof(message), true);
    int done = 0;
    exchange(sockets[0], &done, sizeof(done), false);
    close(sockets[0]);
    int status;
    waitpid(pid, &status, 0);
    if (!done || !WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("IPC client failed");
    CHECK_CUDA(cudaEventDestroy(ready));
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--ipc-client")
            return ipc_client(std::stoi(argv[2]));
        device_info();
        int nccl_version;
        NCCL(ncclGetVersion(&nccl_version));
        printf("NCCL version=%d\n", nccl_version);
        codec_check();
        if (argc > 1 && std::string(argv[1]) == "--quick") {
            puts("PASS: lossless codec stress");
            return 0;
        }
        for (int n : {49152, 100000000, 124000000})
            ipc_case(argv[0], n);
        int devices;
        CHECK_CUDA(cudaGetDeviceCount(&devices));
        if (devices < 2)
            throw std::runtime_error("multi-GPU experiment requires two GPUs");
        for (int source = 0; source < 2; ++source) {
            int peer = 1 - source, access;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&access, source, peer));
            printf("peer_access %d->%d=%d\n", source, peer, access);
            if (!access)
                throw std::runtime_error("peer access unavailable on supplied GPUs");
            CHECK_CUDA(cudaSetDevice(source));
            CHECK_CUDA(cudaDeviceEnablePeerAccess(peer, 0));
        }
        ncclComm_t comm[2];
        int devs[2] = {0, 1};
        NCCL(ncclCommInitAll(comm, 2, devs));
        for (int n : {3072 * 768, 12 * 3072 * 768})
            for (bool broad : {false, true})
                transfer_case(n, broad, comm);
        NCCL(ncclCommDestroy(comm[0]));
        NCCL(ncclCommDestroy(comm[1]));
        puts("PASS: IPC cache and ZipServ-inspired two-GPU reduce; not full training");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

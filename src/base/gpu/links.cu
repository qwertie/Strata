// src/base/gpu/links.cu - links.hpp on the CUDA runtime (HIP through the hip_compat headers).
#include "strata/base/gpu/links.hpp"

#include "strata/base/gpu/on_device.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <functional>

namespace strata::core {
namespace {

constexpr int kRounds = 4;          ///< copies timed per direction; the fastest counts
constexpr size_t kSmallBytes = 64;  ///< the copy whose round trip is the latency

/// Gets the fastest of kRounds runs of `copy` (which waits for its copy) in milliseconds; < 0 when one failed.
double time_best_ms(const std::function<bool()>& copy) {
    // timed on the host: on Windows HIP the events do not bracket the copies (model.cpp's PCIe probe)
    double best = -1;
    for (int r = 0; r < kRounds; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!copy())
            return -1;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        best = best < 0 ? ms : std::min(best, ms);
    }
    return best;
}

/// Gets GB/s of `bytes` copied in `ms` (0 when not measured).
double get_gbps(size_t bytes, double ms) { return ms > 0 ? (double) bytes / (ms * 1e-3) / 1e9 : 0; }

/// Measures copies between `a` and `b` (device or page-locked memory of `bytes`), which are on devices `dev_a` and
/// `dev_b` (-1: the host): `to` is a -> b.
LinkSpeed measure(void* a, void* b, size_t bytes, int dev_a, int dev_b) {
    // a copy between two devices may return before it is done: the cards are waited for
    const auto copy = [&](void* dst, const void* src, size_t n) {
        if (cudaMemcpy(dst, src, n, cudaMemcpyDefault) != cudaSuccess)
            return false;
        for (const int dev : {dev_a, dev_b}) {
            if (dev < 0)
                continue;
            const OnDevice on(dev);
            if (cudaDeviceSynchronize() != cudaSuccess)
                return false;
        }
        return true;
    };
    LinkSpeed s;
    if (!copy(b, a, bytes))   // warm up: the context and the copy engines
        return s;
    s.to_gbps = get_gbps(bytes, time_best_ms([&] { return copy(b, a, bytes); }));
    s.from_gbps = get_gbps(bytes, time_best_ms([&] { return copy(a, b, bytes); }));
    const double rt = time_best_ms([&] { return copy(b, a, kSmallBytes) && copy(a, b, kSmallBytes); });
    s.latency_ms = std::max(rt, 0.0);
    if (!s.is_known())
        s = LinkSpeed{};
    return s;
}

}  // namespace

LinkSpeed measure_host_link(int dev, size_t bytes) {
    const OnDevice on(dev);
    void* h = nullptr;
    void* d = nullptr;
    LinkSpeed s;
    if (cudaMallocHost(&h, bytes) == cudaSuccess && cudaMalloc(&d, bytes) == cudaSuccess)
        s = measure(h, d, bytes, -1, dev);
    (void) cudaGetLastError();
    (void) cudaFree(d);
    (void) cudaFreeHost(h);
    return s;
}

LinkSpeed measure_card_link(int from, int to, size_t bytes) {
    void* a = nullptr;
    void* b = nullptr;
    LinkSpeed s;
    {
        const OnDevice on(from);
        (void) cudaMalloc(&a, bytes);
    }
    {
        const OnDevice on(to);
        (void) cudaMalloc(&b, bytes);
    }
    if (a != nullptr && b != nullptr)
        s = measure(a, b, bytes, from, to);
    (void) cudaGetLastError();
    for (const auto& [dev, p] : {std::make_pair(from, a), std::make_pair(to, b)}) {
        const OnDevice on(dev);
        (void) cudaFree(p);
    }
    return s;
}

double get_card_speed(int dev) {
    int sms = 0, khz = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess || khz <= 0)
        khz = 1800000;   // (the old engine's guess)
    (void) cudaGetLastError();
    return std::max(1.0, sms * (khz / 1e6));
}

}  // namespace strata::core

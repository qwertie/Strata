#pragma once

#include <cstdint>

namespace strata::kernels {

/// Bounds the token rows accepted by the grouped expert kernels and speculative verifier.
inline constexpr int kVerifyMaxT = 8;

/// copy_from_mapped, or zeros when *skip == value (n a multiple of 4, 16-byte aligned).
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream);
/// Records the device clock in nanoseconds for a captured window's profiler.
void gpu_stamp(unsigned long long* buf, int i, void* stream);

}  // namespace strata::kernels

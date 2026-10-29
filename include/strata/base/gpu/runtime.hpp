// include/strata/base/gpu/runtime.hpp - the device runtime as shared code sees it: memory and copies, without the
// CUDA runtime's own names, so that code above this layer builds for every backend.  Streams are passed as `void*`
// (a cudaStream_t; null: the legacy default stream).  Each call works on the calling thread's current device.
#pragma once

#include <cstddef>

namespace strata::core {

/// The memories a copy reads and writes.
enum class GpuCopy { host_to_device, device_to_host, device_to_device };

/// Allocates `bytes` of device memory and zeroes it; nullptr when the device has not that much.
void* gpu_alloc_zeroed(size_t bytes);
void gpu_free(void* p);
/// Queues a copy of `height` rows of `width` bytes, whose starts are `src_pitch` and `dst_pitch` bytes apart.
/// Returns false when the runtime refuses it.
bool gpu_copy_2d_async(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height,
                       GpuCopy kind, void* stream);
bool gpu_memset_async(void* p, int value, size_t bytes, void* stream);

}  // namespace strata::core

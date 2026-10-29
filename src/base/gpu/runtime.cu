// src/base/gpu/runtime.cu - runtime.hpp on the CUDA runtime (HIP through the hip_compat headers).
#include "strata/base/gpu/runtime.hpp"

#include <cuda_runtime.h>

namespace strata::core {

void* gpu_alloc_zeroed(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    if (cudaMemset(p, 0, bytes) == cudaSuccess) return p;
    (void) cudaFree(p);
    return nullptr;
}

void gpu_free(void* p) { (void) cudaFree(p); }

bool gpu_copy_2d_async(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height,
                       GpuCopy kind, void* stream) {
    return cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height,
                             kind == GpuCopy::host_to_device   ? cudaMemcpyHostToDevice
                             : kind == GpuCopy::device_to_host ? cudaMemcpyDeviceToHost
                                                               : cudaMemcpyDeviceToDevice,
                             (cudaStream_t) stream) == cudaSuccess;
}

bool gpu_memset_async(void* p, int value, size_t bytes, void* stream) {
    return cudaMemsetAsync(p, value, bytes, (cudaStream_t) stream) == cudaSuccess;
}

}  // namespace strata::core

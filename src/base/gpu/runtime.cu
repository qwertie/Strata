// src/base/gpu/runtime.cu - runtime.hpp on the CUDA runtime (HIP through the hip_compat headers).
#include "strata/base/gpu/runtime.hpp"

#include <cuda_runtime.h>

namespace strata::core {

void* gpu_alloc_zeroed(size_t bytes) {
    void* p = gpu_alloc(bytes);
    if (p == nullptr)
        return nullptr;
    if (cudaMemset(p, 0, bytes) == cudaSuccess) return p;
    (void) cudaFree(p);
    return nullptr;
}

void* gpu_alloc(size_t bytes) {
    void* p = nullptr;
    return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}

void gpu_free(void* p) { (void) cudaFree(p); }

bool gpu_copy_2d_async(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height,
                       void* stream) {
    // one row as a plain copy: HIP's 2D copy of pageable host memory is not ordered like the 1D one
    const cudaError_t e = height == 1
        ? cudaMemcpyAsync(dst, src, width, cudaMemcpyDefault, (cudaStream_t) stream)
        : cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDefault, (cudaStream_t) stream);
    return e == cudaSuccess;
}

bool gpu_memset_async(void* p, int value, size_t bytes, void* stream) {
    return cudaMemsetAsync(p, value, bytes, (cudaStream_t) stream) == cudaSuccess;
}

bool gpu_stream_synchronize(void* stream) {
    return cudaStreamSynchronize((cudaStream_t) stream) == cudaSuccess;
}

bool gpu_copy(void* dst, const void* src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) == cudaSuccess;
}

void* gpu_host_alloc_mapped(size_t bytes, void** device_view) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    if (cudaHostGetDevicePointer(device_view, p, 0) == cudaSuccess)
        return p;
    (void) cudaFreeHost(p);
    return nullptr;
}

void gpu_host_free(void* p) { (void) cudaFreeHost(p); }

}  // namespace strata::core

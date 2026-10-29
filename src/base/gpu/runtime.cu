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

}  // namespace strata::core

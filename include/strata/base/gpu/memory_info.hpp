#pragma once
// A GPU's memory as the runtime reports it now: its name, size and free bytes, and whether it is integrated (its
// memory is the system RAM, as on Strix Halo).  Used by the engines' answer to GET_MEMORY_USE
// (strata/common/serve/memory_report.hpp).

#include "strata/base/gpu/on_device.hpp"

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

namespace strata::core {

/// Reads GPU `device`'s name, total and free bytes and whether it is integrated; what the runtime refuses is left as
/// it was.
inline void read_gpu_memory(int device, std::string& name, uint64_t& total, uint64_t& free, bool& integrated) {
    const OnDevice on(device);
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
        total = t;
        free = f;
    } else {
        (void) cudaGetLastError();
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) == cudaSuccess)
        name = p.name;
    else
        (void) cudaGetLastError();
#if !defined(STRATA_HIP_GFX906)   // (the gfx906 compat layer has no cudaDevAttrIntegrated; that GPU is discrete)
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, device) == cudaSuccess)
        integrated = v != 0;
    else
        (void) cudaGetLastError();
#endif
}

}  // namespace strata::core

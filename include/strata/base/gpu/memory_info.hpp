#pragma once
// A GPU's memory as the runtime reports it now: its name, size and free bytes, and whether it is integrated (its
// memory is the system RAM, as on Strix Halo).  Used by the engines' answer to GET_MEMORY_USE
// (strata/common/serve/memory_report.hpp).

#include "strata/base/gpu/on_device.hpp"
#include "strata/base/platform/memory.hpp"

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

/// Gets how many bytes of integrated GPU `device`'s dedicated share (the part of the system RAM the BIOS gives it)
/// this process may still take: the share less what the process holds on the GPU.  Past the share, the GPU's
/// allocations borrow system RAM.  `mapped`: the page-locked host memory the process holds that the GPU maps, which
/// Windows counts in the GPU's local use as well (on the Strix Halo, 4 GiB of hipHostMalloc raised it by 4 GiB) but
/// which is system RAM, not the share.  False for a discrete GPU, and where the share is not known (elsewhere than
/// Windows).
inline bool get_share_room(int device, uint64_t mapped, uint64_t& room) {
    std::string name, why;
    uint64_t total = 0, free = 0, share = 0, usage = 0;
    bool integrated = false;
    read_gpu_memory(device, name, total, free, integrated);
    cudaDeviceProp p{};
    if (!integrated || cudaGetDeviceProperties(&p, device) != cudaSuccess ||
        !platform::gpu_dedicated_memory(p.luid, share, usage, why)) {
        (void) cudaGetLastError();
        return false;
    }
    const uint64_t on_gpu = usage > mapped ? usage - mapped : 0;
    room = share > on_gpu ? share - on_gpu : 0;
    return true;
}

}  // namespace strata::core

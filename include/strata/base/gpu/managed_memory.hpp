#pragma once

#include "strata/base/platform/memory_manager.hpp"

#include <cuda_runtime.h>

namespace strata::core {

/// Allocates device storage while retaining the runtime's error and the allocation's labels.
template<class T>
cudaError_t allocate_device(T** address, size_t bytes, MemoryLabel name = {}) {
    const cudaError_t result = cudaMalloc(reinterpret_cast<void**>(address), bytes);
    if (result == cudaSuccess) {
        int device = 0;
        (void) cudaGetDevice(&device);
        get_memory_manager().add(*address, bytes, "gpu" + std::to_string(device), std::move(name));
    }
    return result;
}

/// Releases device storage and its labels together, retaining labels if the runtime refuses the release.
inline cudaError_t release_device(void* address) {
    const uint64_t id = get_memory_manager().get_allocation_id(address);
    const cudaError_t result = cudaFree(address);
    if (result == cudaSuccess)
        get_memory_manager().remove_if_current(address, id);
    return result;
}

/// Allocates page-locked RAM; a device mapping is an alias, not another allocation.
template<class T>
cudaError_t allocate_host(T** address, size_t bytes, unsigned flags, MemoryLabel name = {}) {
    const cudaError_t result = cudaHostAlloc(reinterpret_cast<void**>(address), bytes, flags);
    if (result == cudaSuccess) {
        if (name.pool == "Device allocations")
            name.pool = "Page-locked RAM";
        get_memory_manager().add(*address, bytes, "ram", std::move(name), true);
    }
    return result;
}

/// Releases page-locked RAM and its labels together.
inline cudaError_t release_host(void* address) {
    const uint64_t id = get_memory_manager().get_allocation_id(address);
    const cudaError_t result = cudaFreeHost(address);
    if (result == cudaSuccess)
        get_memory_manager().remove_if_current(address, id);
    return result;
}

}  // namespace strata::core

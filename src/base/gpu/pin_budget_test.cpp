// pin_budget_test: the pinned-memory headroom (include/strata/base/gpu/pin_budget.hpp) with the simulated ceiling.
// Needs a GPU for the pinned calls; without one it prints SKIP and passes.
#include "strata/base/platform/env.hpp"
#include "strata/base/gpu/pin_budget.hpp"
#include "strata/base/gpu/managed_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/// Checks the managed runtime on small real allocations without loading a model.
int test_managed_allocations();

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--gpu") == 0)
        return test_managed_allocations();
    // the singleton reads its environment on first use: set everything first
    strata::platform::set_env("STRATA_TEST_PIN_BUDGET_MIB", "100");
    strata::platform::set_env("STRATA_PIN_RESERVE_MIB", "32");   // soft 32 MiB, hard 64 MiB (a quarter of 32 is below the 64 MiB floor)
    using namespace strata::core;
    if (cudaFree(0) != cudaSuccess) {
        std::printf("pin_budget_test: SKIP (no GPU)\n");
        return 0;
    }
    (void) cudaGetLastError();

    // 1. a registration beyond the ceiling is refused, one inside it works and is charged
    {
        void* big = std::malloc(160u << 20);
        void* small = std::malloc(8u << 20);
        CHECK(pin_register(big, 160u << 20, cudaHostRegisterPortable) == cudaErrorMemoryAllocation);
        CHECK(pin_register(small, 8u << 20, cudaHostRegisterPortable) == cudaSuccess);
        CHECK(pin_detail::st().sim_used == (8u << 20));
        CHECK(pin_unregister(small) == cudaSuccess);
        CHECK(pin_detail::st().sim_used == 0);
        std::free(big);
        std::free(small);
    }
    // 2. without a reserve a refused guarded allocation stays refused
    {
        void* p = nullptr;
        CHECK(host_alloc_guarded(&p, 200u << 20, cudaHostAllocMapped) != cudaSuccess);
        (void) cudaGetLastError();
    }
    // 3. the reserves are held, the arena (here: a registration) takes what is left, a required buffer is then
    //    refused, the held headroom is handed back and the retry works
    {
        pin_reserve_take();
        CHECK(pin_detail::st().hard != nullptr && pin_detail::st().hard_bytes == (64u << 20));
        CHECK(pin_detail::st().soft != nullptr && pin_detail::st().soft_bytes == (32u << 20));
        CHECK(pin_detail::st().sim_used == (96u << 20));
        void* arena = std::malloc(4u << 20);
        CHECK(pin_register(arena, 4u << 20, cudaHostRegisterPortable) == cudaSuccess);   // 100 MiB used: the ceiling
        pin_reserve_release_soft();
        CHECK(pin_detail::st().soft == nullptr);
        CHECK(pin_detail::st().sim_used == (68u << 20));
        void* a = nullptr;
        CHECK(host_alloc_guarded(&a, 16u << 20, cudaHostAllocMapped) == cudaSuccess);   // fits beside the hard reserve
        void* b = nullptr;
        CHECK(host_alloc_guarded(&b, 40u << 20, cudaHostAllocMapped) == cudaSuccess);   // refused first, then the hard reserve goes
        CHECK(pin_detail::st().hard == nullptr);
        CHECK(pin_detail::st().hard_retries == 1);
        void* c = nullptr;
        CHECK(host_alloc_guarded(&c, 60u << 20, cudaHostAllocMapped) != cudaSuccess);   // truly out: reported, not hidden
        (void) cudaGetLastError();
        CHECK(pin_exhausted_hint().find("[pinned-memory-exhausted]") != std::string::npos);
        (void) pin_unregister(arena);
        std::free(arena);
    }
    std::printf(g_fail == 0 ? "pin_budget_test: OK\n" : "pin_budget_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

int test_managed_allocations() {
    using namespace strata::core;
    constexpr size_t bytes = 1u << 20;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 1;
    const std::string memory = "gpu" + std::to_string(device);
    const uint64_t mapped = get_memory_manager().get_mapped_bytes();
    void* gpu = nullptr;
    void* host = nullptr;
    if (allocate_device(&gpu, bytes, {"allocation_test", "Test pool", "Device block"}) != cudaSuccess)
        return 1;
    if (allocate_host(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable,
                      {"allocation_test", "Test pool", "Host block"}) != cudaSuccess) {
        (void) release_device(gpu);
        return 1;
    }
    bool found_device = false, found_host = false;
    for (const MemoryBlock& block : get_memory_manager().get_blocks(memory))
        found_device = found_device || (block.name.category == "allocation_test" && block.bytes == bytes &&
                                       block.name.label == "Device block");
    for (const MemoryBlock& block : get_memory_manager().get_blocks("ram"))
        found_host = found_host || (block.name.category == "allocation_test" && block.bytes == bytes &&
                                   block.name.label == "Host block");
    const bool counted = found_device && found_host && get_memory_manager().get_mapped_bytes() == mapped + bytes;
    const bool released_host = release_host(host) == cudaSuccess;
    const bool released_device = release_device(gpu) == cudaSuccess;
    bool retained = false;
    for (const std::string& place : {memory, std::string("ram")})
        for (const MemoryBlock& block : get_memory_manager().get_blocks(place))
            retained = retained || block.name.category == "allocation_test";
    const bool ok = counted && released_host && released_device && !retained &&
                    get_memory_manager().get_mapped_bytes() == mapped;
    std::printf("managed allocations: labels, aliases and releases %s\n", ok ? "passed" : "FAILED");
    return ok ? 0 : 1;
}

// src/common/serve/paged_start.cpp - what every model's start does before the paged engine runs it (paged_engine.hpp,
// "a model's start"): the engine's own options, the runtime's environment, the cards' links, the pools' sizes.
#include "strata/common/serve/paged_engine.hpp"

#include "strata/base/gpu/links.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/env.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/split_rules.hpp"
#include "strata/common/serve/request.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace strata::serve {

/// The bytes each way that measure a GPU's link to RAM at start.
constexpr size_t kLinkProbeBytes = 64ull << 20;

int take_paged_options(int& argc, char** argv, const std::string& program, PagedSettings& s) {
    struct Own {
        const char* name;
        int64_t* value;
        std::string* text = nullptr;   ///< instead of `value`, for a word
    };
    int64_t rows = s.step.rows, lookahead = -1, concurrent = -1;   // -1: chosen once the options are read
    std::string drafter = "on", gpu_mib;
    const Own own[] = {{"--tokens-min", &s.tokens_min},       {"--tokens-max", &s.tokens_max},
                       {"--gpu-mib", nullptr, &gpu_mib},      {"--ram-mib", &s.ram_mib},
                       {"--ssd-pause-mib", &s.ssd_pause_mib}, {"--rows", &rows},
                       {"--lookahead", &lookahead},           {"--concurrent", &concurrent},
                       {"--pause-dir", nullptr, &s.pause_dir}, {"--copy-back", nullptr, &s.copy_back},
                       {"--drafter", nullptr, &drafter},      {"--remote-experts", nullptr, &s.remote_experts}};
    const char* const name = program.c_str();
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const Own* o = std::find_if(std::begin(own), std::end(own),
                                    [&](const Own& x) { return std::strcmp(argv[i], x.name) == 0; });
        if (o == std::end(own)) {
            argv[out++] = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s: %s needs a value\n", name, o->name);
            return 2;
        }
        if (o->text != nullptr)
            *o->text = argv[++i];
        else
            *o->value = std::atoll(argv[++i]);
    }
    argc = out;
    s.step.rows = (int) std::clamp<int64_t>(rows, 0, INT32_MAX);
    s.step.lookahead = (int) std::clamp<int64_t>(lookahead, -1, INT32_MAX);
    s.step.concurrent = (int) std::clamp<int64_t>(concurrent, -1, INT32_MAX);   // -1: --batch, set once it is read
    s.drafter = drafter == "on";
    if (s.copy_back != "auto" && s.copy_back != "on" && s.copy_back != "off") {
        std::fprintf(stderr, "%s: --copy-back takes auto, on or off\n", name);
        return 2;
    }
    if (s.remote_experts != "auto" && s.remote_experts != "on" && s.remote_experts != "off") {
        std::fprintf(stderr, "%s: --remote-experts takes auto, on or off\n", name);
        return 2;
    }
    if (drafter != "on" && drafter != "off") {
        std::fprintf(stderr, "%s: --drafter takes on or off\n", name);
        return 2;
    }
    if (std::string err; !gpu_mib.empty() && !parse_i64_list(gpu_mib.c_str(), s.gpu_mib, err)) {
        std::fprintf(stderr, "%s: --gpu-mib takes MiB, one per card of a layer split: %s\n", name, err.c_str());
        return 2;
    }
    return -1;
}

void set_paged_runtime_env(const PagedSettings& s, bool layer_split) {
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    // ROCclr on Windows keeps freed memory in its resource cache and gives none of it back (on the Strix Halo, 8 GiB
    // freed with hipHostFree or hipFree stayed taken from Windows' RAM), so the pools could not shrink
    strata::platform::set_env("GPU_RESOURCE_CACHE_SIZE", "0", false);
#endif
    // a card computing another card's experts (--remote-experts) runs them on a stream of its own while that card's
    // verify window spins on the GPU for them; the runtime spreads its streams over a few hardware queues (HIP 4 a
    // priority, CUDA 8 connections), and a hop that landed behind a spinning kernel waited for ever (two stages on the
    // Strix Halo's GPU, 2 of 2 runs; 0 of 1 with 16 queues, which a stream of the highest priority did not fix)
    if (s.remote_experts != "off" && layer_split) {
        strata::platform::set_env("GPU_MAX_HW_QUEUES", "16", false);
        strata::platform::set_env("CUDA_DEVICE_MAX_CONNECTIONS", "32", false);
    }
}

void measure_card(PagedCard& card, const std::string& program, uint64_t expert_bytes) {
    const strata::core::LinkSpeed link = card.link = strata::core::measure_host_link(card.dev, kLinkProbeBytes);
    card.refill_batch = strata::core::get_refill_batch(link, expert_bytes, kRefillBudgetMs, kRefillBatch);
    std::fprintf(stderr, "%s: %s's link to RAM: %.1f GB/s to the GPU, %.1f GB/s back, %.3f ms a round trip\n",
                 program.c_str(), card.id.c_str(), link.to_gbps, link.from_gbps, link.latency_ms);
    card.layer_ms = strata::program::split_rules::get_layer_ms(strata::core::get_card_speed(card.dev));
}

strata::core::ChunkMemory get_gpu_chunk_memory(int dev) {
    strata::core::ChunkMemory memory;
    memory.alloc = [dev](uint64_t n) {
        const strata::core::OnDevice on(dev);
        return (uint8_t*) strata::core::gpu_alloc_zeroed((size_t) n);
    };
    memory.free = [dev](uint8_t* p) {
        const strata::core::OnDevice on(dev);
        strata::core::gpu_free(p);
    };
    memory.copy = [dev](uint8_t* to, const uint8_t* from, uint64_t n) {
        const strata::core::OnDevice on(dev);
        return strata::core::gpu_copy(to, from, (size_t) n);
    };
    return memory;
}

uint64_t get_gpu_pool_bytes(int dev, uint64_t keep, int64_t gpu_mib, uint64_t most, uint64_t& room) {
    room = get_gpu_room(dev, keep, 0);
    return gpu_mib >= 0 ? std::min((uint64_t) gpu_mib << 20, room) : std::min(room, most);
}

uint64_t get_conversation_bytes(const strata::core::StateLayout& layout, int64_t tokens, uint64_t page_bytes) {
    uint64_t bytes = layout.bytes_for(tokens);
    for (const strata::core::StateRegion& r : layout.regions)
        bytes += strata::core::choose_block_pages(r, page_bytes) * page_bytes;
    return bytes;
}

uint64_t get_chunk_bytes(uint64_t pool_bytes, uint64_t page_bytes) {
    return std::max<uint64_t>(std::min(kChunkBytes, pool_bytes) / page_bytes, 1) * page_bytes;
}

void say_gpu_pool(const std::string& program, const PagedCard& card, uint64_t room, uint64_t chunk_bytes) {
    const strata::core::PagePool& pool = card.experts.pool();
    std::fprintf(stderr, "%s: the page pool of %s (layers %lld-%lld): %.2f GiB (of %.2f GiB it could take) in chunks "
                         "of %.0f MiB, pages of %llu bytes, for the experts and the conversations, %lld tokens of it "
                         "kept for the conversations\n",
                 program.c_str(), card.id.c_str(), (long long) card.lb, (long long) card.le - 1,
                 (double) pool.report().total.bytes / 1073741824.0, (double) room / 1073741824.0,
                 (double) chunk_bytes / 1048576.0, (unsigned long long) pool.page_bytes(),
                 (long long) pool.tokens_min());
}

uint64_t get_ram_pool_bytes(const PagedSettings& s, uint64_t headroom, uint64_t most) {
    return s.ram_mib < 0 ? std::min(get_ram_room(headroom), most) : (uint64_t) s.ram_mib << 20;
}

strata::core::PagePoolConfig get_ram_pool_config(const PagedCard& card) {
    strata::core::PagePoolConfig config;
    config.page_bytes = card.experts.pool().page_bytes();
    config.chunk_bytes = kChunkBytes / config.page_bytes * config.page_bytes;
    return config;
}

}  // namespace strata::serve

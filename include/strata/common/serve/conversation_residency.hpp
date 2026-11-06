#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::serve {

/// Chooses a cached-place allowance from decode concurrency and the smallest card; zero VRAM means unknown.
inline int get_default_resident_conversations(int concurrent, int capacity, uint64_t smallest_vram,
                                               uint64_t extra_threshold) {
    const int wanted = smallest_vram && smallest_vram >= extra_threshold ? std::max(concurrent, 2) : concurrent;
    return std::min(capacity, wanted + 1);
}

/// Protects active work while making sufficiently old idle state eligible for pressure-driven reclamation.
inline bool is_conversation_stale(bool decoding, bool reading, uint64_t modified, uint64_t now, uint64_t age) {
    return !decoding && !reading && (modified == 0 || (now >= modified && now - modified >= age));
}

}  // namespace strata::serve

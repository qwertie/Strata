// include/strata/common/state/state_layout.hpp - what memory one conversation keeps, as a list of regions a model
// describes once, and the bookkeeping every engine needs on it: its size at N tokens, a session's buffers, and
// copies of the first N tokens' state between two sessions or between a session and host memory.
// src/common/state/README.md says how each engine maps onto it.
#pragma once

#include "strata/base/gpu/runtime.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

/// How a region grows with the conversation.
enum class RegionKind {
    fixed,       ///< the same at any length: a recurrent (GDN/SSM) state, a conv history, the last hidden row
    per_token,   ///< K or V cells
    per_block,   ///< one entry per completed block of tokens: compressed K/V, index keys
};

/// One buffer of a conversation's memory.  A growing region is `rows` rows (head-major K/V: one per KV head), and in
/// a session each row has room for the session's whole context.
struct StateRegion {
    RegionKind kind = RegionKind::fixed;
    uint64_t unit_bytes = 0;   ///< the bytes of one row's token or block; of the whole region when fixed
    uint64_t rows = 1;
    int64_t block = 1;         ///< tokens per block (per_block)

    /// Gets how many tokens or blocks a row holds for `tokens` tokens (1 for a fixed region).
    uint64_t units_for(int64_t tokens) const {
        if (kind == RegionKind::fixed)
            return 1;
        return (uint64_t) tokens / (uint64_t) (kind == RegionKind::per_block ? block : 1);
    }
    /// Gets the bytes of one row at `tokens` tokens.
    uint64_t row_bytes(int64_t tokens) const { return units_for(tokens) * unit_bytes; }
    uint64_t bytes_for(int64_t tokens) const { return rows * row_bytes(tokens); }
};

/// The regions of one conversation on one device, in the order the model gave them.  A model split across devices
/// describes one layout per device.
struct StateLayout {
    std::vector<StateRegion> regions;

    /// Appends a region and returns its index.
    size_t add(RegionKind kind, uint64_t unit_bytes, uint64_t rows = 1, int64_t block = 1) {
        regions.push_back({kind, unit_bytes, rows, block});
        return regions.size() - 1;
    }
    /// Gets the bytes the state of `tokens` tokens takes (a session's allocation at its context, a packed image).
    uint64_t bytes_for(int64_t tokens) const {
        uint64_t n = 0;
        for (const StateRegion& r : regions)
            n += r.bytes_for(tokens);
        return n;
    }
};

/// Where one conversation's regions are: one address per region, whose rows each hold `cells` tokens.
struct StateBuffers {
    std::vector<uint8_t*> at;
    int64_t cells = 0;
    bool on_host = false;   ///< host memory (a packed image), else the device's
};

/// Gets the packed host image of `tokens` tokens at `base`: each region's bytes for them, one after the other
/// (layout.bytes_for(tokens) in all).  For parking a conversation in RAM or writing it to a file.
inline StateBuffers get_host_image(const StateLayout& layout, int64_t tokens, uint8_t* base) {
    StateBuffers image{{}, tokens, true};
    for (const StateRegion& r : layout.regions) {
        image.at.push_back(base);
        base += r.bytes_for(tokens);
    }
    return image;
}

/// Calls `copy(dst, dst_pitch, src, src_pitch, width, height)` for each region that holds bytes at `tokens` tokens:
/// the rectangle that moves the first `tokens` tokens' state of that region from `from` to `to`.  Stops at the
/// first call that returns false and returns false.
template <class CopyFn>
bool for_each_region_copy(const StateLayout& layout, const StateBuffers& from, const StateBuffers& to, int64_t tokens,
                          CopyFn&& copy) {
    for (size_t i = 0; i < layout.regions.size(); ++i) {
        const StateRegion& r = layout.regions[i];
        const uint64_t width = r.row_bytes(tokens);
        if (width == 0)
            continue;
        if (!copy(to.at[i], r.row_bytes(to.cells), from.at[i], r.row_bytes(from.cells), width, r.rows))
            return false;
    }
    return true;
}

/// Allocates one zeroed device buffer per region, with room for `cells` tokens.  Returns false, holding nothing,
/// when the device has not that much.
inline bool alloc_state(const StateLayout& layout, int64_t cells, StateBuffers& out) {
    StateBuffers b{{}, cells, false};
    for (const StateRegion& r : layout.regions) {
        b.at.push_back((uint8_t*) gpu_alloc_zeroed(r.bytes_for(cells)));
        if (b.at.back() != nullptr)
            continue;
        for (uint8_t* p : b.at)
            gpu_free(p);
        return false;
    }
    out = std::move(b);
    return true;
}

/// Queues on `stream` the copy of the first `tokens` tokens' state from one place to another: two sessions on the
/// device, or a session and a host image.
inline bool copy_state(const StateLayout& layout, const StateBuffers& from, const StateBuffers& to, int64_t tokens,
                       void* stream) {
    const GpuCopy kind = from.on_host ? GpuCopy::host_to_device
                         : to.on_host ? GpuCopy::device_to_host
                                      : GpuCopy::device_to_device;
    return for_each_region_copy(layout, from, to, tokens, [&](void* d, size_t dp, const void* s, size_t sp, size_t w,
                                                               size_t h) {
        return gpu_copy_2d_async(d, dp, s, sp, w, h, kind, stream);
    });
}

/// Queues on `stream` the zeroing of a session's fixed regions: the start of a new conversation (its growing
/// regions need none, as nothing reads cells past the position).
inline bool clear_fixed_state(const StateLayout& layout, const StateBuffers& state, void* stream) {
    for (size_t i = 0; i < layout.regions.size(); ++i)
        if (layout.regions[i].kind == RegionKind::fixed &&
            !gpu_memset_async(state.at[i], 0, layout.regions[i].bytes_for(0), stream))
            return false;
    return true;
}

}  // namespace strata::core

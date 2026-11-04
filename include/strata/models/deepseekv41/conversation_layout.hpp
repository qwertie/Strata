// include/strata/models/deepseekv41/conversation_layout.hpp - one DeepSeek V4.1 conversation's state as regions of
// the page pool (state_layout.hpp, page_pool.hpp; docs/MEMORY_MANAGER_PLAN.md, step D1): each layer's window ring
// and each compressor's partial group are fixed; each KV source's compressed KV and index keys grow by a row per
// group of `ratio` tokens the source completes.
#pragma once

#include "strata/common/state/state_layout.hpp"
#include "strata/common/state/pool_copy.hpp"
#include "strata/models/deepseekv41/geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ds4 {

/// Stands for a region a card does not keep.
constexpr size_t kNoRegion = SIZE_MAX;
/// Stands for a part of a fixed region a card does not keep.
constexpr uint64_t kNoOffset = UINT64_MAX;

/// A layer whose compressor writes compressed KV and index keys, which it and the later layers of its ratio read.
struct KvSource {
    int layer = 0;
    int ratio = 1;   ///< the tokens one compressed row stands for
};

/// Gets the KV sources in layer order and, per trunk layer, the index of the source it reads (-1 for a layer of
/// ratio 0, which attends to its window only): the last source at or before it.  False (and `err`) when a layer of
/// ratio r > 0 has no source before it, or that source's ratio is not r.
bool find_kv_sources(const Geometry& g, std::vector<KvSource>& sources, std::vector<int>& source_of,
                     std::string& err);

/// Where describe_conversation put each part of a conversation's state.
struct ConversationRegions {
    uint64_t taps = kNoOffset; ///< Recent trunk inputs retained while DSpark weights are unloaded.
    size_t trunk = kNoRegion;     ///< fixed: the card's trunk layers' rings, then its KV sources' partial groups
    size_t drafter = kNoRegion;   ///< fixed: the DSpark blocks' rings
    /// Per layer (the trunk's, then the DSpark blocks'): where its ring starts in its fixed region.
    std::vector<uint64_t> ring;
    /// Per KV source: where its compressor's partial group starts in `trunk`, `ratio` x head_dim floats of KV and
    /// then as many of scores; kNoOffset for a source of ratio 1 (every token completes its group).
    std::vector<uint64_t> group;
    /// Per KV source: the growing region of its compressed rows, and of its index keys.
    std::vector<size_t> ckv, ik;
};

/// Describes what one conversation keeps on a card that runs trunk layers [lb, le), and with `drafter` the DSpark
/// blocks: a fixed region of its layers' rings (the window's last `win` tokens, kKvFp8 rows) and partial groups,
/// one of the DSpark blocks' rings and recent trunk taps, and per KV source a layer of the card reads, its
/// compressed rows (kKvFp4) and,
/// when a layer of the card scores them, its index keys (kKvFp4p), as per_block regions of one row per `ratio`
/// tokens.  A source's rows are on every card that reads them.  False (and `err`) when the geometry or the range is
/// not one the old engine runs.
bool describe_conversation(const Geometry& g, int lb, int le, bool drafter, strata::core::StateLayout& layout,
                           ConversationRegions& at, std::string& err);

/// Selects a stage's state in its own layout or the whole model's layout (`whole`), in matching piece order.
strata::core::RegionSpan get_conversation_span(const Geometry& g, int lb, int le, bool drafter, bool whole);

}  // namespace strata::ds4

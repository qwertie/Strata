// src/models/deepseekv41/conversation_layout.cpp - one DeepSeek V4.1 conversation's state as page-pool regions
// (conversation_layout.hpp).
#include "strata/models/deepseekv41/conversation_layout.hpp"

#include "strata/models/deepseekv41/kv_rows.hpp"

#include <algorithm>

namespace strata::ds4 {

using strata::core::RegionKind;
using strata::core::StateLayout;

bool find_kv_sources(const Geometry& g, std::vector<KvSource>& sources, std::vector<int>& source_of,
                     std::string& err) {
    sources.clear();
    source_of.assign((size_t) std::max(g.nl, 0), -1);
    if (g.nl <= 0 || g.ratios.size() < (size_t) g.nl) {
        err = "the geometry has no compress ratio for each of its " + std::to_string(g.nl) + " layers";
        return false;
    }
    for (int il = 0; il < g.nl; ++il) {
        const int ratio = g.ratios[(size_t) il];
        if (std::find(g.kv_src.begin(), g.kv_src.end(), il) != g.kv_src.end())
            sources.push_back({il, ratio});
        if (ratio == 0)
            continue;
        if (sources.empty() || sources.back().ratio != ratio) {
            err = "layer " + std::to_string(il) + ": compress ratio " + std::to_string(ratio) +
                  " without a KV source of that ratio before it";
            return false;
        }
        source_of[(size_t) il] = (int) sources.size() - 1;
    }
    return true;
}

bool describe_conversation(const Geometry& g, int lb, int le, bool drafter, StateLayout& layout,
                           ConversationRegions& at, std::string& err) {
    std::vector<KvSource> sources;
    std::vector<int> source_of;
    if (!find_kv_sources(g, sources, source_of, err))
        return false;
    if (lb < 0 || le < lb || le > g.nl || g.ratios.size() < (size_t) (g.nl + g.n_mtp)) {
        err = "layers [" + std::to_string(lb) + ", " + std::to_string(le) + ") are not a range of the model's " +
              std::to_string(g.nl) + " layers, or its " + std::to_string(g.n_mtp) + " DSpark blocks have no ratios";
        return false;
    }
    const uint64_t ring = (uint64_t) g.win * kv_row_bytes(kKvFp8, g.hd);
    layout = {};
    at = {};
    at.ring.assign((size_t) (g.nl + g.n_mtp), kNoOffset);
    at.group.assign(sources.size(), kNoOffset);
    at.ckv.assign(sources.size(), kNoRegion);
    at.ik.assign(sources.size(), kNoRegion);

    uint64_t trunk = 0;
    for (int il = lb; il < le; ++il) {
        at.ring[(size_t) il] = trunk;
        trunk += ring;
    }
    for (size_t s = 0; s < sources.size(); ++s)
        if (sources[s].layer >= lb && sources[s].layer < le && sources[s].ratio > 1) {
            at.group[s] = trunk;
            trunk += 2 * (uint64_t) sources[s].ratio * (uint64_t) g.hd * sizeof(float);
        }
    if (trunk > 0)
        at.trunk = layout.add(RegionKind::fixed, trunk);
    if (drafter && g.n_mtp > 0) {
        for (int b = 0; b < g.n_mtp; ++b)
            at.ring[(size_t) (g.nl + b)] = (uint64_t) b * ring;
        at.drafter = layout.add(RegionKind::fixed, (uint64_t) g.n_mtp * ring);
    }

    std::vector<bool> reads(sources.size()), scores(sources.size());
    for (int il = lb; il < le; ++il) {
        const int s = source_of[(size_t) il];
        if (s < 0)
            continue;
        reads[(size_t) s] = true;
        if (std::find(g.ix_src.begin(), g.ix_src.end(), il) != g.ix_src.end())
            scores[(size_t) s] = true;
    }
    for (size_t s = 0; s < sources.size(); ++s) {
        if (reads[s])
            at.ckv[s] = layout.add(RegionKind::per_block, kv_row_bytes(kKvFp4, g.hd), 1, sources[s].ratio);
        if (scores[s])
            at.ik[s] = layout.add(RegionKind::per_block, kv_row_bytes(kKvFp4p, g.ix_hd), 1, sources[s].ratio);
    }
    return true;
}

}  // namespace strata::ds4

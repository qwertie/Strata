// include/strata/models/qwen38/conversation_pages.hpp - a Qwen3.8 conversation's state in a GPU page pool: the GDN
// layers' recurrent state in a fixed region and every QSA layer's K/V and pooled indexer keys in units of the pool's
// pages, which grow with the conversation (docs/MEMORY_MANAGER_PLAN.md, step M2d).
//
// The session the pages are for is carved with session_set_pool_state, so it holds neither; `init` points it at the
// pool's pages.  The small running state (the PLE history and the indexer's tail, dead key and block position, the
// first region of describe_conversation) stays in the session for now.
#pragma once

#include "strata/common/state/page_pool.hpp"
#include "strata/core/geometry.hpp"
#include "strata/models/qwen38/kernels/kv_units.hpp"
#include "strata/models/qwen38/session.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::qwen38 {

/// One conversation's pages in a pool of the current GPU's memory; strata-q38 holds one conversation at a time.
class ConversationPages {
public:
    ConversationPages() = default;
    ~ConversationPages();
    ConversationPages(const ConversationPages&) = delete;
    ConversationPages& operator=(const ConversationPages&) = delete;

    /// Takes `pool_bytes` of device memory (whole chunks; no more than a conversation at the session's context can
    /// use) for the pool, gives the conversation its fixed region and points `ss` at it: `ss.gdn_state`, and each
    /// QSA state's K/V and pooled keys as units (QsaState::units).
    bool init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, uint64_t pool_bytes,
              std::string& err);
    /// Gives the conversation the pages for appending positions below `tokens` (and the spare indexer block after
    /// them), zeroes the new pages and writes their units' addresses into the QSA states' tables.  Call before each
    /// append; false when the pool has no room.
    bool reserve(int64_t tokens, std::string& err);
    /// Gives back the growing pages past the first `tokens` tokens (0: a new conversation); the fixed region stays.
    void truncate(int64_t tokens);
    /// Gets the pool's pages by use.
    strata::core::PagePoolReport report() const { return pool_->report(); }

private:
    static constexpr strata::core::ConversationId kId = 1;   ///< the one conversation's id in the pool

    std::unique_ptr<strata::core::PagePool> pool_;
    strata::kernels::KvUnitLayout unit_;         ///< one indexer block of a QSA layer
    uint64_t page_bytes_ = 0;
    int64_t idx_block_ = 0;
    int64_t max_blocks_ = 0;                     ///< entries of each table (the session's max_cells in blocks, + 2)
    int64_t reserved_ = 0;                       ///< the tokens the last reserve() gave pages for
    std::vector<int64_t*> tables_;               ///< per QSA layer, the device table its state's `units` points at
    std::vector<size_t> pages_in_table_;         ///< per QSA layer, the pages whose units its table holds
};

}  // namespace strata::qwen38

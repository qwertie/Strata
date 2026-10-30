// include/strata/common/state/page_pool.hpp - one memory (a GPU's, or the RAM the engine uses) cut into equal pages
// that experts and conversations trade while the engine runs.  docs/MEMORY_MANAGER_PLAN.md gives the design;
// src/common/state/README.md says how a conversation's state regions (state_layout.hpp) map onto pages.
#pragma once

#include "strata/common/state/state_layout.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// Index of a page in its pool.  It stays valid until a shrink gives back the chunk that holds it.
using PageId = uint32_t;
constexpr PageId no_page = UINT32_MAX;
/// The engine's handle on a conversation.
using ConversationId = uint64_t;

/// Names one expert's weights: its layer, and its index among that layer's experts.
struct ExpertKey {
    int32_t layer = 0, expert = 0;
    bool operator<(const ExpertKey& o) const { return layer != o.layer ? layer < o.layer : expert < o.expert; }
};

/// Sizes and limits of a pool.  The defaults are the plan's starting points, not measured ones.
struct PagePoolConfig {
    uint64_t page_bytes = 2ull << 20;      ///< a Qwen3.8 IQ3_S expert is 1.95 MiB
    uint64_t chunk_bytes = 256ull << 20;   ///< the amount taken from and given back to the allocator at once
    /// Conversation tokens whose pages experts may not take, so that a new conversation never waits.  Counted as
    /// one conversation of that many tokens.
    int64_t tokens_min = 0;
    /// The most tokens all the pool's conversations may hold together.
    int64_t tokens_max = INT64_MAX;
    /// Clock units during which a page that a conversation gave back stays closed to experts, so that the two do not
    /// take the same pages back and forth.
    uint64_t hysteresis = 2000;
};

/// What the pool calls outside itself.  The engine passes a device allocator (base/gpu) and a page copy on its
/// stream; tests pass host memory and a hand-moved clock.
struct PagePoolHooks {
    std::function<void*(size_t)> alloc;   ///< gets a chunk; nullptr when the memory has not that much
    std::function<void(void*)> free;      ///< gives a chunk back
    /// Gets the time for the experts' recency and the hysteresis: milliseconds by default.
    std::function<uint64_t()> now = [] {
        return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    /// Tells the expert's owner that its page goes to another use, before the new user gets it, so that the owner can
    /// copy the expert to RAM or drop it.
    std::function<void(ExpertKey, PageId)> evicted = [](ExpertKey, PageId) {};
    /// Copies a whole page when a shrink moves a conversation's page out of a chunk; returns false when it cannot.
    std::function<bool(PageId to, PageId from)> copy = [](PageId, PageId) { return false; };
};

/// A pool's pages by use, for the GET_MEMORY_USE command.
struct PagePoolReport {
    struct Use {
        uint64_t pages = 0, bytes = 0;
    };
    Use experts;
    Use active;    ///< the growing regions (K/V) of active conversations
    Use fixed;     ///< the fixed regions of active conversations: recurrent state, the drafter's state
    Use paused;    ///< all pages of paused conversations
    Use free;
    Use total;
    int active_conversations = 0, paused_conversations = 0;
    int64_t tokens = 0;   ///< held by all the pool's conversations
};

/// Outcome of PagePool::resize.
struct PoolResize {
    std::string refusal;   ///< why nothing (or, for a failed allocation or copy, not all) was done; empty when done
    /// Conversations holding pages in the chunks a shrink would give back, when the rest of the pool has no room for
    /// those pages: pausing (and removing) some of them, or any others, makes the room.
    std::vector<ConversationId> to_pause;
};

/// The pages of one memory: each is free, holds an expert, or holds part of one region of a conversation (a fixed
/// region's bytes, or a block of tokens of a growing region, all rows).  A growing conversation takes free pages,
/// then the least-used experts' pages; experts take free pages only above the token minimum's reserve and after the
/// hysteresis.  Not thread-safe: the engine's step loop is its only caller.
class PagePool {
public:
    /// Throws std::invalid_argument when a chunk holds no page or a page holds no token of a growing region.
    PagePool(const StateLayout& layout, PagePoolConfig config, PagePoolHooks hooks);
    ~PagePool();
    PagePool(const PagePool&) = delete;
    PagePool& operator=(const PagePool&) = delete;

    /// Takes or gives back chunks so that the pool holds `bytes` (rounded down to whole chunks).  A shrink gives back
    /// the last chunks: their experts are evicted and their conversation pages copied into the others (so read
    /// conversation_pages again).  It is refused, changing nothing, when an active conversation or the token minimum
    /// would not fit.
    PoolResize resize(uint64_t bytes);
    /// Changes the token limits; returns why not (changing nothing) or an empty string.
    std::string set_token_limits(int64_t min, int64_t max);

    /// Gives conversation `id` (added if new) the pages for `tokens` tokens; fewer tokens than it has keep its
    /// pages.  Returns why not (changing nothing) or an empty string.
    std::string grow_conversation(ConversationId id, int64_t tokens);
    /// Marks a conversation paused (it may not grow, and a resize or the limits need not fit it) or active again.
    /// Its pages stay until remove_conversation.
    void set_paused(ConversationId id, bool paused);
    /// Frees a conversation's pages; experts may take them after the hysteresis.
    void remove_conversation(ConversationId id);
    /// Gets the pages of one region of a conversation, in order: a fixed region's bytes, or its blocks of tokens.
    const std::vector<PageId>& conversation_pages(ConversationId id, size_t region) const;

    /// Gets a page for an expert: the one it has, a free page, or the least-used expert's.  no_page when none.
    PageId add_expert(ExpertKey key);
    /// Counts a use of an expert for eviction and gets its page; no_page when the pool does not hold it.
    PageId use_expert(ExpertKey key);
    /// Frees an expert's page at its owner's request (without calling `evicted`).
    void remove_expert(ExpertKey key);

    /// Gets where a page starts in the pool's memory (a device address for a GPU's pool).
    uint8_t* page_address(PageId page) const;
    /// Gets the pages one conversation of `tokens` tokens takes.
    uint64_t pages_for(int64_t tokens) const;
    PagePoolReport report() const;

private:
    enum class PageUse : uint8_t { free, expert, conversation };
    struct Page {
        PageUse use = PageUse::free;
        uint64_t open_at = 0;     ///< free: when experts may take it
        uint64_t last_use = 0;    ///< expert
        uint64_t uses = 0;        ///< expert
        ExpertKey expert;
        ConversationId conversation = 0;
        uint32_t region = 0, index = 0;   ///< conversation: the page is pages[region][index] of it
    };
    struct Conversation {
        int64_t tokens = 0;
        bool paused = false;
        std::vector<std::vector<PageId>> pages;   ///< per region of the layout
    };

    PoolResize shrink(size_t chunks);
    std::string check_limits(uint64_t pages, int64_t min, int64_t max) const;
    std::string check_growth(const Conversation& c, int64_t tokens) const;
    uint64_t region_pages(size_t region, int64_t tokens) const;
    uint64_t reserved_pages() const;
    PageId find_page(bool for_expert, PageId end);
    PageId evict_least_used(PageId end);
    void evict(PageId page);
    void put(PageId page, const Page& content);
    void release(PageId page, uint64_t open_at);
    std::string gib(uint64_t pages) const;

    const StateLayout layout_;
    PagePoolConfig config_;
    PagePoolHooks hooks_;
    const uint64_t pages_per_chunk_;
    std::vector<void*> chunks_;
    std::vector<Page> pages_;
    std::set<PageId> free_;
    std::map<ExpertKey, PageId> experts_;
    std::map<ConversationId, Conversation> conversations_;
};

}  // namespace strata::core

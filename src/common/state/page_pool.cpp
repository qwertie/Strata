// src/common/state/page_pool.cpp - the page pool (page_pool.hpp).  Experts are found by linear scans: a pool holds
// some ten thousand pages and changes a few of them per step.
#include "strata/common/state/page_pool.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <tuple>

namespace strata::core {

PagePool::PagePool(const StateLayout& layout, PagePoolConfig config, PagePoolHooks hooks)
    : layout_(layout), config_(config), hooks_(std::move(hooks)),
      pages_per_chunk_(config.chunk_bytes / config.page_bytes) {
    if (pages_per_chunk_ == 0)
        throw std::invalid_argument("a chunk of the page pool holds no page");
    for (const StateRegion& r : layout_.regions)
        if (r.kind != RegionKind::fixed && r.tokens_in(config_.page_bytes) == 0)
            throw std::invalid_argument("a page of the page pool holds no token of a growing region");
}

PagePool::~PagePool() {
    for (void* chunk : chunks_)
        hooks_.free(chunk);
}

PoolResize PagePool::resize(uint64_t bytes) {
    const size_t chunks = bytes / config_.chunk_bytes;
    while (chunks_.size() < chunks) {
        void* chunk = hooks_.alloc(config_.chunk_bytes);
        if (chunk == nullptr)
            return {"the memory had room for " + std::to_string(chunks_.size()) + " of " + std::to_string(chunks) +
                    " chunks", {}};
        chunks_.push_back(chunk);
        for (uint64_t i = 0; i < pages_per_chunk_; ++i) {
            free_.insert((PageId) pages_.size());
            pages_.emplace_back();
        }
    }
    return chunks < chunks_.size() ? shrink(chunks) : PoolResize{};
}

std::string PagePool::set_token_limits(int64_t min, int64_t max) {
    std::string why = check_limits(pages_.size(), min, max);
    if (why.empty()) {
        config_.tokens_min = min;
        config_.tokens_max = max;
    }
    return why;
}

std::string PagePool::grow_conversation(ConversationId id, int64_t tokens) {
    auto [it, added] = conversations_.try_emplace(id);
    Conversation& c = it->second;
    if (added)
        c.pages.resize(layout_.regions.size());
    if (std::string why = check_growth(c, tokens); !why.empty()) {
        if (added)
            conversations_.erase(it);
        return why;
    }
    for (uint32_t r = 0; r < c.pages.size(); ++r)
        for (const uint64_t n = region_pages(r, tokens); c.pages[r].size() < n;) {
            const PageId page = find_page(false, (PageId) pages_.size());
            Page content;
            content.use = PageUse::conversation;
            content.conversation = id;
            content.region = r;
            content.index = (uint32_t) c.pages[r].size();
            put(page, content);
            c.pages[r].push_back(page);
        }
    c.tokens = std::max(c.tokens, tokens);
    return {};
}

void PagePool::set_paused(ConversationId id, bool paused) {
    conversations_.at(id).paused = paused;
}

void PagePool::remove_conversation(ConversationId id) {
    const uint64_t open_at = hooks_.now() + config_.hysteresis;
    for (const std::vector<PageId>& region : conversations_.at(id).pages)
        for (PageId page : region)
            release(page, open_at);
    conversations_.erase(id);
}

const std::vector<PageId>& PagePool::conversation_pages(ConversationId id, size_t region) const {
    return conversations_.at(id).pages.at(region);
}

PageId PagePool::add_expert(ExpertKey key) {
    if (const PageId held = use_expert(key); held != no_page)
        return held;
    const PageId page = find_page(true, (PageId) pages_.size());
    if (page == no_page)
        return no_page;
    Page content;
    content.use = PageUse::expert;
    content.expert = key;
    content.last_use = hooks_.now();
    content.uses = 1;
    put(page, content);
    experts_[key] = page;
    return page;
}

PageId PagePool::use_expert(ExpertKey key) {
    const auto it = experts_.find(key);
    if (it == experts_.end())
        return no_page;
    Page& page = pages_[it->second];
    page.last_use = hooks_.now();
    ++page.uses;
    return it->second;
}

void PagePool::remove_expert(ExpertKey key) {
    const auto it = experts_.find(key);
    if (it == experts_.end())
        return;
    release(it->second, 0);
    experts_.erase(it);
}

uint8_t* PagePool::page_address(PageId page) const {
    return (uint8_t*) chunks_[page / pages_per_chunk_] + page % pages_per_chunk_ * config_.page_bytes;
}

uint64_t PagePool::pages_for(int64_t tokens) const {
    uint64_t n = 0;
    for (size_t r = 0; r < layout_.regions.size(); ++r)
        n += region_pages(r, tokens);
    return n;
}

PagePoolReport PagePool::report() const {
    PagePoolReport out;
    uint64_t active = 0, fixed = 0, paused = 0;
    for (const auto& [id, c] : conversations_) {
        ++(c.paused ? out.paused_conversations : out.active_conversations);
        out.tokens += c.tokens;
        for (size_t r = 0; r < c.pages.size(); ++r)
            (c.paused ? paused : layout_.regions[r].kind == RegionKind::fixed ? fixed : active) += c.pages[r].size();
    }
    const auto use = [&](uint64_t pages) { return PagePoolReport::Use{pages, pages * config_.page_bytes}; };
    out.experts = use(experts_.size());
    out.active = use(active);
    out.fixed = use(fixed);
    out.paused = use(paused);
    out.free = use(free_.size());
    out.total = use(pages_.size());
    return out;
}

/// Gives back all chunks past the first `chunks`, after evicting their experts and moving their conversation pages.
PoolResize PagePool::shrink(size_t chunks) {
    const PageId end = (PageId) (chunks * pages_per_chunk_);
    if (std::string why = check_limits(end, config_.tokens_min, config_.tokens_max); !why.empty())
        return {why, {}};
    uint64_t moving = 0, room = 0;
    std::set<ConversationId> holders;
    for (PageId p = 0; p < pages_.size(); ++p) {
        const bool kept = p < end, conversation = pages_[p].use == PageUse::conversation;
        if (kept && !conversation)
            ++room;
        else if (!kept && conversation) {
            ++moving;
            holders.insert(pages_[p].conversation);
        }
    }
    if (moving > room)
        return {"the conversations' pages need " + gib(moving - room) + " more than the smaller pool keeps",
                {holders.begin(), holders.end()}};
    for (PageId from = end; from < pages_.size(); ++from) {
        const Page content = pages_[from];
        if (content.use == PageUse::expert)
            evict(from);
        if (content.use != PageUse::conversation)
            continue;
        const PageId to = find_page(false, end);
        if (!hooks_.copy(to, from))
            return {"a conversation's page could not be copied", {}};
        put(to, content);
        conversations_.at(content.conversation).pages[content.region][content.index] = to;
        release(from, 0);
    }
    for (size_t c = chunks; c < chunks_.size(); ++c)
        hooks_.free(chunks_[c]);
    chunks_.resize(chunks);
    pages_.resize(end);
    free_.erase(free_.lower_bound(end), free_.end());
    return {};
}

/// Gets why a pool of `pages` pages could not keep the token limits `min` and `max` and the active conversations,
/// or an empty string.
std::string PagePool::check_limits(uint64_t pages, int64_t min, int64_t max) const {
    if (min < 0 || min > max)
        return "the token minimum must be from 0 to the maximum";
    if (min > 0 && pages_for(min) > pages)
        return "the token minimum needs " + gib(pages_for(min)) + ", the pool has " + gib(pages);
    for (const auto& [id, c] : conversations_) {
        if (c.paused)
            continue;
        if (c.tokens > max)
            return "an active conversation holds " + std::to_string(c.tokens) + " tokens";
        if (pages_for(c.tokens) > pages)
            return "an active conversation needs " + gib(pages_for(c.tokens)) + ", the pool has " + gib(pages);
    }
    return {};
}

/// Gets why conversation `c` cannot grow to `tokens` tokens, or an empty string.
std::string PagePool::check_growth(const Conversation& c, int64_t tokens) const {
    if (c.paused)
        return "the conversation is paused";
    int64_t all = tokens - c.tokens;
    for (const auto& [id, other] : conversations_)
        all += other.tokens;
    if (tokens > c.tokens && all > config_.tokens_max)
        return "conversations would hold " + std::to_string(all) + " tokens, above the maximum of " +
               std::to_string(config_.tokens_max);
    uint64_t needed = 0;
    for (size_t r = 0; r < c.pages.size(); ++r)
        needed += std::max(region_pages(r, tokens), (uint64_t) c.pages[r].size()) - c.pages[r].size();
    const uint64_t room = free_.size() + experts_.size();
    if (needed > room)
        return "the conversation needs " + gib(needed) + " more, the pool can free " + gib(room);
    return {};
}

uint64_t PagePool::region_pages(size_t region, int64_t tokens) const {
    const StateRegion& r = layout_.regions[region];
    const uint64_t page = config_.page_bytes;
    if (r.kind == RegionKind::fixed)
        return (r.bytes_for(0) + page - 1) / page;
    const uint64_t per_page = r.tokens_in(page);
    return ((uint64_t) tokens + per_page - 1) / per_page;
}

/// Gets how many free pages experts must leave, so that the conversations can reach the token minimum.
uint64_t PagePool::reserved_pages() const {
    if (config_.tokens_min == 0)
        return 0;
    uint64_t held = 0;
    for (const auto& [id, c] : conversations_)
        for (const std::vector<PageId>& region : c.pages)
            held += region.size();
    return std::max(pages_for(config_.tokens_min), held) - held;
}

/// Gets a free page before `end` (evicting the least-used expert if none is free), still marked free; no_page when
/// there is none.  A conversation takes the lowest free page.  An expert takes the highest open one above the
/// reserve, so that the last chunks, which a shrink gives back, hold experts rather than conversation pages.
PageId PagePool::find_page(bool for_expert, PageId end) {
    if (!for_expert && !free_.empty() && *free_.begin() < end)
        return *free_.begin();
    if (for_expert && free_.size() > reserved_pages()) {
        const uint64_t now = hooks_.now();
        for (auto it = free_.rbegin(); it != free_.rend(); ++it)
            if (*it < end && pages_[*it].open_at <= now)
                return *it;
    }
    return evict_least_used(end);
}

/// Evicts the expert before `end` used longest ago (of equals, the one used least often) and gets its page, now
/// free; no_page when there is none.
PageId PagePool::evict_least_used(PageId end) {
    auto victim = experts_.end();
    for (auto it = experts_.begin(); it != experts_.end(); ++it) {
        const Page& p = pages_[it->second];
        if (it->second < end && (victim == experts_.end() || std::tie(p.last_use, p.uses) <
                                 std::tie(pages_[victim->second].last_use, pages_[victim->second].uses)))
            victim = it;
    }
    if (victim == experts_.end())
        return no_page;
    const PageId page = victim->second;
    evict(page);
    return page;
}

/// Tells the owner that the expert on `page` leaves it, and frees the page.
void PagePool::evict(PageId page) {
    const ExpertKey key = pages_[page].expert;
    hooks_.evicted(key, page);
    experts_.erase(key);
    release(page, 0);
}

void PagePool::put(PageId page, const Page& content) {
    free_.erase(page);
    pages_[page] = content;
}

/// Marks a page free; experts may take it from clock time `open_at`.
void PagePool::release(PageId page, uint64_t open_at) {
    pages_[page] = Page{};
    pages_[page].open_at = open_at;
    free_.insert(page);
}

/// Gets the size of `pages` pages as text, in GiB.
std::string PagePool::gib(uint64_t pages) const {
    char text[32];
    std::snprintf(text, sizeof text, "%.2f GiB", (double) (pages * config_.page_bytes) / (1ull << 30));
    return text;
}

}  // namespace strata::core

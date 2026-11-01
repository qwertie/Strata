// src/common/experts/expert_pages.cpp - a GPU's expert cache in the pages of its page pool (expert_pages.hpp).
#include "strata/common/experts/expert_pages.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace strata::core {

uint64_t choose_expert_page_bytes(const std::vector<uint64_t>& blob_bytes) {
    constexpr uint64_t align = 256, smallest = 16 << 10, largest = 512 << 10;
    uint64_t gcd = 0;
    for (const uint64_t b : blob_bytes)
        gcd = std::gcd(gcd, b);
    if (gcd % align == 0 && gcd >= smallest)
        return gcd;
    uint64_t best = smallest, best_used = UINT64_MAX;
    for (uint64_t page = smallest; page <= largest; page += align) {
        uint64_t used = 0;
        for (const uint64_t b : blob_bytes)
            used += (b + page - 1) / page * page;
        if (used < best_used) {
            best = page;
            best_used = used;
        }
    }
    return best;
}

bool ExpertPages::open(ExpertCache& cache, const StateLayout& layout, PagePoolConfig config,
                       const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
                       const std::vector<int32_t>& ranked_layers, std::string& err) {
    const uint64_t page = (uint64_t) cache.slot_bytes(), pages = (uint64_t) cache.slots();
    if (!cache.valid() || cache.slot_offsets() != nullptr || cache.resident() != 0 || page == 0) {
        err = "the expert pages need an open, empty cache of uniform slots";
        return false;
    }
    // The whole arena is one chunk: nothing gives part of it back yet, and an expert may not cross a chunk's end, so
    // chunks would leave a run too short for any expert at each end (0.36% of the Strix Halo's 15.5 GiB pool in 256
    // MiB chunks).  Giving memory back while running (MEMORY_MANAGER_PLAN.md step M6) has to choose chunks again.
    config.page_bytes = page;
    config.chunk_bytes = pages * page;
    cache_ = &cache;
    n_expert_ = n_expert;
    max_experts_ = max_experts;
    pages_.clear();
    for (const uint64_t b : blob_bytes)
        pages_.push_back((uint32_t) ((b + page - 1) / page));
    rank_.assign(blob_bytes.size() * (size_t) n_expert, -1);
    config.expert_ranking.clear();
    for (size_t i = 0; i < ranked_layers.size() && (int64_t) i < max_experts; ++i)
        config.expert_ranking.push_back(pages_.at((size_t) ranked_layers[i]));
    PagePoolHooks hooks;
    hooks.alloc = [this, chunk = config.chunk_bytes, pages, page](size_t) -> void* {
        if ((chunks_given_ + 1) * chunk > pages * page)
            return nullptr;
        return cache_->device_slot((int32_t) (chunks_given_++ * chunk / page));
    };
    hooks.free = [this](void*) { --chunks_given_; };
    hooks.evicted = [this](ExpertKey key, PageId) { on_evicted(key); };
    now_ = hooks.now;
    t0_ = std::max<uint64_t>(now_(), rank_.size() + 1);
    try {
        pool_ = std::make_unique<PagePool>(layout, config, std::move(hooks));
    } catch (const std::exception& e) {
        err = std::string("the page pool: ") + e.what();
        return false;
    }
    if (const PoolResize r = pool_->resize(config.chunk_bytes); !r.refusal.empty()) {
        err = "the page pool: " + r.refusal;
        return false;
    }
    cache.set_admitter([this](int64_t layer, int64_t expert) { return admit(layer, expert); });
    return true;
}

bool ExpertPages::is_refill_wanted() const {
    return !evicted_.empty() && pool_->report().free.pages != blocked_at_free_;
}

int ExpertPages::refill(int max, uint64_t& wait_ms, std::string& err) {
    wait_ms = 0;
    int loaded = 0;
    while (loaded < max && !evicted_.empty()) {
        const ExpertKey key = ranked_[(size_t) *evicted_.begin()];
        const int32_t slot = cache_->admit(key.layer, key.expert);
        if (slot == kNotResident) {
            const uint64_t opening = pool_->get_next_opening();
            if (opening > 0)
                wait_ms = opening - std::min(opening, now_());
            else
                blocked_at_free_ = pool_->report().free.pages;
            break;
        }
        if (!hooks_.load(key, slot, err))
            return -1;
        hooks_.moved(key, slot);
        ++loaded;
    }
    return loaded;
}

/// Gets pages for an expert the cache admits (ExpertCache::set_admitter): a new one ranks after those admitted
/// before it, an evicted one comes back with its rank.
int32_t ExpertPages::admit(int64_t layer, int64_t expert) {
    const ExpertKey key{(int32_t) layer, (int32_t) expert};
    int32_t& rank = rank_[index_of(key)];
    if (rank < 0 && (int64_t) ranked_.size() >= max_experts_)
        return kNotResident;
    const int32_t r = rank >= 0 ? rank : (int32_t) ranked_.size();
    const PageId page = pool_->add_expert(key, pages_[(size_t) layer], get_used_at(r));
    if (page == no_page)
        return kNotResident;
    if (rank < 0) {
        rank = r;
        ranked_.push_back(key);
    }
    evicted_.erase(r);
    blocked_at_free_ = UINT64_MAX;
    return (int32_t) page;
}

/// Forgets an expert whose pages the pool gave to another use, and keeps it for `refill`.
void ExpertPages::on_evicted(ExpertKey key) {
    cache_->evict(key.layer, key.expert);
    evicted_.insert(rank_[index_of(key)]);
    if (hooks_.moved)
        hooks_.moved(key, kNotResident);
}

}  // namespace strata::core

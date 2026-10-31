// include/strata/common/experts/expert_pages.hpp - a GPU's expert cache in the pages of the page pool that the GPU's
// conversations take pages from too (docs/MEMORY_MANAGER_PLAN.md, step M2e).
//
// The cache's arena is opened as uniform slots of one page each, and the pool is made over that arena, so a page's
// number is a slot's: the kernels find an expert at the arena plus its first page times the page size, as in any
// cache of uniform slots, and an expert larger than a page takes several (ExpertCache::set_expert_bytes).  Experts
// come in through ExpertCache::admit, in the expert profile's order, and leave when a growing conversation takes
// their pages.  An expert that leaves is not copied anywhere: the CPU then computes it from the RAM copy of the
// experts when that holds it, else from the model's files, as for any expert the GPU does not hold.  Copying it back
// to RAM first is worth it only where that is clearly faster than reading the drive again and the RAM copy has room
// for it; the RAM copy (FileExpertSource::pin_cache_complement) has no room beyond its experts yet.
#pragma once

#include "strata/common/state/page_pool.hpp"
#include "strata/core/expert_cache.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// Gets the page size for experts whose sizes are `blob_bytes` (one per layer; the experts of a layer are all one
/// size): the multiple of 256 bytes from 256 KiB to 512 KiB that wastes the least of the pages the experts take.  The
/// lower bound keeps the pool's page count, which its linear scans walk, in the tens of thousands.
uint64_t choose_expert_page_bytes(const std::vector<uint64_t>& blob_bytes);

/// The expert cache of one GPU, whose slots are the pages of a page pool it makes over the cache's arena.  The pool's
/// conversations take the least recently used experts' pages as they grow, and `refill` puts those experts back once
/// pages are free again.  An expert's last use is its rank in the order experts were first admitted (the expert
/// profile's, hottest first), so the coldest leave first.  Not thread-safe: the engine's step loop is its only caller.
class ExpertPages {
public:
    /// The engine's side of the cache: copying an expert in and publishing where experts are.
    struct Hooks {
        /// Copies an expert's bytes into its slot and waits for the copy; false (and `err`) when it could not.
        std::function<bool(ExpertKey key, int32_t slot, std::string& err)> load;
        /// Tells the engine where an expert is now: `slot`, or kNotResident when it left the GPU.  The engine updates
        /// the residency tables its kernels read before its next step.
        std::function<void(ExpertKey key, int32_t slot)> moved;
    };

    /// Makes the pool over `cache`'s arena (open, with uniform slots of the page size and no expert yet), with the
    /// conversations' `layout` and `config`'s limits; `config`'s page and chunk sizes are taken from the cache.  The
    /// experts of layer l are `blob_bytes[l]` bytes, `n_expert` per layer; at most `max_experts` experts are
    /// admitted.  From then on the cache's `admit` takes pages from the pool.
    bool open(ExpertCache& cache, const StateLayout& layout, PagePoolConfig config,
              const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts, std::string& err);
    void set_hooks(Hooks hooks) { hooks_ = std::move(hooks); }
    /// Gets the page pool that the experts share with the GPU's conversations.
    PagePool& pool() { return *pool_; }
    const PagePool& pool() const { return *pool_; }

    /// Tells whether experts that conversations evicted wait for pages that may be free for them now or soon.
    bool is_refill_wanted() const;
    /// Puts back up to `max` evicted experts, hottest first, into the pages experts may take.  Gets how many (-1 when
    /// a copy failed, with `err`); `wait_ms` gets 0, or the milliseconds until more pages open to experts when the
    /// next expert found no room before then.
    int refill(int max, uint64_t& wait_ms, std::string& err);

private:
    int32_t admit(int64_t layer, int64_t expert);
    void on_evicted(ExpertKey key);
    size_t index_of(ExpertKey key) const { return (size_t) key.layer * (size_t) n_expert_ + (size_t) key.expert; }
    /// Gets the clock time that stands for the last use of the expert of admission rank `rank`.
    uint64_t get_used_at(int32_t rank) const { return t0_ - (uint64_t) rank; }

    ExpertCache* cache_ = nullptr;
    std::unique_ptr<PagePool> pool_;
    Hooks hooks_;
    std::function<uint64_t()> now_;      ///< the pool's clock
    std::vector<uint32_t> pages_;        ///< per layer, the pages one of its experts takes
    int64_t n_expert_ = 0, max_experts_ = 0;
    uint64_t t0_ = 0;                    ///< the clock time given to the hottest expert's last use
    std::vector<int32_t> rank_;          ///< per expert, its admission rank; -1 before its first admission
    std::vector<ExpertKey> ranked_;      ///< the experts in rank order
    std::set<int32_t> evicted_;          ///< the ranks of the experts a conversation evicted
    uint64_t chunks_given_ = 0;          ///< the pool's chunks, carved in order from the cache's arena
    /// The pool's free pages when `refill` last found no room and no page about to open; refills wait until that
    /// changes.  UINT64_MAX: not waiting.
    uint64_t blocked_at_free_ = UINT64_MAX;
};

}  // namespace strata::core

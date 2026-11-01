// include/strata/common/experts/expert_pages.hpp - experts in the pages of a page pool that conversations take pages
// from too: a GPU's expert cache, or the RAM's copy of the experts (docs/MEMORY_MANAGER_PLAN.md, steps M2e, M2f and
// M5).
//
// For a GPU, the cache's arena is opened as uniform slots of one page each and is the pool's first chunk; the other
// chunks are allocated above it at whole pages from its start, so that every page is a slot: the kernels find an
// expert at the arena plus its slot times the page size, as in any cache of uniform slots, and an expert takes
// several consecutive pages (ExpertCache::set_expert_bytes).  Experts come in through ExpertCache::admit, in the
// expert profile's order, and leave when a growing conversation takes their pages or a shrink gives back their
// chunk.  The RAM's pool (RamExperts) is made of page-locked host memory the same way, without a cache, its slots
// the pool's page numbers.  Before an expert's pages go to another use, the `leaving` hook may copy it elsewhere (a
// GPU's expert back to RAM).
#pragma once

#include "strata/common/state/page_pool.hpp"
#include "strata/core/expert_cache.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// Gets the page size for experts whose sizes are `blob_bytes` (one per layer; the experts of a layer are all one
/// size): their greatest common divisor, so that every expert is a whole number of pages and no page is part empty
/// (25,600 bytes for Qwen3.8 IQ3_S, 92,160 for DeepSeek's 2.00 bpw set).  A page must be a multiple of 256 bytes,
/// the alignment the cache gives every slot, and at least 16 KiB, so that a pool has no more pages than its slot
/// tables can count; when the divisor is not, the multiple of 256 bytes from 16 KiB to 512 KiB that wastes the least
/// of the pages the experts take.
uint64_t choose_expert_page_bytes(const std::vector<uint64_t>& blob_bytes);

/// The memory a pool's chunks come from (a GPU's, or page-locked RAM), given by the engine.
struct ChunkMemory {
    /// Gets `bytes` of the memory; null when it has not that much.
    std::function<uint8_t*(uint64_t bytes)> alloc;
    /// Gives back what `alloc` gave.
    std::function<void(uint8_t*)> free;
    /// Copies `bytes` inside the memory and waits for the copy (a shrink moves conversation pages out of a chunk);
    /// false when it failed.
    std::function<bool(uint8_t* to, const uint8_t* from, uint64_t bytes)> copy;
};

/// Experts in the pages of a page pool made over one arena: a GPU's expert cache (whose slots are the pages) or host
/// memory.  The pool's conversations take the least recently used experts' pages as they grow, and `refill` puts
/// those experts back once pages are free again.  An expert's last use is its rank in the ranking `open` is given
/// (an expert outside it ranks after it, in the order it is first admitted) until the engine counts a use
/// (note_use), so the coldest leave first.  Not thread-safe: the engine's step loop is its only caller.
class ExpertPages {
public:
    /// The engine's side of the pool: copying an expert in, publishing where experts are, and an expert's last look.
    struct Hooks {
        /// Copies an expert's bytes into its pages (from `first`) and waits for the copy; false (and `err`) when it
        /// could not.
        std::function<bool(ExpertKey key, int32_t first, std::string& err)> load;
        /// Tells the engine where an expert is now: its first page, or kNotResident when it left the pool.  A GPU's
        /// engine updates the residency tables its kernels read before its next step.
        std::function<void(ExpertKey key, int32_t first)> moved;
        /// Called with an expert's bytes (in the pool's memory) before its pages go to another use; may be empty.
        std::function<void(ExpertKey key, const uint8_t* bytes)> leaving;
    };

    /// Makes the pool over `cache`'s arena (open, with uniform slots of the page size and no expert yet), which is
    /// its first chunk, and chunks of `memory` up to `bytes` (whole chunks), with the conversations' `layout` and
    /// `config`'s limits; `config`'s page and chunk sizes are taken from the cache.  The experts of layer l are
    /// `blob_bytes[l]` bytes, `n_expert` per layer; at most `max_experts` experts are admitted.  `ranking` is the
    /// experts hottest first (the expert profile's order): their ranks, and the pool gives each expert size room for
    /// the ones of the first `max_experts` that fit (PagePoolConfig::expert_ranking).  From then on the cache's
    /// `admit` takes pages from the pool.  The cache's own fill takes the arena's slots; the experts of those first
    /// `max_experts` that it leaves out wait for `refill` (is_refill_wanted).
    bool open(ExpertCache& cache, ChunkMemory memory, uint64_t bytes, const StateLayout& layout,
              PagePoolConfig config, const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
              const std::vector<ExpertKey>& ranking, std::string& err);
    /// The same in chunks of `config.chunk_bytes` of `memory` only, without a cache: `admit` gives experts their
    /// pages, and a slot is a page's number.
    bool open(ChunkMemory memory, uint64_t bytes, const StateLayout& layout, PagePoolConfig config,
              const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
              const std::vector<ExpertKey>& ranking, std::string& err);
    /// Takes or gives back chunks so that the pool holds `bytes` (PagePool::resize, whose refusal it gets).  After a
    /// growth, `refill` brings in the experts of the first `max_experts` that the pool lacks.
    PoolResize resize(uint64_t bytes);
    /// Gets the slot of a pool's page: what the cache's residency and the kernels use.
    int32_t get_slot(PageId page) const;
    void set_hooks(Hooks hooks) { hooks_ = std::move(hooks); }
    /// Gets the page pool that the experts share with the memory's conversations.
    PagePool& pool() { return *pool_; }
    const PagePool& pool() const { return *pool_; }

    /// Gets pages for an expert: the free run nearest its size's home, or the run of the least recently used expert
    /// of its size when that one was last used before `used_at` (the pool's clock), which also becomes the new
    /// expert's last use; 0 stands for the time of the expert's rank, so that only lower-ranked experts give way.
    /// Gets its first page, or kNotResident when it may not come in or there is no room.  Its bytes are the caller's
    /// to copy in.
    int32_t admit(int64_t layer, int64_t expert, uint64_t used_at = 0);
    /// Moves expert `in` of `layer` into the pages of expert `out` of the same layer, which leaves as it would for a
    /// conversation (the `leaving` and `moved` hooks) but is not put back by `refill`, and copies `in`'s bytes in.
    /// False (and `err`) when the copy failed: then neither is in the pool.  Nothing changes (true) when the pool
    /// does not hold `out` or already holds `in`.
    bool replace(int32_t layer, int32_t out, int32_t in, std::string& err);
    /// Counts a use of an expert the pool holds, so that a conversation evicts it after the experts used less
    /// recently (nothing when the pool does not hold it).
    void note_use(ExpertKey key) { (void) pool_->use_expert(key); }
    /// Gets the pool's clock now (the times admit takes).
    uint64_t get_time() const { return now_(); }
    /// Tells whether experts that conversations evicted wait for pages that may be free for them now or soon.
    bool is_refill_wanted() const;
    /// Puts back up to `max` evicted experts, hottest first, into the pages experts may take.  Gets how many (-1 when
    /// a copy failed, with `err`); `wait_ms` gets 0, or the milliseconds until more pages open to experts when the
    /// next expert found no room before then.
    int refill(int max, uint64_t& wait_ms, std::string& err);

private:
    bool open_pool(ChunkMemory memory, uint64_t bytes, const StateLayout& layout, PagePoolConfig config,
                   const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
                   const std::vector<ExpertKey>& ranking, std::string& err);
    uint8_t* alloc_chunk(uint64_t bytes);
    void free_chunk(uint8_t* chunk);
    void on_evicted(ExpertKey key, PageId first);
    void want_missing();
    size_t index_of(ExpertKey key) const { return (size_t) key.layer * (size_t) n_expert_ + (size_t) key.expert; }
    /// Gets the clock time that stands for the last use of the expert of rank `rank`.
    uint64_t get_used_at(int32_t rank) const { return t0_ - (uint64_t) rank; }

    ExpertCache* cache_ = nullptr;       ///< the GPU's cache whose slots are the pages; null for host memory
    uint8_t* arena_ = nullptr;           ///< the cache's arena: its first chunk, and slot 0's address
    ChunkMemory memory_;
    std::map<uint8_t*, uint8_t*> raw_;   ///< each chunk `memory_` gave, by its start: the allocation it starts in
    std::unique_ptr<PagePool> pool_;
    Hooks hooks_;
    std::function<uint64_t()> now_;      ///< the pool's clock
    std::vector<uint32_t> pages_;        ///< per layer, the pages one of its experts takes
    int64_t n_expert_ = 0, max_experts_ = 0;
    uint64_t t0_ = 0;                    ///< the clock time given to the hottest expert's last use
    std::vector<int32_t> rank_;          ///< per expert, its rank; -1 while it has none
    std::vector<ExpertKey> ranked_;      ///< the experts in rank order
    std::set<int32_t> evicted_;          ///< the ranks of the experts `refill` brings in: evicted, or not yet in
    /// The pool's free pages when `refill` last found no room and no page about to open; refills wait until that
    /// changes.  UINT64_MAX: not waiting.
    uint64_t blocked_at_free_ = UINT64_MAX;
};

}  // namespace strata::core

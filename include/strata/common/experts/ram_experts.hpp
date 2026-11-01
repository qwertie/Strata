// include/strata/common/experts/ram_experts.hpp - the RAM's page pool (docs/MEMORY_MANAGER_PLAN.md, step M5): the
// experts a GPU's pool does not hold, copies of some it does while room allows, and paused conversations, in one pool
// of page-locked memory that the GPU can read in place.  The experts in neither pool are read from the SSD without
// the OS file cache (the file source's unbuffered reads), so that the cache does not take the RAM back.
#pragma once

#include "strata/common/experts/expert_pages.hpp"
#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The experts the CPU computes and a GPU's pool loads, as an ExpertSource: from the RAM's page pool when it holds
/// them, else read from the SSD.  An expert a step reads from the SSD (blob, prefetch) takes the place of the least
/// recently used one of its size, unless the current layer has used that one, so the pool keeps the experts the
/// conversations use; else it is read into a ring of staging buffers.  The pool's conversations are paused ones
/// (pool_copy.hpp); they take the least recently used experts' pages, and `pages().refill` puts those back.
///
/// Threads: the engine's thread makes every call but copy_blob and copy_blobs, which the prompt path's stager
/// threads make while the engine's thread reads a prompt chunk and changes nothing here.
class RamExperts : public ExpertSource {
public:
    RamExperts() = default;
    ~RamExperts() override;
    RamExperts(const RamExperts&) = delete;
    RamExperts& operator=(const RamExperts&) = delete;

    /// Makes the pool in `bytes` of page-locked RAM, less a staging ring of `ring` buffers of the largest expert (as
    /// many as the experts a step's layer may read from the SSD), with the conversations' `layout` and `config`'s
    /// limits; `config.page_bytes` is the GPU pool's, so that a conversation's blocks are alike in both.  Then reads
    /// the experts of `ranking` into it, in order, as far as they fit.  The experts of layer l are `blob_bytes[l]`
    /// bytes, `n_expert` per layer.  `files` reads the experts from the SSD and must outlive this object.
    bool open(ExpertSource& files, uint64_t bytes, int ring, const StateLayout& layout, PagePoolConfig config,
              const std::vector<uint64_t>& blob_bytes, int64_t n_expert, const std::vector<ExpertKey>& ranking,
              std::string& err);
    /// Copies expert `key` in from `bytes` (any memory: a GPU's pages it is leaving) when the pool does not hold it
    /// and has room for it among colder experts of its size.
    void keep_copy(ExpertKey key, const uint8_t* bytes);
    /// Tells whether the pool holds expert (`layer`, `expert`).
    bool holds(int64_t layer, int64_t expert) const { return first_page_[index_of(layer, expert)] != kNotResident; }
    /// Gets the experts in the pool, which the paused conversations share.
    ExpertPages& pages() { return pages_; }
    const ExpertPages& pages() const { return pages_; }
    /// Tells whether the pool's memory is page-locked, which the GPU reads in place (device_alias).
    bool is_page_locked() const { return device_view_ != nullptr; }
    /// Gets the bytes of the staging ring (outside the pool).
    uint64_t ring_bytes() const { return ring_slots_ * slot_bytes_; }
    /// Times reading a staging ring's worth of experts (one of every layer in turn) from the SSD, in GB/s; each
    /// `round` reads other experts, so that no read finds the drive's own cache warm.
    double time_ssd_reads(int round);
    /// Times copying `n` bytes from `bytes` (a GPU's memory) into the staging ring, in GB/s.
    double time_copy_in(const uint8_t* bytes, uint64_t n);

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override { return is_page_locked() && holds(layer, expert); }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t) const override { return is_page_locked(); }
    bool transient(int64_t layer, int64_t expert) const override { return !holds(layer, expert); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;

private:
    /// Experts to read from the SSD, and where each goes.
    struct Reads {
        std::vector<int32_t> layers, experts;
        std::vector<uint8_t*> dst;
        void add(int64_t layer, int64_t expert, uint8_t* to) {
            layers.push_back((int32_t) layer);
            experts.push_back((int32_t) expert);
            dst.push_back(to);
        }
    };

    size_t index_of(int64_t layer, int64_t expert) const { return (size_t) (layer * n_expert_ + expert); }
    uint8_t* get_page(int32_t first) const { return pages_.pool().page_address((PageId) first); }
    bool read_from_ssd(const Reads& reads);
    bool read_in(const Reads& reads);
    uint8_t* take_place(int64_t layer, int64_t expert);
    uint8_t* take_slot(int64_t layer, int64_t expert);
    const uint8_t* find_staged(int64_t layer, int64_t expert) const;

    ExpertSource* files_ = nullptr;
    ExpertPages pages_;
    uint8_t* host_ = nullptr;              ///< the pool's pages, then the staging ring
    uint8_t* device_view_ = nullptr;       ///< `host_` as the GPU reads it; null when the memory is not page-locked
    std::vector<uint64_t> blob_bytes_;
    int64_t n_expert_ = 0;
    std::vector<int32_t> first_page_;      ///< per expert, its first page in the pool, or kNotResident
    uint8_t* ring_ = nullptr;
    uint64_t slot_bytes_ = 0;              ///< a staging buffer: the largest expert, rounded up to 4 KiB
    size_t ring_slots_ = 0, next_slot_ = 0;
    std::vector<int64_t> ring_owner_;      ///< per staging buffer, the expert it holds (index_of), or -1
    /// The pool's clock when the CPU began the current layer (begin_layer): no expert used since gives its pages to
    /// one read from the SSD, as the layer's work may still read it (the previous layers' work is done).  0 before
    /// the first layer: only lower-ranked experts give way.
    uint64_t layer_t0_ = 0;
};

}  // namespace strata::core

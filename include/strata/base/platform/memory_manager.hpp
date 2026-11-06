#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace strata::core {

/// Names an allocation's category, owning pool and purpose. An empty category denotes backing for labelled blocks.
struct MemoryLabel {
    std::string category = "other", pool = "Device allocations", label = "Engine buffers";
};

/// Describes disjoint bytes within one memory; labels may contain arbitrary UTF-8 text.
struct MemoryBlock {
    MemoryLabel name;
    uint64_t bytes = 0;
};

/// Combines blocks with the same labels without prescribing any category vocabulary.
inline void add_memory_block(std::vector<MemoryBlock>& blocks, MemoryLabel name, uint64_t bytes) {
    if (!bytes)
        return;
    for (MemoryBlock& block : blocks)
        if (std::tie(block.name.category, block.name.pool, block.name.label) ==
            std::tie(name.category, name.pool, name.label)) {
            block.bytes += bytes;
            return;
        }
    blocks.push_back({std::move(name), bytes});
}

/// Keeps allocation lifetimes and labels shared by the device allocators and paged pools.
class MemoryManager {
public:
    /// Records a successful allocation. Reusing an address replaces its previous entry.
    void add(void* address, uint64_t bytes, std::string memory, MemoryLabel name, bool mapped = false) {
        if (!address)
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[address] = {std::move(memory), {std::move(name), bytes}, mapped, ++next_id_};
    }
    /// Forgets an allocation after its allocator successfully releases it.
    void remove(void* address) {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_.erase(address);
    }
    /// Gets the allocation's lifetime token; zero means the address is not registered.
    uint64_t get_allocation_id(void* address) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = allocations_.find(address);
        return it == allocations_.end() ? 0 : it->second.id;
    }
    /// Forgets a released allocation without removing a concurrent allocation that reused its address.
    void remove_if_current(void* address, uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = allocations_.find(address);
        if (it != allocations_.end() && it->second.id == id)
            allocations_.erase(it);
    }
    /// Assigns backing storage to a pool, or changes the purpose of an existing allocation.
    void set_label(void* address, MemoryLabel name) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = allocations_.find(address);
        if (it != allocations_.end())
            it->second.block.name = std::move(name);
    }
    /// Gets totals by label, excluding pool backing whose blocks are reported by the pool itself.
    std::vector<MemoryBlock> get_blocks(const std::string& memory) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<std::tuple<std::string, std::string, std::string>, uint64_t> totals;
        for (const auto& [address, allocation] : allocations_) {
            const MemoryBlock& b = allocation.block;
            if (allocation.memory == memory && !b.name.category.empty())
                totals[{b.name.category, b.name.pool, b.name.label}] += b.bytes;
        }
        std::vector<MemoryBlock> out;
        for (const auto& [name, bytes] : totals)
            out.push_back({{std::get<0>(name), std::get<1>(name), std::get<2>(name)}, bytes});
        return out;
    }
    /// Gets page-locked RAM already accounted for in RAM, including backing for page pools.
    uint64_t get_mapped_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t bytes = 0;
        for (const auto& [address, allocation] : allocations_)
            if (allocation.mapped)
                bytes += allocation.block.bytes;
        return bytes;
    }

private:
    /// Retains the physical memory separately from its pool and block labels.
    struct Allocation {
        std::string memory;
        MemoryBlock block;
        bool mapped = false;
        uint64_t id = 0;   ///< distinguishes successive allocations at the same address
    };
    mutable std::mutex mutex_;
    std::map<void*, Allocation> allocations_;
    uint64_t next_id_ = 0;   ///< assigned while the ledger is locked, independently of device synchronization
};

/// Gets the process allocator ledger, retained while static device caches release their allocations at exit.
inline MemoryManager& get_memory_manager() {
    static MemoryManager* manager = new MemoryManager;
    return *manager;
}

}  // namespace strata::core

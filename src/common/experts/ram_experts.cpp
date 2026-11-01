// src/common/experts/ram_experts.cpp - the RAM's page pool and the SSD's experts (ram_experts.hpp).
#include "strata/common/experts/ram_experts.hpp"

#include "strata/base/gpu/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace strata::core {
namespace {

/// The experts read from the SSD in one batch of requests while the pool fills.
constexpr size_t kFillBatch = 64;

/// Gets the seconds from `t0` to now.
double get_seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

RamExperts::~RamExperts() {
    if (device_view_ != nullptr)
        gpu_host_free(host_);
    else
        delete[] host_;
}

bool RamExperts::open(ExpertSource& files, uint64_t bytes, int ring, const StateLayout& layout, PagePoolConfig config,
                      const std::vector<uint64_t>& blob_bytes, int64_t n_expert, const std::vector<ExpertKey>& ranking,
                      std::string& err) {
    files_ = &files;
    blob_bytes_ = blob_bytes;
    n_expert_ = n_expert;
    first_page_.assign(blob_bytes.size() * (size_t) n_expert, kNotResident);
    slot_bytes_ = (*std::max_element(blob_bytes.begin(), blob_bytes.end()) + 4095) / 4096 * 4096;
    ring_slots_ = (size_t) std::max(ring, 1);
    ring_owner_.assign(ring_slots_, -1);
    const uint64_t pages = bytes > ring_bytes() ? (bytes - ring_bytes()) / config.page_bytes : 0;
    const uint64_t total = pages * config.page_bytes + ring_bytes();
    void* view = nullptr;
    host_ = (uint8_t*) gpu_host_alloc_mapped((size_t) total, &view);
    if (host_ != nullptr)
        device_view_ = (uint8_t*) view;
    else   // pageable memory: the CPU reads it, the GPU does not
        host_ = new (std::nothrow) uint8_t[(size_t) total];
    if (host_ == nullptr || pages == 0) {
        err = "the RAM's page pool: no room for " + std::to_string(total >> 20) + " MiB";
        return false;
    }
    ring_ = host_ + pages * config.page_bytes;
    // every expert may come in (the pool's room decides how many)
    if (!pages_.open(host_, pages, layout, config, blob_bytes, n_expert, (int64_t) first_page_.size(), ranking, err))
        return false;
    ExpertPages::Hooks hooks;
    hooks.load = [this](ExpertKey key, int32_t first, std::string& e) {
        Reads one;
        one.add(key.layer, key.expert, get_page(first));
        if (read_from_ssd(one))
            return true;
        e = "the SSD read of layer " + std::to_string(key.layer) + " expert " + std::to_string(key.expert) + " failed";
        return false;
    };
    hooks.moved = [this](ExpertKey key, int32_t first) { first_page_[index_of(key.layer, key.expert)] = first; };
    pages_.set_hooks(std::move(hooks));
    Reads batch;
    for (size_t i = 0; i < ranking.size(); ++i) {
        const ExpertKey key = ranking[i];
        if (const int32_t first = pages_.admit(key.layer, key.expert); first != kNotResident) {
            first_page_[index_of(key.layer, key.expert)] = first;
            batch.add(key.layer, key.expert, get_page(first));
        }
        if (batch.layers.size() < kFillBatch && i + 1 < ranking.size())
            continue;
        if (!read_from_ssd(batch)) {
            err = "the RAM's page pool: an SSD read failed";
            return false;
        }
        batch = {};
    }
    return true;
}

void RamExperts::keep_copy(ExpertKey key, const uint8_t* bytes) {
    if (holds(key.layer, key.expert))
        return;
    const int32_t first = pages_.admit(key.layer, key.expert);
    if (first == kNotResident)
        return;
    if (gpu_copy(get_page(first), bytes, (size_t) blob_bytes_[(size_t) key.layer]))
        first_page_[index_of(key.layer, key.expert)] = first;
    else
        pages_.pool().remove_expert(key);
}

double RamExperts::time_ssd_reads(int round) {
    std::fill(ring_owner_.begin(), ring_owner_.end(), -1);
    Reads reads;
    uint64_t bytes = 0;
    for (size_t i = 0; i < ring_slots_; ++i) {
        const size_t layer = i % blob_bytes_.size();
        // a prime stride spreads the reads over the files
        const size_t expert = (i * 7919 + (size_t) round * ring_slots_) % (size_t) n_expert_;
        reads.add((int64_t) layer, (int64_t) expert, ring_ + i * slot_bytes_);
        bytes += blob_bytes_[layer];
    }
    const auto t0 = std::chrono::steady_clock::now();
    return read_from_ssd(reads) ? (double) bytes / 1e9 / std::max(get_seconds_since(t0), 1e-9) : 0.0;
}

double RamExperts::time_copy_in(const uint8_t* bytes, uint64_t n) {
    n = std::min(n, ring_bytes());
    std::fill(ring_owner_.begin(), ring_owner_.end(), -1);
    const auto t0 = std::chrono::steady_clock::now();
    if (!gpu_copy(ring_, bytes, (size_t) n))
        return 0.0;
    return (double) n / 1e9 / std::max(get_seconds_since(t0), 1e-9);
}

const uint8_t* RamExperts::blob(int64_t layer, int64_t expert) {
    if (holds(layer, expert))
        return get_page((int32_t) pages_.pool().use_expert({(int32_t) layer, (int32_t) expert}));
    if (const uint8_t* staged = find_staged(layer, expert))
        return staged;
    Reads one;
    one.add(layer, expert, take_place(layer, expert));
    return read_in(one) ? one.dst[0] : nullptr;
}

const uint8_t* RamExperts::device_alias(int64_t layer, int64_t expert) const {
    const int32_t first = first_page_[index_of(layer, expert)];
    if (device_view_ == nullptr || first == kNotResident)
        return nullptr;
    return device_view_ + (get_page(first) - host_);
}

bool RamExperts::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const int32_t l = (int32_t) layer, e = (int32_t) expert;
    return copy_blobs(&l, &e, &dst, 1);
}

bool RamExperts::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    Reads reads;
    for (size_t i = 0; i < n; ++i)
        if (const int32_t first = first_page_[index_of(layers[i], experts[i])]; first != kNotResident)
            std::memcpy(dst[i], get_page(first), (size_t) blob_bytes_[(size_t) layers[i]]);
        else
            reads.add(layers[i], experts[i], dst[i]);
    return read_from_ssd(reads);
}

void RamExperts::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    Reads reads;
    for (int64_t i = 0; i < n; ++i)
        if (!holds(layer, experts[i]) && find_staged(layer, experts[i]) == nullptr)
            reads.add(layer, experts[i], take_place(layer, experts[i]));
    (void) read_in(reads);   // blob() reads them again, and fails there
}

void RamExperts::begin_layer(int64_t, const int32_t*, int64_t) {
    layer_t0_ = pages_.get_time();
}

/// Reads `reads` from the SSD (nothing: true).
bool RamExperts::read_from_ssd(const Reads& reads) {
    return reads.layers.empty() ||
           files_->copy_blobs(reads.layers.data(), reads.experts.data(), reads.dst.data(), reads.layers.size());
}

/// Reads `reads`, whose places take_place gave, from the SSD; when that fails, the places hold no expert.
bool RamExperts::read_in(const Reads& reads) {
    if (read_from_ssd(reads))
        return true;
    for (size_t i = 0; i < reads.dst.size(); ++i)
        if (const ExpertKey key{reads.layers[i], reads.experts[i]}; holds(key.layer, key.expert)) {
            pages_.pool().remove_expert(key);
            first_page_[index_of(key.layer, key.expert)] = kNotResident;
        } else {
            ring_owner_[(size_t) (reads.dst[i] - ring_) / slot_bytes_] = -1;
        }
    return false;
}

/// Gets where an expert read from the SSD goes: pages of the pool when it has room for it (the least recently used
/// expert of its size giving way, unless the current layer used it), else the next buffer of the staging ring.
uint8_t* RamExperts::take_place(int64_t layer, int64_t expert) {
    const int32_t first = pages_.admit(layer, expert, layer_t0_);
    if (first == kNotResident)
        return take_slot(layer, expert);
    first_page_[index_of(layer, expert)] = first;
    return get_page(first);
}

/// Gets the next buffer of the staging ring for an expert the pool does not hold; it keeps the expert's bytes until
/// the ring comes round to it again, which is after more experts than one step's layer reads.
uint8_t* RamExperts::take_slot(int64_t layer, int64_t expert) {
    const size_t slot = next_slot_;
    next_slot_ = (next_slot_ + 1) % ring_slots_;
    ring_owner_[slot] = (int64_t) index_of(layer, expert);
    return ring_ + slot * slot_bytes_;
}

/// Gets the staging buffer that holds an expert, or null.
const uint8_t* RamExperts::find_staged(int64_t layer, int64_t expert) const {
    const auto at = std::find(ring_owner_.begin(), ring_owner_.end(), (int64_t) index_of(layer, expert));
    return at == ring_owner_.end() ? nullptr : ring_ + (size_t) (at - ring_owner_.begin()) * slot_bytes_;
}

}  // namespace strata::core

// The RAM's page pool (ram_experts.hpp) over an in-memory "SSD": it fills with the ranking's first experts, hands out
// their bytes and reads the others into its staging ring (once each, also after prefetch), gives experts' pages to a
// paused conversation and takes them back after it leaves, copies in an expert a GPU evicts, and keeps an expert a
// step reads from the SSD in place of one used before the step's layer.
#include "strata/common/experts/ram_experts.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

constexpr uint64_t kPage = 16384;
/// Layers 0 and 2 have experts of one page, layer 1 of two.
const std::vector<uint64_t> kBlobs = {kPage, 2 * kPage, kPage};
constexpr int64_t kExperts = 8;

/// Gets the byte at `i` of expert (`layer`, `expert`).
uint8_t get_byte(int64_t layer, int64_t expert, uint64_t i) { return (uint8_t) (layer * 31 + expert * 7 + i); }

/// The experts' bytes as the SSD would give them, counting the reads.
struct Ssd : ExpertSource {
    int reads = 0;
    std::vector<uint8_t> buffer = std::vector<uint8_t>(2 * kPage);
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        (void) copy_blob(layer, expert, buffer.data());
        return buffer.data();
    }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override {
        ++reads;
        for (uint64_t i = 0; i < kBlobs[(size_t) layer]; ++i)
            dst[i] = get_byte(layer, expert, i);
        return true;
    }
};

/// Tells whether `bytes` are expert (`layer`, `expert`)'s.
bool is_expert(const uint8_t* bytes, int64_t layer, int64_t expert) {
    for (uint64_t i = 0; i < kBlobs[(size_t) layer]; ++i)
        if (bytes == nullptr || bytes[i] != get_byte(layer, expert, i))
            return false;
    return true;
}
}  // namespace

int main() {
    Ssd ssd;
    StateLayout layout;   // a conversation of a 3-page state and 1 page per 4 tokens
    layout.add(RegionKind::fixed, 3 * kPage);
    layout.add(RegionKind::per_block, kPage, 1, 4);
    std::vector<ExpertKey> ranking;
    for (int32_t e = 0; e < kExperts; ++e)
        for (int32_t l = 0; l < 3; ++l)
            ranking.push_back({l, e});
    PagePoolConfig config;
    config.page_bytes = kPage;
    config.hysteresis = 0;
    RamExperts ram;
    std::string err;
    // room for 12 pages: experts 0-2 of every layer (12 pages)
    check(ram.open(ssd, 12 * kPage + 2 * 2 * kPage, 2, layout, config, kBlobs, kExperts, ranking, err), err.c_str());
    check(ram.holds(1, 2) && !ram.holds(0, 3) && ram.pages().pool().report().experts.pages == 12,
          "the pool holds the ranking's first experts");
    check(is_expert(ram.blob(1, 2), 1, 2) && !ram.transient(1, 2), "an expert in the pool is its bytes there");
    const int filled = ssd.reads;
    check(is_expert(ram.blob(2, 5), 2, 5) && ram.transient(2, 5) && ssd.reads == filled + 1,
          "an expert outside the pool is read into the ring");
    const int64_t wanted[] = {3, 4, 5};
    ram.prefetch(0, wanted, 3);   // two buffers: the third read takes the first one again
    check(ssd.reads == filled + 4 && is_expert(ram.blob(0, 5), 0, 5) && ssd.reads == filled + 4,
          "a prefetched expert is not read again");
    check(is_expert(ram.blob(0, 3), 0, 3) && ssd.reads == filled + 5, "a buffer the ring took back is read again");
    std::vector<uint8_t> out(2 * kPage);
    check(ram.copy_blob(1, 0, out.data()) && is_expert(out.data(), 1, 0) && ram.copy_blob(1, 6, out.data()) &&
              is_expert(out.data(), 1, 6),
          "copy_blob gives the pool's and the SSD's experts");

    PagePool& pool = ram.pages().pool();
    check(pool.grow_conversation(9, 8).empty(), "a paused conversation takes pages");
    check(pool.report().experts.pages <= 12 - 5, "it took experts' pages");
    check(is_expert(ram.blob(2, 2), 2, 2) && is_expert(ram.blob(1, 7), 1, 7), "an evicted expert is read from the SSD");
    std::vector<uint8_t> gpu(kPage);
    for (uint64_t i = 0; i < kPage; ++i)
        gpu[i] = get_byte(0, 7, i);
    pool.remove_conversation(9);
    ram.keep_copy({0, 7}, gpu.data());   // a GPU's evicted expert, colder than those the conversation took
    check(ram.holds(0, 7) && is_expert(ram.blob(0, 7), 0, 7), "an expert a GPU evicts is copied in when there is room");
    uint64_t wait_ms = 0;
    check(ram.pages().is_refill_wanted() && ram.pages().refill(16, wait_ms, err) >= 1, "the evicted experts come back");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));   // the pool's clock counts milliseconds
    ram.begin_layer(1, nullptr, 0);
    check(!ram.holds(1, 5) && is_expert(ram.blob(1, 5), 1, 5) && ram.holds(1, 5),
          "an expert a step reads from the SSD takes the place of one used before the layer");
    for (int64_t l = 0; l < 3; ++l)
        for (int64_t e = 0; e < kExperts; ++e)
            check(!ram.holds(l, e) || is_expert(ram.blob(l, e), l, e), "every expert in the pool has its bytes");
    std::printf("ram_experts_test: %d checks passed\n", checks);
    return 0;
}

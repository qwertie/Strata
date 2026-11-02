// DeepSeek V4.1-Flash's conversation as page-pool regions (conversation_layout.hpp), on the geometry of the IST-DAS
// 2.00 bpw pack's ds4.txt: its KV sources, the regions of one card and of a layer split's cards, the bytes a
// conversation takes against the old engine's caches, its experts' page size, and a pool of those pages holding a
// conversation of 32K tokens with no page part empty.  No GPU, no model files.
#include "strata/common/state/page_pool.hpp"
#include "strata/models/deepseekv41/conversation_layout.hpp"
#include "strata/models/deepseekv41/kv_rows.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>

using namespace strata::ds4;
using strata::core::PagePool;
using strata::core::RegionKind;
using strata::core::StateLayout;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

/// The keys of the pack's ds4.txt that shape the state (deepseek-v4.1-flash-ist-das-2.00bpw).
constexpr const char* kDs4Txt = R"(num_hidden_layers 40
head_dim 512
sliding_window 128
index_head_dim 128
compress_ratios 0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0
kv_source_layer_ids 2,8,14,20
index_source_layer_ids 2,8,14,20,24,28,32,36
dspark_block_size 5
)";

/// One expert's bytes per layer (gate, up and down, each rounded up to 256 bytes) in that pack's tensors.txt: 8 sizes
/// from 3 IQ1_M matrices to 3 IQ2_XS ones, 123.0 GiB for the 40 x 384 experts.
const std::vector<uint64_t> kBlobBytes = {
    7741440, 7741440, 7741440, 7741440, 7741440, 7741440, 7741440, 8202240, 8202240, 7741440,
    9123840, 8663040, 8663040, 7741440, 7741440, 8663040, 7741440, 9400320, 9123840, 9400320,
    7741440, 7741440, 9123840, 9492480, 8663040, 9123840, 9492480, 9400320, 9123840, 8663040,
    7741440, 8663040, 8663040, 9492480, 9400320, 9400320, 8663040, 8663040, 9861120, 10229760};

/// A pack as Geometry::from reads it: the ds4.txt keys above, and the DSpark blocks' first tensor.
struct TextPack {
    std::map<std::string, std::string> kv;
    TextPack() {
        std::istringstream in(kDs4Txt);
        for (std::string key, value; in >> key >> value;)
            kv[key] = value;
    }
    double num(const std::string& k, double def) const {
        const auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v;
        std::istringstream in(kv.count(k) ? kv.at(k) : "");
        for (std::string x; std::getline(in, x, ',');)
            v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const void* get(const std::string& name) const { return name == "mtp.0.attn.wq_a.weight" ? this : nullptr; }
};

Geometry get_geometry() {
    Geometry g;
    g.from(TextPack());
    return g;
}

/// Gets the bytes per token a layout's growing regions take, measured over `tokens` tokens.
uint64_t get_bytes_per_token(const StateLayout& layout, int64_t tokens) {
    return (layout.bytes_for(tokens) - layout.bytes_for(0)) / (uint64_t) tokens;
}

void sources_fixture() {
    const Geometry g = get_geometry();
    check(g.nl == 40 && g.n_mtp == 3 && g.win == 128 && g.hd == 512 && g.ix_hd == 128, "the pack's geometry");
    check(kv_row_bytes(kKvFp8, g.hd) == 528 && kv_row_bytes(kKvFp4, g.hd) == 288 &&
              kv_row_bytes(kKvFp4p, g.ix_hd) == 80,
          "a ring row, a compressed row and an index key");
    std::vector<KvSource> sources;
    std::vector<int> source_of;
    std::string err;
    check(find_kv_sources(g, sources, source_of, err) && sources.size() == 4, "four KV sources");
    check(sources[0].layer == 2 && sources[0].ratio == 2 && sources[2].layer == 14 && sources[3].layer == 20 &&
              sources[3].ratio == 1,
          "the sources' layers and ratios");
    check(source_of[0] == -1 && source_of[1] == -1 && source_of[2] == 0 && source_of[7] == 0 && source_of[8] == 1 &&
              source_of[19] == 2 && source_of[20] == 3 && source_of[39] == 3,
          "each layer reads the last source at or before it");

    Geometry bad = g;
    bad.ratios[9] = 1;   // a layer of ratio 1 after a source of ratio 2
    check(!find_kv_sources(bad, sources, source_of, err) && err.find("layer 9") == 0, "a ratio without its source");
}

void one_card_fixture() {
    const Geometry g = get_geometry();
    StateLayout layout;
    ConversationRegions at;
    std::string err;
    check(describe_conversation(g, 0, g.nl, true, layout, at, err), "one card describes the conversation");
    check(layout.regions.size() == 10 && at.trunk == 0 && at.drafter == 1, "two fixed regions, then 4 x 2 growing");
    const uint64_t ring = 128 * 528, group = 2 * 2 * 512 * 4;
    check(layout.regions[at.trunk].unit_bytes == 40 * ring + 3 * group, "the trunk's rings and partial groups");
    check(layout.regions[at.drafter].unit_bytes == 3 * ring, "the DSpark blocks' rings");
    check(at.ring[39] == 39 * ring && at.ring[40] == 0 && at.ring[42] == 2 * ring, "where the rings start");
    check(at.group[0] == 40 * ring && at.group[2] == 40 * ring + 2 * group && at.group[3] == kNoOffset,
          "the partial groups after the rings; none at ratio 1");
    for (size_t s = 0; s < 4; ++s) {
        const auto& ckv = layout.regions[at.ckv[s]];
        const auto& ik = layout.regions[at.ik[s]];
        check(ckv.kind == RegionKind::per_block && ckv.unit_bytes == 288 && ik.unit_bytes == 80 &&
                  ckv.block == (s < 3 ? 2 : 1) && ik.block == ckv.block,
              "a source's rows: one a group of `ratio` tokens");
    }
    // 3 sources at ratio 2 ((288 + 80) / 2 each) and one at ratio 1 (288 + 80): 920 bytes a token
    check(get_bytes_per_token(layout, 32768) == 920, "920 bytes a token");
    // the old engine's caches at a context of 32768: `cap` = 32768 / ratio + 1 rows a source, one row more than the
    // pool's 32768 / ratio
    uint64_t old_rows = 0;
    for (int ratio : {2, 2, 2, 1})
        old_rows += (32768 / (uint64_t) ratio + 1) * (288 + 80);
    check(layout.bytes_for(32768) - layout.bytes_for(0) == old_rows - 4 * (288 + 80),
          "the growing regions hold what the old engine's caches do, but its spare row");

    check(describe_conversation(g, 0, g.nl, false, layout, at, err) && at.drafter == kNoRegion &&
              layout.regions.size() == 9,
          "without the drafter, no DSpark rings");
    check(!describe_conversation(g, 0, 41, false, layout, at, err), "a range past the trunk is refused");
}

void split_fixture() {
    const Geometry g = get_geometry();
    StateLayout a, b, whole;
    ConversationRegions at_a, at_b, at;
    std::string err;
    check(describe_conversation(g, 0, g.nl, false, whole, at, err), "one card");
    // a cut at 20: no source is read across it, so the cards' bytes add up to one card's
    check(describe_conversation(g, 0, 20, false, a, at_a, err) &&
              describe_conversation(g, 20, g.nl, false, b, at_b, err),
          "a cut at layer 20");
    check(at_a.ckv[3] == kNoRegion && at_b.ckv[0] == kNoRegion && at_b.group[0] == kNoOffset && at_b.ring[0] ==
              kNoOffset && at_b.ring[20] == 0,
          "each card keeps its own layers' state");
    check(a.bytes_for(10000) + b.bytes_for(10000) == whole.bytes_for(10000), "the two cards hold one card's state");
    // a cut at 10: layers 10 to 13 read source 1 (layer 8), so its compressed rows are on both cards; no layer of the
    // second card scores its index keys, and its partial group stays with layer 8
    check(describe_conversation(g, 10, g.nl, false, b, at_b, err), "a cut at layer 10");
    check(at_b.ckv[1] != kNoRegion && at_b.ik[1] == kNoRegion && at_b.group[1] == kNoOffset && at_b.ckv[0] ==
              kNoRegion,
          "a source read across the cut is on both cards");
}

void pages_fixture() {
    const Geometry g = get_geometry();
    const uint64_t page = strata::core::choose_expert_page_bytes(kBlobBytes);
    check(page == 92160, "the experts' greatest common divisor: pages of 92,160 bytes");
    uint64_t pages = 0, smallest = UINT64_MAX, largest = 0;
    for (const uint64_t b : kBlobBytes) {
        check(b % page == 0, "every expert is a whole number of pages");
        smallest = std::min(smallest, b / page);
        largest = std::max(largest, b / page);
        pages += b / page * 384;
    }
    check(smallest == 84 && largest == 111 && pages * page == 132073390080ull, "84 to 111 pages an expert, 123 GiB");

    StateLayout layout;
    ConversationRegions at;
    std::string err;
    check(describe_conversation(g, 0, g.nl, true, layout, at, err), "one card with the drafter");
    for (size_t s = 0; s < 4; ++s) {
        const auto& ckv = layout.regions[at.ckv[s]];
        const auto& ik = layout.regions[at.ik[s]];
        check(strata::core::choose_block_pages(ckv, page) == 1 && strata::core::choose_block_pages(ik, page) == 1,
              "a block of one page");
        check(ckv.tokens_in(page) == 320 * ckv.block && ik.tokens_in(page) == 1152 * ik.block,
              "a page holds 320 compressed rows or 1,152 index keys, none part empty");
    }

    // a pool of those pages (chunks of 128 pages on host memory) and a conversation of 32K tokens in it
    strata::core::PagePoolConfig config;
    config.page_bytes = page;
    config.chunk_bytes = 128 * page;
    strata::core::PagePoolHooks hooks;
    hooks.alloc = [](size_t bytes) { return std::calloc(1, bytes); };
    hooks.free = [](void* p) { std::free(p); };
    PagePool pool(layout, config, hooks);
    check(pool.resize(4 * config.chunk_bytes).refusal.empty(), "the pool takes 4 chunks");
    check(pool.grow_conversation(1, 32768).empty(), "a conversation of 32K tokens fits");
    // the fixed regions: 2,727,936 bytes (30 pages) and 202,752 (3); a source at ratio 2: 16,384 rows (52 pages of
    // compressed rows, 15 of index keys); at ratio 1: 32,768 (103 and 29)
    const uint64_t expected = 30 + 3 + 3 * (52 + 15) + 103 + 29;
    check(pool.pages_for(32768) == expected && pool.report().active.pages + pool.report().fixed.pages == expected,
          "the conversation takes 366 pages (33.7 MB)");
    check(pool.conversation_blocks(1, at.ckv[3]).size() == 103, "one block a page");
}

/// Checks every cut against one canonical copy, including sources whose rows appear on both stages.
void copy_split_fixture() {
    const Geometry g = get_geometry();
    const uint64_t page = strata::core::choose_expert_page_bytes(kBlobBytes);
    StateLayout whole;
    ConversationRegions at;
    std::string err;
    check(describe_conversation(g, 0, g.nl, true, whole, at, err), "the canonical copy includes DSpark");
    const auto make_pool = [&](const StateLayout& layout) {
        strata::core::PagePoolConfig config;
        config.page_bytes = page;
        config.chunk_bytes = 128 * page;
        strata::core::PagePoolHooks hooks;
        hooks.alloc = [](size_t bytes) { return std::calloc(1, bytes); };
        hooks.free = [](void* p) { std::free(p); };
        auto pool = std::make_unique<PagePool>(layout, config, hooks);
        check(pool->resize(config.chunk_bytes).refusal.empty() && pool->grow_conversation(1, 512).empty(),
              "the test conversation fits");
        return pool;
    };
    auto source = make_pool(whole);
    for (size_t r = 0; r < whole.regions.size(); ++r)
        for (const auto block : source->conversation_blocks(1, r))
            for (uint64_t i = 0; i < source->block_bytes(r); ++i)
                source->page_address(block)[i] = (uint8_t) (r * 17 + i % 251);
    for (int cut = 1; cut < g.nl; ++cut) {
        auto restored = make_pool(whole);
        for (int stage = 0; stage < 2; ++stage) {
            const int lb = stage == 0 ? 0 : cut, le = stage == 0 ? cut : g.nl;
            StateLayout layout;
            check(describe_conversation(g, lb, le, stage == 1, layout, at, err), "the stage has a layout");
            auto pool = make_pool(layout);
            const auto local = get_conversation_span(g, lb, le, stage == 1, false);
            const auto canonical = get_conversation_span(g, lb, le, stage == 1, true);
            check(strata::core::copy_conversation(*source, 1, *pool, 1, err, canonical, local) &&
                      strata::core::copy_conversation(*pool, 1, *restored, 1, err, local, canonical),
                  "a stage's packed state copies through the canonical layout");
        }
        for (size_t r = 0; r < whole.regions.size(); ++r) {
            const uint64_t bytes = whole.regions[r].kind == RegionKind::fixed ? whole.regions[r].unit_bytes
                                                                            : restored->block_bytes(r);
            for (const auto block : restored->conversation_blocks(1, r)) {
                const uint8_t* data = restored->page_address(block);
                bool same = true;
                for (uint64_t i = 0; same && i < bytes; ++i)
                    same = data[i] == (uint8_t) (r * 17 + i % 251);
                check(same, "every ring, compressor group, drafter ring and compressed row survives every cut");
            }
        }
    }
}

}  // namespace

int main() {
    sources_fixture();
    one_card_fixture();
    split_fixture();
    pages_fixture();
    copy_split_fixture();
    std::printf("conversation_layout_test: %d checks passed\n", checks);
    return 0;
}

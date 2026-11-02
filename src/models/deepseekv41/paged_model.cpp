// src/models/deepseekv41/paged_model.cpp - strata-ds41: DeepSeek V4.1 on the shared paged engine (paged_model.hpp):
// its start and its side of the engine's model interface (strata/common/serve/paged_model.hpp).
#include "strata/models/deepseekv41/paged_model.hpp"

#include "ds4_model.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/os.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/serve/paged_engine.hpp"
#include "strata/common/serve/paged_model.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4 {
namespace {

using strata::serve::PagedCard;
using strata::serve::PagedCards;
using strata::serve::PagedParts;
using strata::serve::PagedSettings;

/// The program's name, which starts its stderr lines.
const std::string kProgram = "strata-ds41";
/// The RAM the RAM pool leaves free at its largest (the old engine's kRamMargin).
constexpr uint64_t kRamHeadroom = 8ull << 30;
/// The top_k of a request that names none (the old engine's).
constexpr int kDefaultTopK = 64;
/// The checkpoint the prompt's read keeps the state before its last token in (Model::feed_mark), as the old engine's
/// does.
constexpr int kBeforeLastCheck = 1;

/// The experts' source of the RAM's pool, which holds none of DeepSeek's experts before step D5.
class NoExperts final : public strata::core::ExpertSource {
public:
    const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
};

/// The conversation's state on the one card: its regions in the card's pool (describe_conversation's), the tables of
/// its growing regions' blocks that the kernels read, and the model pointed at them.
struct Part final : strata::serve::PagedPart {
    Model& m;
    const ConversationRegions& at;
    const int dev;                              ///< the card's device
    strata::core::PagePool* pool = nullptr;     ///< null before init
    strata::core::ConversationId cid = 0;
    /// per region of the layout (empty for a fixed one): the first address of each of its blocks, on the host, and
    /// the device copy the kernels read; sized once for the whole context, so that KvRows may point at them
    std::vector<std::vector<uint8_t*>> blocks;
    std::vector<uint8_t**> tables;
    std::vector<size_t> zeroed;                 ///< per region: its blocks that are zeroed and in the table

    Part(Model& model, const ConversationRegions& regions, int device) : m(model), at(regions), dev(device) {}
    ~Part() override {
        const strata::core::OnDevice on(dev);
        for (uint8_t** t : tables)
            strata::core::gpu_free(t);
    }

    bool is_ready() const override { return pool != nullptr; }
    bool init(strata::core::PagePool& p, strata::core::ConversationId id, std::string& err) override {
        if (const std::string why = p.grow_conversation(id, 0); !why.empty()) {
            err = "the page pool: " + why;
            return false;
        }
        pool = &p;
        cid = id;
        return alloc_tables(err) && write_tables(err);
    }
    bool reserve(int64_t tokens, std::string& err) override {
        if (const std::string why = pool->grow_conversation(cid, tokens); !why.empty()) {
            err = "the conversation's memory is full: " + why;
            return false;
        }
        return write_tables(err);
    }
    void truncate(int64_t tokens) override {
        pool->truncate_conversation(cid, tokens);
        for (size_t r = 0; r < blocks.size(); ++r)
            zeroed[r] = std::min(zeroed[r], pool->conversation_blocks(cid, r).size());
    }
    void release() override {
        pool->remove_conversation(cid);
        pool = nullptr;
        std::fill(zeroed.begin(), zeroed.end(), 0);
    }
    bool refresh(std::string& err) override { return write_tables(err, false); }
    void clear_state() override { m.reset(); }
    void set_layers(int64_t, int64_t) override {}   // (one card: the cut never moves)
    uint64_t bytes() const override {
        uint64_t bytes = 0;
        for (size_t r = 0; is_ready() && r < blocks.size(); ++r)
            bytes += pool->conversation_blocks(cid, r).size() * pool->block_bytes(r);
        return bytes;
    }
    strata::core::ConversationId id() const override { return cid; }
    int64_t get_pool_tokens(int64_t tokens) const override { return tokens; }

    /// Sizes the host and device tables of the growing regions for a conversation of the whole context.
    bool alloc_tables(std::string& err) {
        if (!tables.empty())
            return true;
        const strata::core::StateLayout& layout = pool->layout();
        const strata::core::OnDevice on(dev);
        blocks.resize(layout.regions.size());
        tables.assign(layout.regions.size(), nullptr);
        zeroed.assign(layout.regions.size(), 0);
        for (size_t r = 0; r < layout.regions.size(); ++r) {
            const strata::core::StateRegion& region = layout.regions[r];
            if (region.kind == strata::core::RegionKind::fixed)
                continue;
            const uint64_t rows = region.units_for(m.max_context()), per_block = get_rows_per_block(r);
            const size_t most = (size_t) ((rows + per_block - 1) / per_block + 1);
            blocks[r].reserve(most);
            tables[r] = (uint8_t**) strata::core::gpu_alloc_zeroed(most * sizeof(uint8_t*));
            if (tables[r] == nullptr) {
                err = "no room for the conversation's block tables";
                return false;
            }
        }
        return true;
    }

    /// Writes the blocks the pool gives the conversation into the tables, zeroing the new ones first with
    /// `zero_new` (their pages held other data), and points the model at its fixed region and its rows.
    bool write_tables(std::string& err, bool zero_new = true) {
        const strata::core::OnDevice on(dev);
        const strata::core::StateLayout& layout = pool->layout();
        std::vector<KvRows> ckv(at.ckv.size()), ik(at.ik.size());
        for (size_t r = 0; r < layout.regions.size(); ++r) {
            if (tables[r] == nullptr)
                continue;
            const std::vector<strata::core::PageId>& runs = pool->conversation_blocks(cid, r);
            blocks[r].clear();
            for (const strata::core::PageId page : runs)
                blocks[r].push_back(pool->page_address(page));
            for (size_t i = zeroed[r]; zero_new && i < blocks[r].size(); ++i)
                if (!strata::core::gpu_memset_async(blocks[r][i], 0, pool->block_bytes(r), nullptr))
                    return fail("zeroing the conversation's new pages", err);
            zeroed[r] = blocks[r].size();
            if (!blocks[r].empty() &&
                !strata::core::gpu_copy(tables[r], blocks[r].data(), blocks[r].size() * sizeof(uint8_t*)))
                return fail("writing the conversation's block tables", err);
        }
        for (size_t s = 0; s < at.ckv.size(); ++s) {
            ckv[s] = get_rows(at.ckv[s]);
            ik[s] = get_rows(at.ik[s]);
        }
        uint8_t* fixed = at.trunk != kNoRegion ? pool->page_address(pool->conversation_blocks(cid, at.trunk).at(0))
                                               : nullptr;
        m.use_state_memory(0, fixed, at, ckv, ik);
        return true;
    }

    /// Gets how region `r`'s rows lie in its blocks; none for kNoRegion.
    KvRows get_rows(size_t r) const {
        KvRows rows;
        if (r == kNoRegion)
            return rows;
        rows.blocks = tables[r];
        rows.host_blocks = blocks[r].data();
        rows.per_block = (uint32_t) get_rows_per_block(r);
        rows.bytes = (uint32_t) pool->layout().regions[r].unit_bytes;
        return rows;
    }
    uint64_t get_rows_per_block(size_t r) const { return pool->block_bytes(r) / pool->layout().regions[r].unit_bytes; }
    static bool fail(const char* what, std::string& err) {
        err = std::string(what) + " failed";
        return false;
    }
};

/// DeepSeek for the paged engine: one conversation (the Model's own state, its memory the Part's pages), one row a
/// step.  Pausing, the experts in the pools, the drafter and a layer split are the plan's steps D5 to D9; until then
/// their calls refuse or do nothing.
class PagedDeepSeek final : public strata::serve::PagedModel {
public:
    PagedDeepSeek(Model& m, const ConversationRegions& at, const Options& o, const std::vector<int>& eos);

    PagedParts make_parts(PagedCard& card, size_t count, std::string& err) override;
    bool get_card_layout(const PagedCard&, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                         std::string& err) const override {
        ConversationRegions at;
        return describe_conversation(m_.geometry(), (int) lb, (int) le, false, layout, at, err);
    }
    uint64_t get_gpu_keep(const PagedCard&) const override { return (uint64_t) o_.reserve_mib << 20; }
    std::vector<strata::core::ExpertKey> get_card_ranking(int64_t, int64_t, int64_t& max_experts) const override {
        max_experts = 0;
        return {};
    }
    bool load_expert(const PagedCard&, strata::core::ExpertKey, int32_t, std::string& err) override {
        err = "DeepSeek's experts come into the pools in step D5";
        return false;
    }
    std::vector<int32_t>& get_residency() override { return residency_; }
    bool publish_residency(const PagedCard&, std::string&) override { return true; }   // (no expert in the pools)
    std::vector<float>& get_usage() override { return usage_; }
    int get_adapt_interval(int64_t) const override { return 1; }
    bool set_card_layers(const PagedCard&, std::string& err) override {
        err = "strata-ds41 runs on one card";
        return false;
    }
    strata::core::LayerSizes get_layer_sizes(uint64_t page_bytes) const override;
    bool can_run_layers(int64_t lb, int64_t le) const override { return lb == 0 && le == m_.geometry().nl; }
    strata::core::RegionSpan get_gpu_span(const PagedCard&) const override { return {}; }
    strata::core::RegionSpan get_ram_span(int64_t, int64_t) const override { return {}; }
    uint64_t get_weight_bytes(const PagedCard&) const override { return 0; }
    uint64_t get_scratch_bytes(const PagedCard&) const override { return 0; }
    bool attach_hop(PagedCard&, std::string& err) override {
        err = "strata-ds41 runs on one card";
        return false;
    }
    void set_hop_holders(const int32_t*) override {}
    void report_hops() const override {}
    void set_sampling(int, const strata::serve::Sampling& smp) override { smp_ = smp; }
    bool save_state(const PagedParts&, std::vector<uint8_t>&, std::string& err) const override {
        err = "pausing a DeepSeek conversation comes in step D6";
        return false;
    }
    bool restore_state(const std::vector<uint8_t>&, const PagedParts&, const std::vector<int32_t>&,
                       std::string& err) const override {
        err = "pausing a DeepSeek conversation comes in step D6";
        return false;
    }
    void forget_drafts(int) override {}
    int64_t get_prompt_chunk() const override { return m_.feed_unit(); }
    bool read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n, bool drafting,
                     const ChunkDone& on_chunk, std::string& err) override;
    bool run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                  std::string& err) override;
    bool keep_rows(const std::vector<strata::spec::StepRow>&, const std::vector<int>&, bool, std::string&) override {
        return true;   // (one row a step, which the model has fed)
    }
    int propose(strata::serve::PagedPart&, int, int, int32_t*, std::string&) override { return 0; }
    bool switch_drafter(bool, std::string& err) override {
        err = "DeepSeek's drafter comes in step D7";
        return false;
    }
    void print_info(int, int) const override {
        for (const std::string& line : m_.info_lines())
            std::printf("%s\n", line.c_str());
    }
    void print_done(const strata::serve::RequestDone& d) const override {
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld 0 0\n", (long long) d.produced, (long long) d.prompt,
                    d.prompt_ms, d.decode_ms, d.finish, (long long) d.guessed_right, (long long) d.guessed,
                    (long long) d.resumed);
    }

private:
    bool go_to(int64_t pos, std::string& err);

    Model& m_;
    const ConversationRegions& at_;
    const Options& o_;
    std::vector<int32_t> residency_;   ///< every expert kNotResident: none is in the pools before step D5
    std::vector<float> usage_;         ///< empty: the engine's adaptive tier is off (the model has its own)
    strata::serve::Sampling smp_;
    uint64_t counter_ = 0;             ///< the sampler's draw count, which the old engine keeps across requests
    /// the prompt's last token when the prompt's read took it along (feed_mark): its position (-1: none) and the
    /// token, whose logits the next step samples instead of feeding it again
    int64_t primed_pos_ = -1;
    int32_t primed_token_ = 0;
};

int read_model_options(int argc, char** argv, Options& o, std::string& eos);
bool open_card_pool(const Model& m, const PagedSettings& s, const strata::core::StateLayout& layout, uint64_t keep,
                    PagedCard& card, std::string& err);

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // each protocol line reaches the server as it is printed
    strata::platform::fine_timer();
    PagedSettings settings;
    if (const int status = strata::serve::take_paged_options(argc, argv, kProgram, settings); status >= 0)
        return status;
    Options o;
    std::string eos = "1";
    if (const int status = read_model_options(argc, argv, o, eos); status >= 0)
        return status;
    if (settings.step.lookahead > 0) {
        std::fprintf(stderr, "%s: --lookahead: the drafter comes in step D7\n", kProgram.c_str());
        return 2;
    }
    settings.step.concurrent = settings.step.rows = 1;   // one conversation, one row a step
    settings.drafter = false;
    strata::serve::set_paged_runtime_env(settings, false);
    Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    const std::vector<int> devices = m.get_devices();
    if (devices.size() != 1) {
        std::fprintf(stderr, "%s: runs on one GPU (--gpus N) until step D9\n", kProgram.c_str());
        return 2;
    }
    const Geometry& g = m.geometry();
    PagedCards cards;
    PagedCard& card = cards.emplace_back();
    card.dev = devices[0];
    card.le = card.weights_le = g.nl;
    card.id = "gpu" + std::to_string(card.dev);
    const std::vector<uint64_t> blobs = m.get_expert_bytes();
    strata::serve::measure_card(card, kProgram, *std::max_element(blobs.begin(), blobs.end()));
    strata::core::StateLayout layout;
    ConversationRegions at;
    if (!describe_conversation(g, 0, g.nl, false, layout, at, err) ||
        !open_card_pool(m, settings, layout, (uint64_t) o.reserve_mib << 20, card, err) || !m.place_experts(err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    // the RAM's pool: paused conversations only, before step D5 puts the experts there too
    NoExperts no_experts;
    strata::core::RamExperts ram;
    const uint64_t ram_bytes = strata::serve::get_ram_pool_bytes(
        settings, kRamHeadroom, layout.bytes_for(m.max_context()) + *std::max_element(blobs.begin(), blobs.end()));
    if (!ram.open(no_experts, ram_bytes, 1, layout, strata::serve::get_ram_pool_config(card), blobs, g.n_exp, {},
                  err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    PagedDeepSeek model(m, at, o, parse_ids(eos));
    return strata::serve::serve_paged_model(model, cards, ram, false, settings, 0);
}

namespace {

PagedDeepSeek::PagedDeepSeek(Model& m, const ConversationRegions& at, const Options& o, const std::vector<int>& eos)
    : m_(m), at_(at), o_(o) {
    const Geometry& g = m.geometry();
    facts.name = kProgram;
    facts.n_layers = g.nl;
    facts.n_expert = g.n_exp;
    facts.context = o.max_context;
    facts.max_rows = 1;
    facts.end_tokens.assign(eos.begin(), eos.end());
    facts.sampling.top_k = kDefaultTopK;
    facts.expert_bytes = m.get_expert_bytes();
    facts.ram_headroom = kRamHeadroom;
    residency_.assign((size_t) g.nl * (size_t) g.n_exp, strata::core::kNotResident);
}

PagedParts PagedDeepSeek::make_parts(PagedCard& card, size_t count, std::string& err) {
    if (count != 1) {
        err = "strata-ds41 keeps one conversation (no --batch before step D8)";
        return {};
    }
    PagedParts parts;
    parts.push_back(std::make_unique<Part>(m_, at_, card.dev));
    return parts;
}

strata::core::LayerSizes PagedDeepSeek::get_layer_sizes(uint64_t page) const {
    const Geometry& g = m_.geometry();
    const std::vector<uint64_t> experts = m_.get_expert_bytes();
    std::vector<KvSource> sources;
    std::vector<int> source_of;
    std::string err;
    (void) find_kv_sources(g, sources, source_of, err);
    strata::core::LayerSizes sizes;
    for (int il = 0; il < g.nl; ++il) {
        sizes.dense.push_back(0);   // (outside the pools)
        sizes.expert.push_back((experts[(size_t) il] + page - 1) / page * page);
        sizes.fixed_state.push_back((uint64_t) g.win * kv_row_bytes(kKvFp8, g.hd));
        sizes.token_bytes.push_back(0);
    }
    for (const KvSource& s : sources) {   // a source's rows, counted at its layer
        sizes.fixed_state[(size_t) s.layer] += s.ratio > 1 ? 2 * (uint64_t) s.ratio * g.hd * sizeof(float) : 0;
        sizes.token_bytes[(size_t) s.layer] = (kv_row_bytes(kKvFp4, g.hd) + kv_row_bytes(kKvFp4p, g.ix_hd)) / s.ratio;
    }
    return sizes;
}

bool PagedDeepSeek::read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n,
                                bool, const ChunkDone& on_chunk, std::string& err) {
    if (!go_to(from, err))
        return false;
    // the prompt's last token rides along with the rest, as the old engine reads it (Model::feed_mark: one run of
    // chunks, which gives other numbers than a window of its own), and its logits wait for the step that would feed it
    const int64_t end = from + n;
    const bool last = end + 1 == (int64_t) ids.size();
    const std::vector<int> tokens(ids.begin() + from, ids.begin() + end + (last ? 1 : 0));
    if (last && !parts[0]->reserve(end + 1, err))
        return false;
    if (!(last ? m_.feed_mark(tokens.data(), (int) tokens.size(), 1, kBeforeLastCheck, err)
               : m_.feed(tokens.data(), (int) tokens.size(), 0, err)))
        return false;
    primed_pos_ = last ? end : -1;
    primed_token_ = tokens.back();
    return on_chunk(from, n, err);
}

bool PagedDeepSeek::run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                             std::string& err) {
    if (rows.size() != 1) {
        err = "strata-ds41 runs one row a step (no drafting before step D7)";
        return false;
    }
    const strata::spec::StepRow& row = rows[0];
    // (a request that ended before its first step leaves a primed token that the next request may not have)
    const bool primed = primed_pos_ == row.pos && primed_token_ == row.token && m_.pos() == row.pos + 1;
    primed_pos_ = -1;
    if (!primed && !(go_to(row.pos, err) && m_.step(row.token, true, err)))
        return false;
    picks[0] = m_.sample(smp_.temperature, smp_.top_p, smp_.top_k, smp_.min_p, smp_.seed, counter_++);
    return true;
}

/// Makes the model's state the conversation's at `pos` tokens: it is there, or one past it when the prompt's read
/// took the next token along and no step fed it (the request ended first), and goes back to the checkpoint before it.
bool PagedDeepSeek::go_to(int64_t pos, std::string& err) {
    if (m_.pos() == pos)
        return true;
    if (m_.pos() == pos + 1 && m_.restore(pos) && m_.pos() == pos)
        return true;
    err = "the model's state is at token " + std::to_string(m_.pos()) + ", the conversation's at " +
          std::to_string(pos);
    return false;
}

/// Reads the model's own options and --eos from `argv` (the paged engine's are gone) into `o` and `eos`.  -1: go on,
/// else the exit status.
int read_model_options(int argc, char** argv, Options& o, std::string& eos) {
    bool drafting = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: %s needs a value\n", kProgram.c_str(), what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (read_option(a, val, o, drafting))
            continue;
        if (a == "--eos")
            eos = val("--eos");
        else if (a == "--layer-split")
            (void) val("--layer-split");   // the server's (one card here)
        else if (a != "--serve") {
            std::fprintf(stderr, "%s: unknown option %s\n", kProgram.c_str(), a.c_str());
            return 2;
        }
    }
    if (o.pack.empty()) {
        std::fprintf(stderr, "%s: --pack <model>/maya-ds4 is needed\n", kProgram.c_str());
        return 2;
    }
    if (o.profile.empty())
        o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none")
        o.profile.clear();
    o.dspark = false;   // (the drafter comes in step D7)
    return -1;
}

/// Opens `card`'s page pool for conversations of `layout` (no experts before step D5) before the experts take the
/// GPU memory left: --gpu-mib, by default room for one conversation of the whole context, leaving `keep` free.
bool open_card_pool(const Model& m, const PagedSettings& s, const strata::core::StateLayout& layout, uint64_t keep,
                    PagedCard& card, std::string& err) {
    const std::vector<uint64_t> blobs = m.get_expert_bytes();
    const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
    uint64_t room = 0;
    const uint64_t bytes =
        strata::serve::get_gpu_pool_bytes(card.dev, keep, s.gpu_mib.empty() ? -1 : s.gpu_mib[0],
                                          strata::serve::get_conversation_bytes(layout, m.max_context(), page), room);
    strata::core::PagePoolConfig config;
    config.page_bytes = page;
    config.chunk_bytes = strata::serve::get_chunk_bytes(bytes, page);
    config.tokens_min = std::clamp<int64_t>(s.tokens_min, 0, m.max_context());
    const strata::core::OnDevice on(card.dev);
    if (!card.experts.open(strata::serve::get_gpu_chunk_memory(card.dev), bytes, layout, config, blobs,
                           m.geometry().n_exp, 0, {}, err))
        return false;
    strata::serve::say_gpu_pool(kProgram, card, room, config.chunk_bytes);
    return true;
}

}  // namespace

}  // namespace strata::ds4

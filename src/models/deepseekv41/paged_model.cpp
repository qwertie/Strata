// src/models/deepseekv41/paged_model.cpp - strata-ds41: DeepSeek V4.1 on the shared paged engine (paged_model.hpp):
// its start and its side of the engine's model interface (strata/common/serve/paged_model.hpp).
#include "strata/models/deepseekv41/paged_model.hpp"

#include "ds4_model.hpp"
#include "expert_files.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/os.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/serve/paged_engine.hpp"
#include "strata/common/serve/paged_model.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
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
/// The checkpoints (Model::checkpoint's slots) a prompt's read keeps, as the old engine's does: the state at its end,
/// and before its last token (Model::feed_mark), which Model::pin makes the anchor.
constexpr int kPromptEndCheck = 0;
constexpr int kBeforeLastCheck = 1;

/// strata-ds41's options beside the paged engine's and the model's: the batch slots, the GPU pool's experts and the
/// adaptive tier (as strata-q38's options of the same names).
struct ProgramOptions {
    int batch = 0;              ///< --batch: the conversations held beside the one a GEN uses (one decodes at a time)
    int64_t gpu_experts = -1;   ///< --expert-cache: the most experts the GPU's pool holds; -1 (auto): as many as fit
    int adapt_every = 4;        ///< --adapt-every: decode steps between two rounds of the tier
    /// --adapt-swaps: the most experts a round moves into the GPU's pool (0: the tier is off).  The old engine's expert
    /// cache has at most 8 moves under way (kMaxSwaps); a DeepSeek expert is 4 to 5 times a Qwen3.8 one, whose 96 a
    /// round M5b measured
    int adapt_swaps = 8;
    float adapt_min_gain = 1.5f;   ///< --adapt-min-gain: the routings a move must gain (choose_expert_swaps)
    float adapt_decay = 0.7f;      ///< --adapt-decay: the routing counts' factor after each round
};

/// A conversation's state on the one card: its regions in the card's pool (describe_conversation's) and the tables of
/// its growing regions' blocks that the kernels read; its host side (tokens, checkpoints), which is in the model while
/// the model computes this conversation (it is seated); and what its request needs of the model.
struct Part final : strata::serve::PagedPart {
    Model& m;
    const ConversationRegions& at;
    const int dev;                              ///< the card's device
    Part*& seated;                              ///< the part the model computes (PagedDeepSeek::seated_), or null
    strata::core::PagePool* pool = nullptr;     ///< null before init
    strata::core::ConversationId cid = 0;
    /// per region of the layout (empty for a fixed one): the first address of each of its blocks, on the host, and
    /// the device copy the kernels read; sized once for the whole context, so that KvRows may point at them
    std::vector<std::vector<uint8_t*>> blocks;
    std::vector<uint8_t**> tables;
    std::vector<size_t> zeroed;                 ///< per region: its blocks that are zeroed and in the table
    ConversationState state;                    ///< its host side while it is not seated (take_seat)
    strata::serve::Sampling smp;                ///< its request's sampling
    /// the prompt's last token when the prompt's read took it along (feed_mark): its position (-1: none), the token
    /// and the token sampled after it, which the step that would feed it gives instead (another conversation's steps
    /// may have taken the logits since)
    int64_t primed_pos = -1;
    int32_t primed_token = 0, primed_pick = 0;
    bool prompt_open = false;   ///< its request's prompt is not read to its end, where it keeps two checkpoints
    /// the end of its next prompt becomes its anchor (Model::pin): its conversation started over or went back
    bool pin_next = false;

    Part(Model& model, const ConversationRegions& regions, int device, Part*& seated_part)
        : m(model), at(regions), dev(device), seated(seated_part) {}
    ~Part() override {
        if (seated == this)
            seated = nullptr;
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
        leave_seat();
        state = {};
        primed_pos = -1;
        pool->remove_conversation(cid);
        pool = nullptr;
        std::fill(zeroed.begin(), zeroed.end(), 0);
    }
    bool refresh(std::string& err) override { return write_tables(err, false); }
    void clear_state() override {
        take_seat();
        m.reset();
        primed_pos = -1;
        pin_next = true;
    }
    void set_layers(int64_t, int64_t) override {}   // (one card: the cut never moves)
    uint64_t bytes() const override {
        uint64_t bytes = 0;
        for (size_t r = 0; is_ready() && r < blocks.size(); ++r)
            bytes += pool->conversation_blocks(cid, r).size() * pool->block_bytes(r);
        return bytes;
    }
    strata::core::ConversationId id() const override { return cid; }
    int64_t get_pool_tokens(int64_t tokens) const override { return tokens; }

    /// Makes the model compute this conversation: the seated one's host side goes back to it, this one's into the
    /// model, and the model reads this one's pages.
    void take_seat() {
        if (seated == this)
            return;
        if (seated != nullptr)
            seated->leave_seat();
        m.swap_conversation(state);
        seated = this;
        point_model();
    }
    /// Takes its host side out of the model when it is seated.
    void leave_seat() {
        if (seated != this)
            return;
        m.swap_conversation(state);
        seated = nullptr;
    }
    /// Gets the tokens its state holds.
    int64_t get_pos() const { return seated == this ? m.pos() : (int64_t) state.tokens.size(); }

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
    /// `zero_new` (their pages held other data), and points the model at them when it is seated.
    bool write_tables(std::string& err, bool zero_new = true) {
        const strata::core::OnDevice on(dev);
        const strata::core::StateLayout& layout = pool->layout();
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
        if (seated == this)
            point_model();
        return true;
    }

    /// Points the model at its fixed region and its rows.
    void point_model() {
        std::vector<KvRows> ckv(at.ckv.size()), ik(at.ik.size());
        for (size_t s = 0; s < at.ckv.size(); ++s) {
            ckv[s] = get_rows(at.ckv[s]);
            ik[s] = get_rows(at.ik[s]);
        }
        uint8_t* fixed = at.trunk != kNoRegion ? pool->page_address(pool->conversation_blocks(cid, at.trunk).at(0))
                                               : nullptr;
        m.use_state_memory(0, fixed, at, ckv, ik);
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

/// DeepSeek for the paged engine: the conversations of the batch slots and "main", which the Model computes one at a
/// time (the seated Part's: its memory the Part's pages), one row a step; the GPU computes the experts of the card's
/// pool, the CPU the others from the RAM's pool (Options::expert_pages).  A paused conversation's checkpoints are its
/// state beside the pages (save_state), and a prompt that leaves a conversation before its end goes back to one of
/// them (rewind).  The drafter, several conversations in a step and a layer split are the plan's steps D7 to D9; until
/// then their calls refuse or do nothing.
class PagedDeepSeek final : public strata::serve::PagedModel {
public:
    /// `files`: the experts on the SSD; `ram`: the RAM's pool, which the model reads (Options::expert_pages).
    PagedDeepSeek(Model& m, const ConversationRegions& at, const Options& o, const ProgramOptions& x,
                  const std::vector<int>& eos, ExpertFiles& files, strata::core::RamExperts& ram);

    PagedParts make_parts(PagedCard& card, size_t count, std::string& err) override;
    bool get_card_layout(const PagedCard&, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                         std::string& err) const override {
        ConversationRegions at;
        return describe_conversation(m_.geometry(), (int) lb, (int) le, false, layout, at, err);
    }
    uint64_t get_gpu_keep(const PagedCard&) const override { return (uint64_t) o_.reserve_mib << 20; }
    std::vector<strata::core::ExpertKey> get_card_ranking(int64_t lb, int64_t le, int64_t& max_experts) const override;
    bool load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) override;
    std::vector<int32_t>& get_residency() override { return residency_; }
    bool publish_residency(const PagedCard& card, std::string& err) override;
    std::vector<float>& get_usage() override { return m_.get_usage(); }
    int get_adapt_interval(int64_t) const override { return x_.adapt_every; }
    bool set_card_layers(const PagedCard&, std::string& err) override {
        err = "strata-ds41 runs on one card";
        return false;
    }
    strata::core::LayerSizes get_layer_sizes(uint64_t page_bytes) const override;
    bool can_run_layers(int64_t lb, int64_t le) const override { return lb == 0 && le == m_.geometry().nl; }
    strata::core::RegionSpan get_gpu_span(const PagedCard&) const override { return {}; }
    strata::core::RegionSpan get_ram_span(int64_t, int64_t) const override { return {}; }
    uint64_t get_weight_bytes(const PagedCard&) const override { return m_.get_weight_bytes(0); }
    uint64_t get_scratch_bytes(const PagedCard&) const override { return m_.get_scratch_bytes(0); }
    bool attach_hop(PagedCard&, std::string& err) override {
        err = "strata-ds41 runs on one card";
        return false;
    }
    void set_hop_holders(const int32_t*) override {}
    void report_hops() const override {}
    void set_sampling(int session, const strata::serve::Sampling& smp) override;
    bool save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const override;
    bool restore_state(const std::vector<uint8_t>& state, const PagedParts& parts, const std::vector<int32_t>& live,
                       std::string& err) const override;
    std::vector<int64_t> get_rewind_points(const PagedParts& parts) const override;
    bool rewind(const PagedParts& parts, int64_t tokens, std::string& err) override;
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
    void print_info(int slots, int) const override {
        for (const std::string& line : m_.info_lines())
            std::printf("%s\n", line.c_str());
        if (slots > 0)
            std::printf("INFO batch_slots=%d slot_cache=0\n", slots);
    }
    void print_done(const strata::serve::RequestDone& d) const override {
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld 0 0\n", (long long) d.produced, (long long) d.prompt,
                    d.prompt_ms, d.decode_ms, d.finish, (long long) d.guessed_right, (long long) d.guessed,
                    (long long) d.resumed);
    }
    void end_request() override { m_.save_profile(); }

    /// Puts the experts `card`'s pool wants into it before the engine starts, as strata-q38's expert cache fills at
    /// load, read from the files (through hooks of its own, which the engine's replace); false (and `err`) when a read
    /// or a copy failed.
    bool fill_card(PagedCard& card, std::string& err);

private:
    bool go_to(int64_t pos, std::string& err);
    void end_prompt(Part& p);
    int32_t sample(const Part& p);
    bool copy_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, bool from_ram,
                     std::string& err);

    Model& m_;
    const ConversationRegions& at_;
    const Options& o_;
    const ProgramOptions x_;
    ExpertFiles& files_;
    strata::core::RamExperts& ram_;
    std::vector<uint8_t> staging_;     ///< an expert read from the files on its way to the GPU
    std::vector<int32_t> residency_;
    std::vector<Part*> parts_;         ///< by session, on the one card (make_parts'; a layer split is step D9)
    Part* seated_ = nullptr;           ///< the conversation the model computes (Part::take_seat), or null
    uint64_t counter_ = 0;             ///< the sampler's draw count, which the old engine keeps across requests
};

/// Gets the Part a PagedPart of this model is.
Part& get_part(strata::serve::PagedPart& part) { return static_cast<Part&>(part); }

int read_model_options(int argc, char** argv, Options& o, ProgramOptions& x, std::string& eos);
std::vector<strata::core::ExpertKey> get_ranking(const Model& m, int64_t lb, int64_t le);
bool open_card_pool(const Model& m, const PagedSettings& s, const strata::core::StateLayout& layout, uint64_t keep,
                    int64_t max_experts, PagedCard& card, std::string& err);

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // each protocol line reaches the server as it is printed
    strata::platform::fine_timer();
    PagedSettings settings;
    if (const int status = strata::serve::take_paged_options(argc, argv, kProgram, settings); status >= 0)
        return status;
    Options o;
    ProgramOptions x;
    std::string eos = "1";
    if (const int status = read_model_options(argc, argv, o, x, eos); status >= 0)
        return status;
    if (settings.step.lookahead > 0) {
        std::fprintf(stderr, "%s: --lookahead: the drafter comes in step D7\n", kProgram.c_str());
        return 2;
    }
    settings.step.concurrent = settings.step.rows = 1;   // one conversation a step, one row, no guesses
    settings.step.lookahead = 0;
    settings.drafter = false;
    strata::serve::set_paged_runtime_env(settings, false);
    ExpertFiles files;   // (before the RAM's pool, which reads through it)
    strata::core::RamExperts ram;
    o.expert_pages = &ram;   // (opened below, after the GPU's pool, before the first request)
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
    // the prompt path's buffers first (place_experts leaves the experts to the pools), then the GPU's pool in what
    // is left, filled from the files; then the RAM's pool, first with the experts the GPU's lacks
    strata::core::StateLayout layout;
    ConversationRegions at;
    bool copy_back = false;
    if (!describe_conversation(g, 0, g.nl, false, layout, at, err) || !m.place_experts(err) ||
        !files.open(m.get_file_paths(), m.get_expert_files(), err) ||
        !open_card_pool(m, settings, layout, (uint64_t) o.reserve_mib << 20, x.gpu_experts, card, err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    PagedDeepSeek model(m, at, o, x, parse_ids(eos), files, ram);
    if (x.adapt_every > 0 && x.adapt_swaps > 0)   // the routings the engine's adaptive tier counts
        m.get_usage().assign((size_t) g.nl * (size_t) g.n_exp, 0.0f);
    if (!model.fill_card(card, err) ||
        !strata::serve::open_ram_pool(kProgram, settings, files, kRamHeadroom, kCpuLaneExperts, layout, card, blobs,
                                      g.n_exp, get_ranking(m, 0, g.nl), model.get_residency(),
                                      card.experts.pool().page_address(0), ram, copy_back, err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    const int status = strata::serve::serve_paged_model(model, cards, ram, copy_back, settings, x.batch);
    m.save_profile();
    return status;
}

namespace {

PagedDeepSeek::PagedDeepSeek(Model& m, const ConversationRegions& at, const Options& o, const ProgramOptions& x,
                             const std::vector<int>& eos, ExpertFiles& files, strata::core::RamExperts& ram)
    : m_(m), at_(at), o_(o), x_(x), files_(files), ram_(ram) {
    const Geometry& g = m.geometry();
    facts.name = kProgram;
    facts.n_layers = g.nl;
    facts.n_expert = g.n_exp;
    facts.context = o.max_context;
    facts.max_rows = 1;
    facts.end_tokens.assign(eos.begin(), eos.end());
    facts.sampling.top_k = kDefaultTopK;
    facts.expert_bytes = m.get_expert_bytes();
    facts.gpu_experts = x.gpu_experts;
    facts.ram_headroom = kRamHeadroom;
    facts.adapt_swaps = x.adapt_swaps;
    facts.adapt_min_gain = x.adapt_min_gain;
    facts.adapt_decay = x.adapt_decay;
    staging_.resize(*std::max_element(facts.expert_bytes.begin(), facts.expert_bytes.end()));
    residency_.assign((size_t) g.nl * (size_t) g.n_exp, strata::core::kNotResident);
}

std::vector<strata::core::ExpertKey> PagedDeepSeek::get_card_ranking(int64_t lb, int64_t le,
                                                                     int64_t& max_experts) const {
    if (max_experts < 0)
        max_experts = (le - lb) * facts.n_expert;
    return get_ranking(m_, lb, le);
}

bool PagedDeepSeek::load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) {
    return copy_expert(card, key, first, ram_.holds(key.layer, key.expert), err);
}

bool PagedDeepSeek::publish_residency(const PagedCard& card, std::string& err) {
    const strata::core::PagePool& pool = card.experts.pool();
    std::vector<const uint8_t*> at(residency_.size(), nullptr);
    for (size_t i = 0; i < residency_.size(); ++i)
        if (residency_[i] != strata::core::kNotResident)
            at[i] = pool.page_address((strata::core::PageId) residency_[i]);
    const strata::core::OnDevice on(card.dev);
    return m_.set_expert_addresses(at, err);
}

bool PagedDeepSeek::fill_card(PagedCard& card, std::string& err) {
    strata::core::ExpertPages::Hooks hooks;
    hooks.load = [&](strata::core::ExpertKey key, int32_t first, std::string& e) {
        return copy_expert(card, key, first, false, e);
    };
    hooks.moved = [&](strata::core::ExpertKey key, int32_t first) {
        residency_[(size_t) key.layer * (size_t) facts.n_expert + (size_t) key.expert] = first;
    };
    card.experts.set_hooks(std::move(hooks));
    for (uint64_t wait_ms = 0;;) {
        const int got = card.experts.refill(INT_MAX, wait_ms, err);
        if (got < 0)
            return false;
        if (got == 0)
            return publish_residency(card, err);
    }
}

/// Copies expert `key` into `card`'s pool from page `first`: from the RAM's pool (`from_ram`, which holds it) or from
/// the files through the staging buffer.
bool PagedDeepSeek::copy_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, bool from_ram,
                                std::string& err) {
    const uint8_t* bytes = from_ram ? ram_.blob(key.layer, key.expert)
                                    : files_.copy_blob(key.layer, key.expert, staging_.data()) ? staging_.data()
                                                                                              : nullptr;
    const strata::core::OnDevice on(card.dev);
    if (bytes == nullptr ||
        !strata::core::gpu_copy(card.experts.pool().page_address((strata::core::PageId) first), bytes,
                                (size_t) facts.expert_bytes[(size_t) key.layer])) {
        err = "copying layer " + std::to_string(key.layer) + " expert " + std::to_string(key.expert) + " to " +
              card.id + " failed";
        return false;
    }
    return true;
}

PagedParts PagedDeepSeek::make_parts(PagedCard& card, size_t count, std::string& err) {
    if (!parts_.empty()) {   // (parts_ is one card's, by session: a layer split's cards come in step D9)
        err = "strata-ds41 runs on one card";
        return {};
    }
    PagedParts parts;
    for (size_t i = 0; i < count; ++i) {
        parts.push_back(std::make_unique<Part>(m_, at_, card.dev, seated_));
        parts_.push_back(&get_part(*parts.back()));
    }
    return parts;
}

void PagedDeepSeek::set_sampling(int session, const strata::serve::Sampling& smp) {
    Part& p = *parts_[(size_t) session];
    p.smp = smp;
    p.prompt_open = true;
    p.primed_pos = -1;   // (its pick had the last request's sampling)
}

// The state beside the pages is the conversation's host side (conversation_state.hpp's bytes).
bool PagedDeepSeek::save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string&) const {
    Part& p = get_part(*parts[0]);
    p.leave_seat();
    encode_state(p.state, state);
    return true;
}

bool PagedDeepSeek::restore_state(const std::vector<uint8_t>& state, const PagedParts& parts,
                                  const std::vector<int32_t>& live, std::string& err) const {
    Part& p = get_part(*parts[0]);
    ConversationState s;
    if (!decode_state(state, s, err))
        return false;
    if (const std::string why = m_.get_state_problem(s); !why.empty()) {
        err = "the paused conversation's checkpoints are damaged: " + why;
        return false;
    }
    // (one more token when the prompt's read took its last one along and the request ended before a step fed it)
    if (s.tokens.size() < live.size() || s.tokens.size() > live.size() + 1 ||
        !std::equal(live.begin(), live.end(), s.tokens.begin())) {
        err = "the paused conversation's checkpoints are another conversation's";
        return false;
    }
    p.leave_seat();
    p.state = std::move(s);
    p.pin_next = false;
    return true;
}

std::vector<int64_t> PagedDeepSeek::get_rewind_points(const PagedParts& parts) const {
    const Part& p = get_part(*parts[0]);
    return !p.is_ready() ? std::vector<int64_t>{} : seated_ == &p ? m_.check_positions() : get_check_positions(p.state);
}

bool PagedDeepSeek::rewind(const PagedParts& parts, int64_t tokens, std::string& err) {
    Part& p = get_part(*parts[0]);
    p.take_seat();
    p.primed_pos = -1;
    p.pin_next = true;
    if (m_.restore(tokens) && m_.pos() == tokens)
        return true;
    err = "no checkpoint at token " + std::to_string(tokens);
    return false;
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
    Part& p = get_part(*parts[0]);
    p.take_seat();
    if (!go_to(from, err))
        return false;
    // the prompt's last token rides along with the rest, as the old engine reads it (Model::feed_mark: one run of
    // chunks, which gives other numbers than a window of its own), and the token after it is sampled now, for the
    // step that would feed it
    const int64_t end = from + n;
    const bool last = end + 1 == (int64_t) ids.size();
    const std::vector<int> tokens(ids.begin() + from, ids.begin() + end + (last ? 1 : 0));
    if (last && !p.reserve(end + 1, err))
        return false;
    if (!(last ? m_.feed_mark(tokens.data(), (int) tokens.size(), 1, kBeforeLastCheck, err)
               : m_.feed(tokens.data(), (int) tokens.size(), 0, err)))
        return false;
    if (last) {
        end_prompt(p);
        p.primed_pos = end;
        p.primed_token = tokens.back();
        p.primed_pick = sample(p);
    }
    return on_chunk(from, n, err);
}

bool PagedDeepSeek::run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                             std::string& err) {
    if (rows.size() != 1) {
        err = "strata-ds41 runs one row a step (no drafting before step D7)";
        return false;
    }
    const strata::spec::StepRow& row = rows[0];
    Part& p = *parts_[(size_t) row.session];
    const bool primed = p.primed_pos == row.pos && p.primed_token == row.token && p.get_pos() == row.pos + 1;
    p.primed_pos = -1;
    if (primed) {
        picks[0] = p.primed_pick;
        return true;
    }
    // a prompt the conversation held up to its last token: that token is read here, with the prompt's checkpoints
    const bool closes = p.prompt_open;
    p.take_seat();
    if (!go_to(row.pos, err))
        return false;
    if (closes)
        m_.checkpoint(kBeforeLastCheck);
    if (!m_.step(row.token, true, err))
        return false;
    if (closes)
        end_prompt(p);
    picks[0] = sample(p);
    return true;
}

/// Keeps the model's state at the end of `p`'s prompt as a checkpoint, and the one before its last token as the anchor
/// when its conversation started over or went back (Part::pin_next), as the old engine does.
void PagedDeepSeek::end_prompt(Part& p) {
    m_.checkpoint(kPromptEndCheck);
    if (p.pin_next)
        m_.pin();
    p.pin_next = p.prompt_open = false;
}

/// Gets the token after the model's last logits by `p`'s request's sampling.
int32_t PagedDeepSeek::sample(const Part& p) {
    const strata::serve::Sampling& s = p.smp;
    return m_.sample(s.temperature, s.top_p, s.top_k, s.min_p, s.seed, counter_++);
}

/// Makes the model's state the seated conversation's at `pos` tokens: it is there, or one past it when the prompt's
/// read took the next token along and no step fed it (the request ended first), and goes back to the checkpoint
/// before it.
bool PagedDeepSeek::go_to(int64_t pos, std::string& err) {
    if (m_.pos() == pos)
        return true;
    if (m_.pos() == pos + 1 && m_.restore(pos) && m_.pos() == pos)
        return true;
    err = "the model's state is at token " + std::to_string(m_.pos()) + ", the conversation's at " +
          std::to_string(pos);
    return false;
}

/// Reads the model's own options, strata-ds41's (`x`) and --eos from `argv` (the paged engine's are gone) into `o`, `x`
/// and `eos`.  -1: go on, else the exit status.
int read_model_options(int argc, char** argv, Options& o, ProgramOptions& x, std::string& eos) {
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
        else if (a == "--batch")
            x.batch = std::max(0, std::atoi(val("--batch").c_str()));
        else if (a == "--expert-cache") {
            const std::string v = val("--expert-cache");
            x.gpu_experts = v == "auto" ? -1 : std::atoll(v.c_str());
        } else if (a == "--adapt-every")
            x.adapt_every = std::atoi(val("--adapt-every").c_str());
        else if (a == "--adapt-swaps")
            x.adapt_swaps = std::atoi(val("--adapt-swaps").c_str());
        else if (a == "--adapt-min-gain")
            x.adapt_min_gain = (float) std::atof(val("--adapt-min-gain").c_str());
        else if (a == "--adapt-decay")
            x.adapt_decay = (float) std::atof(val("--adapt-decay").c_str());
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
    if (!o.expert_cache)   // --no-expert-cache: the experts stay where the start put them, as strata-dsv41's do
        x.adapt_swaps = 0;
    o.dspark = false;   // (the drafter comes in step D7)
    return -1;
}

/// Gets the experts of layers [lb, le) in the order the GPU's pool takes them: each layer's hottest (Model::
/// get_heat_order), then each layer's second, and so on, so that a pool holds the same share of every layer, as the
/// old engine's VRAM does.
std::vector<strata::core::ExpertKey> get_ranking(const Model& m, int64_t lb, int64_t le) {
    std::vector<std::vector<int>> order;
    for (int64_t l = lb; l < le; ++l)
        order.push_back(m.get_heat_order((int) l));
    std::vector<strata::core::ExpertKey> ranking;
    for (size_t rank = 0; rank < (size_t) m.geometry().n_exp; ++rank)
        for (int64_t l = lb; l < le; ++l)
            ranking.push_back({(int32_t) l, order[(size_t) (l - lb)][rank]});
    return ranking;
}

/// Opens `card`'s page pool for conversations of `layout` and the first `max_experts` experts of get_ranking (-1:
/// every one) in the GPU memory the model left: --gpu-mib, by default what they and one conversation of the whole
/// context take, leaving `keep` free.
bool open_card_pool(const Model& m, const PagedSettings& s, const strata::core::StateLayout& layout, uint64_t keep,
                    int64_t max_experts, PagedCard& card, std::string& err) {
    const std::vector<uint64_t> blobs = m.get_expert_bytes();
    const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
    const std::vector<strata::core::ExpertKey> ranking = get_ranking(m, 0, m.geometry().nl);
    if (max_experts < 0)
        max_experts = (int64_t) ranking.size();
    uint64_t most = strata::serve::get_conversation_bytes(layout, m.max_context(), page), room = 0;
    for (size_t i = 0; i < ranking.size() && (int64_t) i < max_experts; ++i)
        most += blobs[(size_t) ranking[i].layer];   // (whole pages: the page divides every expert's size)
    const uint64_t bytes =
        strata::serve::get_gpu_pool_bytes(card.dev, keep, s.gpu_mib.empty() ? -1 : s.gpu_mib[0], most, room);
    strata::core::PagePoolConfig config;
    config.page_bytes = page;
    config.chunk_bytes = strata::serve::get_chunk_bytes(bytes, page);
    config.tokens_min = std::clamp<int64_t>(s.tokens_min, 0, m.max_context());
    const strata::core::OnDevice on(card.dev);
    if (!card.experts.open(strata::serve::get_gpu_chunk_memory(card.dev), bytes, layout, config, blobs,
                           m.geometry().n_exp, max_experts, ranking, err))
        return false;
    card.experts.set_ranking(ranking, max_experts);   // (opened without a cache, it wants none of them until then)
    strata::serve::say_gpu_pool(kProgram, card, room, config.chunk_bytes);
    return true;
}

}  // namespace

}  // namespace strata::ds4

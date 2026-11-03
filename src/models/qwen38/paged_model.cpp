// src/models/qwen38/paged_model.cpp - strata-q38: Qwen3.8 on the shared paged engine (paged_model.hpp): its start
// and its side of the engine's model interface (strata/common/serve/paged_model.hpp).
#include "strata/models/qwen38/paged_model.hpp"

#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/models/qwen38/prompt_images.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/expert_pages.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/common/kernels/mrope.hpp"
#include "strata/common/serve/paged_engine.hpp"
#include "strata/common/serve/paged_model.hpp"
#include "strata/common/serve/penalty_history.hpp"
#include "strata/models/qwen38/conversation.hpp"
#include "strata/models/qwen38/conversation_drafter.hpp"
#include "strata/models/qwen38/conversation_pages.hpp"
#include "strata/models/qwen38/conversation_snapshot.hpp"
#include "strata/models/qwen38/kernels/verify_kernels.hpp"
#include "strata/models/qwen38/model.hpp"
#include "strata/models/qwen38/options.hpp"
#include "strata/models/qwen38/session_identity.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::qwen38 {
namespace {

using strata::serve::PagedCard;
using strata::serve::PagedCards;
using strata::serve::PagedParts;
using strata::serve::PagedSettings;

/// The program's name, which starts its stderr lines.
const std::string kProgram = "strata-q38";

/// The lookahead when --mtp is given and --lookahead is not: on the Strix Halo one conversation decoded fastest with 3
/// guesses a step (4 rows) or 2, and several conversations with 8 rows (docs/DETAILS.md, rows per step).
constexpr int kDefaultLookahead = 3;
/// The milliseconds the CPU pool takes for one routed expert of a decode step, from RAM, against which a hop to
/// another card is weighed (--remote-experts auto, card_rules' is_hop_cheaper): the Strix Halo's pool computed 70,349
/// experts in 2,449 ms and 43,808 in 1,550 ms (two stages, 6 requests each, the reads from the SSD not counted).
constexpr double kCpuExpertMs = 0.035;
/// With drafting, the verify windows' graphs kept for the most recent row layouts (the old engine's --batch-mtp).
constexpr size_t kWindowGraphs = 64;

/// Qwen3.8's side of one of the engine's cards (PagedCard of the same index): its prompt path and verify windows,
/// which hand their rows to the next card's.  Card 0 runs on the Model's own cache and session and the Conversation's
/// windows, a later card on one of Model::stages.
struct ModelCard {
    GpuStage* stage = nullptr;                    ///< null for card 0
    /// the hand-offs its verify windows take their rows from (null for card 0) and give them to (null for the last)
    const float* hand_in = nullptr;
    float* hand_out = nullptr;
    strata::core::ExpertCache* cache = nullptr;   ///< the expert cache whose slots are the pool's pages
    strata::core::SessionState* own = nullptr;    ///< the engine's own session on it (the rope table and PLE it lends)
    void* cs = nullptr;                           ///< its stream, as the session functions take it
    strata::core::Verifier* ver = nullptr;
    strata::prefill::Prefill* sp = nullptr;
    int32_t* d_res = nullptr;                     ///< the residency table on its device
    uint64_t sessions_bytes = 0;                  ///< the device memory of the sessions on it (`own`'s included)
    /// computes the experts of other cards' layers its pool holds (PagedCard::lent) that a verify window of their own
    /// card routes to
    strata::core::PeerExperts hop;
};
using ModelCards = std::deque<ModelCard>;   ///< (a deque, as PagedCards)

/// A conversation's state on one card: a session carved for the card's layers, whose GDN state and K/V are pages of
/// the card's pool.
struct Part final : strata::serve::PagedPart {
    std::vector<int32_t> image_positions; ///< rotary coordinates retained with paused state
    const Model& m;
    const ModelCard& card;              ///< its card's model side, whose stream and verify windows it uses
    int dev = 0;                        ///< its card's device
    strata::core::SessionState ss;
    void* mem = nullptr;                ///< `ss`'s memory (without the state the pages hold)
    ConversationPages pages;            ///< given at the conversation's first request
    /// where its GDN state was when the card's verify windows' graphs last saw it (their graphs hold the address)
    const float* gdn_seen = nullptr;

    bool image_table_ready = false; ///< Whether all cards have this conversation's current coordinates.
    int32_t* image_table = nullptr; ///< Stable coordinates for this card's indexer graph launches.
    Part(const Model& model, const ModelCard& on_card, int device) : m(model), card(on_card), dev(device) {}
    ~Part() override {
        const strata::core::OnDevice on(dev);
        strata::core::gpu_free(image_table);
        strata::core::session_release(ss);
        (void) cudaFree(mem);
    }

    bool is_ready() const override { return pages.is_ready(); }
    bool init(strata::core::PagePool& pool, strata::core::ConversationId id, std::string& err) override {
        return pages.init(m.g, ss, pool, id, err) && note_gdn_place(err);
    }
    bool reserve(int64_t tokens, std::string& err) override { return pages.reserve(tokens, err); }
    void truncate(int64_t tokens) override { pages.truncate(tokens); }
    void release() override { pages.release(); }
    bool refresh(std::string& err) override { return pages.refresh(ss, err) && note_gdn_place(err); }
    void clear_state() override {
        const strata::core::OnDevice on(dev);
        strata::core::session_zero(ss, m.g, nullptr, card.cs);
        (void) cudaStreamSynchronize((cudaStream_t) card.cs);
    }
    void set_layers(int64_t lb, int64_t le) override {
        strata::core::session_set_layers(ss, m.g, lb, le);
        gdn_seen = nullptr;
    }
    uint64_t bytes() const override { return pages.bytes(); }
    strata::core::ConversationId id() const override { return pages.id(); }
    int64_t get_pool_tokens(int64_t tokens) const override { return pages.get_pool_tokens(tokens); }

    /// Drops the card's verify windows' graphs when its GDN state is elsewhere than when they last saw it.
    bool note_gdn_place(std::string& err) {
        const bool moved = gdn_seen != nullptr && gdn_seen != ss.gdn_state;
        gdn_seen = ss.gdn_state;
        const strata::core::OnDevice on(dev);
        return !moved || card.ver->drop_batch_graphs(err);
    }
};

/// Gets the Part a PagedPart of this model is.
Part& get_part(strata::serve::PagedPart& part) { return static_cast<Part&>(part); }

/// Qwen3.8 for the paged engine: the Model and its Conversation (card 0's prompt path and verify windows), the cards'
/// stages, the layer split's CPU routing and the MTP drafter.
class PagedQwen final : public strata::serve::PagedModel {
public:
    /// `drafter`: null when drafting is off; `sessions`: the conversations the engine holds (its batch slots and
    /// "main").
    PagedQwen(Model& m, Conversation& conv, const PagedCards& cards, ModelCards& model_cards, SplitDrive& split,
              const strata::core::RamExperts& ram, ConversationDrafter* drafter, int sessions);

    bool prepare_images(const strata::serve::RequestLine& request, const std::vector<int64_t>& ids,
                        uint64_t& key, std::string& err) override;
    void attach_images(const PagedParts& parts) override {
        Part& lead = get_part(*parts.front());
        lead.image_positions = std::move(pending_positions_);
        lead.image_table_ready = false;
    }
    PagedParts make_parts(PagedCard& card, size_t count, std::string& err) override;
    bool get_card_layout(const PagedCard& card, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                         std::string& err) const override;
    uint64_t get_gpu_keep(const PagedCard& card) const override;
    std::vector<strata::core::ExpertKey> get_card_ranking(int64_t lb, int64_t le, int64_t& max_experts) const override;
    bool load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) override;
    std::vector<int32_t>& get_residency() override { return m_.host_res; }
    bool publish_residency(const PagedCard& card, std::string& err) override;
    std::vector<float>& get_usage() override { return m_.drive.d.usage; }
    int get_adapt_interval(int64_t steps) const override { return m_.o.adapt_iv(steps); }
    bool set_card_layers(const PagedCard& card, std::string& err) override;
    strata::core::LayerSizes get_layer_sizes(uint64_t page_bytes) const override;
    bool can_run_layers(int64_t lb, int64_t le) const override;
    strata::core::RegionSpan get_gpu_span(const PagedCard& card) const override;
    strata::core::RegionSpan get_ram_span(int64_t lb, int64_t le) const override;
    uint64_t get_weight_bytes(const PagedCard& card) const override;
    uint64_t get_scratch_bytes(const PagedCard& card) const override { return cards_[card.index].sessions_bytes; }
    bool attach_hop(PagedCard& card, std::string& err) override;
    void set_hop_holders(const int32_t* holder) override { split_.holder = holder; }
    void report_hops() const override;
    void start_request(int session, const strata::serve::RequestStart& r) override;
    bool save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const override;
    bool restore_state(const std::vector<uint8_t>& state, const PagedParts& parts, const std::vector<int32_t>& live,
                       std::string& err) const override;
    bool get_session_identity(strata::core::SessionFileIdentity& id, std::string& err) const override {
        strata::core::SessionIdentityBuilder config(get_session_config_fingerprint(m_));
        config.u64("stages", cards_.size());
        id.config = config.digest();
        return get_session_model_fingerprint(m_, &m_.src, id.model, err);
    }
    void forget_drafts(int session) override { drafter_->forget(session); }
    int64_t get_prompt_chunk() const override { return conv_.sp.chunk(); }
    bool read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n, bool drafting,
                     const ChunkDone& on_chunk, std::string& err) override;
    void set_step_histories(const std::vector<const std::vector<int32_t>*>& histories) override {
        histories_ = histories;
    }
    bool run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                  std::string& err) override;
    bool keep_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept, bool drafting,
                   std::string& err) override;
    int propose(strata::serve::PagedPart& last, int session, int32_t, int64_t, int n, int32_t* out,
                std::string& err, strata::spec::DraftContext* draft) override;
    bool switch_drafter(bool on, std::string& err) override;
    void print_info(int slots, int spec) const override;
    void print_done(const strata::serve::RequestDone& done) const override;

private:
    /// Uploads changed indexer coordinates and, when shared, activates the prompt/drafter table.
    bool activate_images(int session, std::string& err, bool shared = true);
    std::vector<int32_t> pending_positions_;
    std::vector<float> image_rows_;
    std::vector<const float*> image_row_ptrs_;
    std::vector<Part*> leads_;
    std::vector<std::vector<int32_t*>> image_tables_;
    Model& m_;
    Conversation& conv_;   ///< card 0's prompt path and verify windows, shared by every conversation
    const PagedCards& pcards_;
    ModelCards& cards_;
    SplitDrive& split_;
    const strata::core::RamExperts& ram_;
    ConversationDrafter* const drafter_;
    const int sessions_;
    std::vector<int32_t> picks_;       ///< the last step's token after each row, for the drafter
    std::vector<const std::vector<int32_t>*> histories_; ///< consumed tokens for the current step
    std::vector<int> history_lengths_; ///< each session's penalty window
};

std::string get_split_hop_refusal(const Options& o);
std::vector<uint64_t> get_blob_bytes(const Model& m);
PagedCard& add_card(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, GpuStage* stage);
bool open_pages(Model& m, PagedCard& card, ModelCard& mc, int64_t tokens_min, int64_t gpu_mib, int64_t max_experts,
                std::string& err);
bool open_ram(Model& m, const PagedCards& cards, const ModelCards& model_cards, const PagedSettings& s,
              strata::core::RamExperts& ram, bool& copy_back, std::string& err);
bool init_card_paths(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, SplitDrive& split,
                     std::string& err);
std::string get_unsupported_reason(const Model& m);
void append_size(std::vector<uint8_t>& bytes, uint64_t n);
uint64_t read_size(const std::vector<uint8_t>& bytes, size_t& at);

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // each protocol line reaches the server as it is printed
    PagedSettings settings;
    if (const int status = strata::serve::take_paged_options(argc, argv, kProgram, settings); status >= 0)
        return status;
    prepare_process(argc, argv);
    Options o;
    bool have_tokens = false;
    if (const int status = parse_options(argc, argv, o, have_tokens); status >= 0) return status;
    const std::string hop_refusal = get_split_hop_refusal(o);
    if (settings.remote_experts == "on" && !hop_refusal.empty()) {
        std::fprintf(stderr, "%s: --remote-experts on: %s; use --remote-experts auto or off\n", kProgram.c_str(),
                     hop_refusal.c_str());
        return 2;
    }
    strata::serve::set_paged_runtime_env(settings, !o.layer_split.empty() && hop_refusal.empty());
    prime_gpu_queues();
    if (const int status = apply_device_defaults(o); status >= 0) return status;
    const int slots = strata::kernels::kVerifyMaxT;
    if (settings.step.concurrent < 0)
        settings.step.concurrent = std::max(o.batch, 1);
    if (settings.step.lookahead < 0)
        settings.step.lookahead = o.mtp.empty() && settings.suffix_draft == 0 && settings.lookup_chain == 0 ?
                                      0 : kDefaultLookahead;
    if (const std::string why = strata::serve::get_step_settings_problem(settings.step, strata::kernels::kVerifyMaxT);
        !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    const bool drafting = settings.step.lookahead > 0 && !o.mtp.empty();
    settings.drafter = settings.drafter && drafting;
    if (settings.step.lookahead > 0 && o.mtp.empty() && settings.suffix_draft == 0 && settings.lookup_chain == 0) {
        std::fprintf(stderr, "strata-q38: --lookahead needs --mtp or --suffix-draft above 0\n");
        return 2;
    }
    // The MTP's weights are loaded only with a lookahead; no K/V streaming or elastic K/V (the pool
    // replaces them); the batch slots are the engine's own conversations, not the old engine's sessions of the whole
    // context; the experts from the model's files, the RAM's pool in place of the RAM copy or the arena
    o.serve = false;
    o.batch = 0;
    if (!drafting)
        o.mtp.clear();
    o.draft.clear();
    o.spec = strata::kernels::kVerifyMaxT;   // the verify windows' room: --rows may change up to it
    o.kv_grow = false;
    o.kv_resident = 0;
    // strata-q38 does not pipeline verify windows across cards; 0 also keeps the drafter on the last card (mtp_first)
    o.pipeline_windows = 0;
    if (o.prefill_chunk <= 0) o.prefill_chunk = 2048;   // open_pages leaves room for the prompt path's own buffers
    if (o.resident_budget > 0 && settings.ram_mib < 0)
        settings.ram_mib = (int64_t) (o.resident_budget >> 20);
    o.mmap_experts = true;
    o.resident_cpu_experts = o.resident_pin = o.resident_soft = false;
    o.resident_budget = 0;
    if (o.expert_cache_per_layer) {
        std::fprintf(stderr, "strata-q38: --expert-cache-per-layer: its experts share the page pool\n");
        return 2;
    }
    strata::core::session_set_pool_state(true);
    Model m(o);
    m.own_stages = true;
    m.mtp.set_kv_in_pages(true);
    Conversation conv;
    PagedCards cards;
    ModelCards model_cards;
    const auto get_gpu_mib = [&](size_t card) {
        return settings.gpu_mib.empty() ? -1 : settings.gpu_mib[std::min(card, settings.gpu_mib.size() - 1)];
    };
    m.open_expert_pages = [&](int64_t max_experts, std::string& err) {
        // from here on the experts are read from the files without the file cache
        if (std::string why; !m.src.read_unbuffered(why))
            std::fprintf(stderr, "strata-q38: WARNING: the experts are read through the file cache (%s)\n",
                         why.c_str());
        PagedCard& card = add_card(m, conv, cards, model_cards, nullptr);
        return open_pages(m, card, model_cards[card.index], settings.tokens_min, get_gpu_mib(0), max_experts, err);
    };
    m.open_stage_pages = [&](GpuStage& st, std::string& err) {
        PagedCard& card = add_card(m, conv, cards, model_cards, &st);
        return open_pages(m, card, model_cards[card.index], settings.tokens_min, get_gpu_mib(card.index),
                          o.expert_cache, err);
    };
    if (const int status = m.load(); status != 0) return status;
    if (const int status = m.build_hit_tables(false); status != 0) return status;
    for (ModelCard& mc : model_cards)
        mc.d_res = mc.stage != nullptr ? mc.stage->d_res : m.d_res;
    if (const std::string why = get_unsupported_reason(m); !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    m.finish_expert_tiers();
    if (o.adapt_every > 0 && o.adapt_swaps > 0)   // the routings the engine's adaptive tier counts
        m.drive.d.usage.assign((size_t) (m.g.n_layers * m.g.n_expert), 0.0f);
    std::string err;
    strata::core::RamExperts ram;
    bool copy_back = false;
    if (!open_ram(m, cards, model_cards, settings, ram, copy_back, err)) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    m.use_expert_source(ram);
    m.set_prompt_pinned_share();
    m.arm_prompt_cpu_share();
    ConversationDrafter drafter;
    SplitDrive split;
    if (!init_card_paths(m, conv, cards, model_cards, split, err) || (drafting && !drafter.init(m, slots + 1, err))) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    if (drafting)   // the conversations' guesses make many row layouts: the verify windows keep the recent ones' graphs
        for (const ModelCard& mc : model_cards)
            mc.ver->set_batch_graph_limit(kWindowGraphs);
    if (drafting && !settings.drafter)   // loaded for the pool's layout of the drafter's K/V, switched on later
        m.unload_drafter();
    PagedQwen model(m, conv, cards, model_cards, split, ram, drafting ? &drafter : nullptr, slots + 1);
    return strata::serve::serve_paged_model(model, cards, ram, copy_back, settings);
}

namespace {

PagedQwen::PagedQwen(Model& m, Conversation& conv, const PagedCards& cards, ModelCards& model_cards,
                     SplitDrive& split, const strata::core::RamExperts& ram, ConversationDrafter* drafter,
                     int sessions)
    : m_(m), conv_(conv), pcards_(cards), cards_(model_cards), split_(split), ram_(ram), drafter_(drafter),
      sessions_(sessions) {
    const Options& o = m.o;
    facts.name = kProgram;
    facts.n_layers = m.g.n_layers;
    facts.n_expert = m.g.n_expert;
    facts.context = o.max_context;
    facts.max_rows = strata::kernels::kVerifyMaxT;
    facts.end_tokens = o.eos_ids;
    facts.has_drafter = drafter != nullptr;
    facts.draft_limits.confidence = (float) o.spec_min_p;
    facts.drafter_start = "--mtp and a --lookahead above 0";
    facts.expert_bytes = get_blob_bytes(m);
    for (size_t i = 0; i < m.profile_all.size() && (int64_t) i < o.expert_cache; ++i)
        facts.split_ranking.push_back({m.profile_all[i].first, m.profile_all[i].second});
    facts.gpu_experts = o.expert_cache;
    facts.ram_headroom = o.resident_headroom;
    facts.adapt_swaps = o.adapt_swaps;
    facts.adapt_min_gain = o.adapt_min_gain;
    facts.adapt_decay = o.adapt_decay;
    facts.row_bytes = (uint64_t) strata::kernels::cpu::H * sizeof(float);
    facts.cpu_expert_ms = kCpuExpertMs;
}

PagedParts PagedQwen::make_parts(PagedCard& card, size_t count, std::string& err) {
    using namespace strata::core;
    ModelCard& mc = cards_[card.index];
    const OnDevice on(card.dev);
    // every session on the card reads the engine's own session's rope table there
    const QsaState* rope = &mc.own->qsa_states[mc.own->qsa_primary()];
    // (carved for every layer, as the card's own session: a moved cut changes the layers they are for)
    const uint64_t bytes = session_bytes(m_.g, m_.o.max_context, m_.K, 0, -1, false);
    mc.sessions_bytes = session_bytes(m_.g, m_.o.max_context, m_.K) + count * bytes;
    std::vector<SessionState*> sessions;
    std::vector<int32_t*> image_tables;
    PagedParts parts;
    for (size_t i = 0; i < count; ++i) {
        auto part = std::make_unique<Part>(m_, mc, card.dev);
        if (cudaMalloc(&part->mem, bytes) != cudaSuccess ||
            session_init(m_.g, m_.o.max_context, m_.K, part->mem, part->ss, 0, -1, rope) == 0) {
            char text[160];
            std::snprintf(text, sizeof text, "no room for conversation %zu's session on %s (%.1f MiB)", i,
                          card.id.c_str(), (double) bytes / 1048576.0);
            err = text;
            return {};
        }
        if (m_.o.vision) {
            part->image_table = static_cast<int32_t*>(gpu_alloc_zeroed(m_.mrope_host.size() * sizeof(int32_t)));
            if (!part->image_table) {
                err = "no room for the conversation's image coordinates";
                return {};
            }
            image_tables.push_back(part->image_table);
            mc.sessions_bytes += m_.mrope_host.size() * sizeof(int32_t);
        }
        session_set_layers(part->ss, m_.g, card.lb, card.le);
        PleRun& ple = part->ss.ple;   // the engine's PLE weights and buffers, with this conversation's history
        ple = mc.own->ple;
        ple.token = &part->ss.ple_token;
        ple.prev = part->ss.ple_prev;
        ple.hist = part->ss.ple_hist;
        session_zero(part->ss, m_.g, nullptr, mc.cs);
        sessions.push_back(&part->ss);
        if (card.index == 0)
            leads_.push_back(part.get());
        parts.push_back(std::move(part));
    }
    if (cudaStreamSynchronize((cudaStream_t) mc.cs) != cudaSuccess || !mc.ver->init_slots(sessions, err)) {
        err = "the conversations' sessions on " + card.id + ": " + err;
        return {};
    }
    mc.ver->set_slot_image_tables({image_tables.begin(), image_tables.end()});
    image_tables_.push_back(std::move(image_tables));
    return parts;
}

/// Gets the bytes of one expert of each of `m`'s layers.
std::vector<uint64_t> get_blob_bytes(const Model& m) {
    std::vector<uint64_t> blobs;
    for (int64_t l = 0; l < m.g.n_layers; ++l)
        blobs.push_back(strata::kernels::cpu::expert_layout().blob_bytes(l));
    return blobs;
}

/// Gets whether `card` runs `m`'s MTP drafter (--mtp), which Model::load_drafter puts on a layer split's last card.
bool has_drafter(const Model& m, const ModelCard& card) {
    return !m.o.mtp.empty() && card.stage == m.get_drafter_stage();
}

/// Gets the experts of the expert profile of layers [lb, le), hottest first.  With a layer split, `max_experts` (the
/// most the GPUs hold: --expert-cache) becomes how many of them are among the profile's first `max_experts`, so that
/// the cards hold the experts one GPU would; without one, they are CUDA0's profile and `max_experts` stays.
std::vector<strata::core::ExpertKey> get_card_ranking(const Model& m, int64_t lb, int64_t le, int64_t& max_experts) {
    std::vector<strata::core::ExpertKey> ranking;
    if (m.profile_all.empty()) {
        for (const auto& [layer, expert] : m.profile)
            ranking.push_back({layer, expert});
        return ranking;
    }
    int64_t held = 0;
    for (size_t i = 0; i < m.profile_all.size(); ++i)
        if (const auto [layer, expert] = m.profile_all[i]; layer >= lb && layer < le) {
            ranking.push_back({layer, expert});
            held += (int64_t) i < max_experts;
        }
    max_experts = held;
    return ranking;
}

/// Gets the regions a conversation keeps in the pool of `card` (whose pool is open; `mc` its model side) when the card
/// runs layers [lb, le): ConversationPages::get_layout's for its session carved for them, and the drafter's K/V region
/// of its pool when it runs the drafter (has_drafter).
bool get_card_layout(const Model& m, const PagedCard& card, const ModelCard& mc, int64_t lb, int64_t le,
                     strata::core::StateLayout& layout, std::string& err) {
    strata::core::SessionState carve = *mc.own;   // (a copy of the description only: the memory stays the session's)
    strata::core::session_set_layers(carve, m.g, lb, le);
    if (!ConversationPages::get_layout(m.g, carve, nullptr, layout, err))
        return false;
    if (has_drafter(m, mc))
        layout.regions.push_back(card.experts.pool().layout().regions.back());
    return true;
}

/// Gets the GPU memory `card`'s pool leaves free besides what is allocated there: --vram-reserve-mib, and the drafter's
/// head when the card runs it (has_drafter) and has not bound it.  Before the start allocates the card's prompt path
/// (`started` false), also the prompt path's own buffers and the head, which the GPU's free memory still holds.
uint64_t get_gpu_keep(const Model& m, const ModelCard& card, bool started) {
    const uint64_t head = (card.stage != nullptr ? card.stage->head : m.native_head).row_bytes();
    return ((uint64_t) m.o.vram_reserve_mib << 20) +
           (started ? 0 : strata::prefill::Prefill::bytes_needed_owned(m.g, *card.own, m.o.prefill_chunk,
                                                                       m.srcp != nullptr)) +
           (has_drafter(m, card) && (!started || m.mtp.vram_bytes() == 0) ? m.mtp.bind_bytes(head, m.n_vocab) : 0);
}

/// Gives card `k` its layers in its prompt path and verify windows (PagedCard::lb and le), as their stage of the layer
/// split, handing their rows on to the next card's.
void set_stage(const PagedCards& cards, const ModelCards& model_cards, size_t k) {
    const PagedCard& card = cards[k];
    const ModelCard& mc = model_cards[k];
    const bool last = k + 1 == cards.size();
    mc.sp->set_stage(card.lb, card.le, last ? nullptr : model_cards[k + 1].sp);
    mc.ver->set_stage(card.lb, card.le, mc.hand_in, mc.hand_out);
}

bool PagedQwen::get_card_layout(const PagedCard& card, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                                std::string& err) const {
    return qwen38::get_card_layout(m_, card, cards_[card.index], lb, le, layout, err);
}

uint64_t PagedQwen::get_gpu_keep(const PagedCard& card) const {
    return qwen38::get_gpu_keep(m_, cards_[card.index], true);
}

std::vector<strata::core::ExpertKey> PagedQwen::get_card_ranking(int64_t lb, int64_t le, int64_t& max_experts) const {
    return qwen38::get_card_ranking(m_, lb, le, max_experts);
}

bool PagedQwen::load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) {
    const uint8_t* blob = m_.srcp->blob_stable(key.layer, key.expert);
    if (blob == nullptr)
        err = "no bytes for layer " + std::to_string(key.layer) + " expert " + std::to_string(key.expert);
    const strata::core::OnDevice on(card.dev);
    return blob != nullptr &&
           cards_[card.index].cache->fill_slot_blocking(
               first, blob, err, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(key.layer));
}

bool PagedQwen::publish_residency(const PagedCard& card, std::string& err) {
    const strata::core::OnDevice on(card.dev);
    if (const cudaError_t e = m_.put_residency(cards_[card.index].d_res); e != cudaSuccess) {
        err = "the expert residency table of " + card.id + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool PagedQwen::set_card_layers(const PagedCard& card, std::string& err) {
    ModelCard& mc = cards_[card.index];
    if (mc.stage != nullptr) {
        mc.stage->lb = card.lb;
        mc.stage->le = card.le;
    }
    strata::core::session_set_layers(*mc.own, m_.g, card.lb, card.le);
    set_stage(pcards_, cards_, card.index);
    split_.end[card.index] = card.le;
    m_.split_at.clear();   // (complete once the last card has its layers)
    for (size_t k = 1; k < pcards_.size(); ++k)
        m_.split_at.push_back(pcards_[k].lb);
    return mc.ver->drop_batch_graphs(err);
}

strata::core::LayerSizes PagedQwen::get_layer_sizes(uint64_t page) const {
    const strata::core::ModelGeometry& g = m_.g;
    const std::vector<strata::core::StateRegion>& whole = ram_.pages().pool().layout().regions;   // (every layer's)
    strata::core::LayerSizes sizes;
    for (int64_t l = 0, qsa = 0; l < g.n_layers; ++l) {
        const bool is_qsa = is_qsa_layer(g, l);
        sizes.dense.push_back(0);   // (outside the pools)
        sizes.expert.push_back((strata::kernels::cpu::expert_layout().blob_bytes(l) + page - 1) / page * page);
        sizes.fixed_state.push_back(is_qsa ? 0 : whole[0].unit_bytes / (uint64_t) g.n_gdn_layers());
        sizes.token_bytes.push_back(is_qsa ? whole[(size_t) (1 + qsa++)].unit_bytes / whole[1].unit_tokens() : 0);
    }
    return sizes;
}

bool PagedQwen::can_run_layers(int64_t lb, int64_t le) const {
    // a card needs a GDN layer and a QSA layer
    const int64_t interval = std::max<int64_t>(m_.g.qsa_interval, 1), qsa = le / interval - lb / interval;
    return qsa != 0 && qsa != le - lb;
}

strata::core::RegionSpan PagedQwen::get_gpu_span(const PagedCard& card) const {
    // its GDN state, then its QSA layers' regions (and the drafter's, last)
    return {1, SIZE_MAX, 0, 0, card.experts.pool().layout().regions[0].unit_bytes};
}

strata::core::RegionSpan PagedQwen::get_ram_span(int64_t lb, int64_t le) const {
    // the card's share of the GDN state, and its QSA layers' regions (and the drafter's, last)
    const strata::core::ModelGeometry& g = m_.g;
    const int64_t interval = std::max<int64_t>(g.qsa_interval, 1), q0 = lb / interval, q1 = le / interval;
    const uint64_t gdn_layer = ram_.pages().pool().layout().regions[0].unit_bytes / (uint64_t) g.n_gdn_layers();
    return {(size_t) (1 + q0), le == g.n_layers ? SIZE_MAX : (size_t) (q1 - q0), 0, (uint64_t) (lb - q0) * gdn_layer,
            (uint64_t) (le - q1 - (lb - q0)) * gdn_layer};
}

uint64_t PagedQwen::get_weight_bytes(const PagedCard& card) const {
    const GpuStage* stage = cards_[card.index].stage;
    return stage != nullptr ? stage->dense.allocated_bytes() + stage->head.weight_bytes()
                            : m_.pool_bytes + m_.native_dense.allocated_bytes() + m_.native_head.weight_bytes();
}

bool PagedQwen::attach_hop(PagedCard& card, std::string& err) {
    ModelCard& mc = cards_[card.index];
    if (!mc.hop.attach(card.dev, *mc.cache, card.lent.data(), m_.g.n_layers, m_.g.n_expert, err))
        return false;
    split_.hop[card.index] = &mc.hop;
    return true;
}

void PagedQwen::report_hops() const {
    for (size_t k = 0; k < cards_.size(); ++k)
        if (const strata::core::PeerExperts& hop = cards_[k].hop; hop.experts() > 0)
            std::fprintf(stderr, "strata-q38: so far %s computed %lld experts of other cards' layers (%lld rows) in "
                                 "one hop each layer, %.1f ms waited for\n",
                         pcards_[k].id.c_str(), (long long) hop.experts(), (long long) hop.entries(), hop.ms_wait);
}

bool PagedQwen::prepare_images(const strata::serve::RequestLine& request, const std::vector<int64_t>& ids,
                                uint64_t& key, std::string& err) {
    pending_positions_.clear();
    image_rows_.clear();
    image_row_ptrs_.clear();
    if (!request.pictures)
        return true;
    if (!m_.o.vision) {
        err = "this engine was started without --vision";
        return false;
    }
    pending_positions_.resize(m_.mrope_host.size());
    std::vector<uint64_t> keys;
    for (const auto& [name, value] : request.keys)
        if (name == "img_keys")
            keys = strata::serve::parse_hex_list(value);
    std::vector<strata::core::ConversationImageKey> hashes;
    if (!read_prompt_images(request.file, ids, m_.g.n_embd, keys, pending_positions_, image_rows_,
                            image_row_ptrs_, hashes, err))
        return false;
    key = 1469598103934665603ull;
    for (const auto& image : hashes)
        key = hash_image_bytes(&image, sizeof image, key);
    if (key == 0)
        key = 1;
    return true;
}

bool PagedQwen::activate_images(int session, std::string& err, bool shared) {
    if (!m_.o.vision)
        return true;
    Part& lead = *leads_[(size_t) session];
    if (!shared && lead.image_table_ready)
        return true;
    const auto& positions = lead.image_positions;
    if (positions.empty()) {
        for (size_t i = 0; i < m_.mrope_host.size(); ++i)
            m_.mrope_host[i] = (int32_t) (i / 3);
    } else {
        m_.mrope_host = positions;
    }
    for (size_t k = 0; k < cards_.size(); ++k) {
        const strata::core::OnDevice on(pcards_[k].dev);
        int32_t* table = k == 0 ? m_.d_mrope : cards_[k].stage->mrope;
        strata::kernels::mrope_table_set(table);
        const size_t bytes = m_.mrope_host.size() * sizeof(int32_t);
        if ((shared && !strata::core::gpu_copy(table, m_.mrope_host.data(), bytes)) ||
            (!lead.image_table_ready && !strata::core::gpu_copy(image_tables_[k][(size_t) session],
                                                               m_.mrope_host.data(), bytes))) {
            err = "the image position upload failed";
            return false;
        }
    }
    lead.image_table_ready = true;
    return true;
}

void PagedQwen::start_request(int session, const strata::serve::RequestStart& r) {
    strata::kernels::SamplerParams sp = get_sampler_params(r.sampling);
    if (history_lengths_.size() < (size_t) sessions_)
        history_lengths_.resize((size_t) sessions_);
    history_lengths_[(size_t) session] = std::min(sp.penalty_last_n, 4096);
    conv_.ver.set_slot_sampling(session, sp);
}

// The state beside the pages is every QSA layer's indexer state (tail, dead, block_pos) in the layers' order, its
// size first, then each card's PLE history, its size first: the QSA layers move between cards with the cut, the PLE
// history is each card's whole. Image rotary coordinates, when present, follow the histories.
bool PagedQwen::save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const {
    strata::core::ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(m_.g, z, err))
        return false;
    std::vector<uint8_t> qsa, ple;
    for (const std::unique_ptr<strata::serve::PagedPart>& part : parts) {
        const Part& p = get_part(*part);
        const strata::core::OnDevice on(p.dev);
        std::vector<uint8_t> one;   // the PLE history, then each of the card's QSA layers' (tail, dead, block_pos)
        if (!strata::core::conversation_unpaged_save(p.ss, m_.g, one, err))
            return false;
        const size_t n = one.size() - (size_t) p.ss.qsa_alloc * (z.tail + z.dead + z.block_pos);
        append_size(ple, n);
        ple.insert(ple.end(), one.begin(), one.begin() + (std::ptrdiff_t) n);
        qsa.insert(qsa.end(), one.begin() + (std::ptrdiff_t) n, one.end());
    }
    state.clear();
    append_size(state, qsa.size());
    state.insert(state.end(), qsa.begin(), qsa.end());
    state.insert(state.end(), ple.begin(), ple.end());
    const auto& positions = get_part(*parts.front()).image_positions;
    if (!positions.empty()) {
        append_size(state, positions.size());
        const auto* bytes = reinterpret_cast<const uint8_t*>(positions.data());
        state.insert(state.end(), bytes, bytes + positions.size() * sizeof(int32_t));
    }
    return true;
}

bool PagedQwen::restore_state(const std::vector<uint8_t>& state, const PagedParts& parts,
                              const std::vector<int32_t>& live, std::string& err) const {
    strata::core::ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(m_.g, z, err))
        return false;
    const size_t per_layer = z.tail + z.dead + z.block_pos;
    size_t at = 0;
    const size_t qsa = read_size(state, at);
    if (qsa > state.size() - at) {
        err = "the paused conversation's QSA state is damaged";
        return false;
    }
    const auto qsa_begin = state.begin() + (std::ptrdiff_t) at;
    at += qsa;
    for (const std::unique_ptr<strata::serve::PagedPart>& part : parts) {
        Part& p = get_part(*part);
        const strata::core::OnDevice on(p.dev);
        const size_t n = read_size(state, at);
        if (n > state.size() - at || (size_t) (p.ss.qsa_ord0 + p.ss.qsa_alloc) * per_layer > qsa) {
            err = "the paused conversation's running state is damaged";
            return false;
        }
        std::vector<uint8_t> one(state.begin() + (std::ptrdiff_t) at, state.begin() + (std::ptrdiff_t) (at + n));
        at += n;
        const auto first = qsa_begin + (std::ptrdiff_t) ((size_t) p.ss.qsa_ord0 * per_layer);
        one.insert(one.end(), first, first + (std::ptrdiff_t) ((size_t) p.ss.qsa_alloc * per_layer));
        if (!strata::core::conversation_unpaged_restore(one, p.ss, m_.g, live, err))
            return false;
    }
    Part& lead = get_part(*parts.front());
    lead.image_table_ready = false;
    auto& positions = lead.image_positions;
    positions.clear();
    if (at < state.size()) {
        const uint64_t n = read_size(state, at);
        if (!m_.o.vision || n != m_.mrope_host.size() || n > (state.size() - at) / sizeof(int32_t) ||
            at + n * sizeof(int32_t) != state.size()) {
            err = "the paused conversation's image positions are invalid";
            return false;
        }
        positions.resize((size_t) n);
        std::memcpy(positions.data(), state.data() + at, (size_t) n * sizeof(int32_t));
    }
    return true;
}

bool PagedQwen::read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n,
                            bool drafting, const ChunkDone& on_chunk, std::string& err) {
    if (!activate_images((int) get_part(*parts.front()).id() - 1, err))
        return false;
    conv_.sp.embd_rows = image_row_ptrs_.empty() ? nullptr : image_row_ptrs_.data();
    for (size_t k = 0; k < cards_.size(); ++k)
        cards_[k].sp->set_session(get_part(*parts[k]).ss);
    std::vector<int32_t> next;   // the drafter's: the token after each of the chunk's
    strata::prefill::Prefill& last = *cards_.back().sp;   // which has the head
    last.on_chunk = [&](const float* R_rows, int64_t T, int64_t pos0, std::string& e) -> bool {
        if (drafting) {
            next.assign(ids.begin() + pos0 + 1, ids.begin() + pos0 + T + 1);
            Part& part = get_part(*parts.back());
            if (!drafter_->read_prompt(last, part.pages, part.ss, R_rows, next.data(), T, pos0, (int64_t) ids.size(),
                                       e))
                return false;
        }
        return on_chunk(pos0, T, e);
    };
    const bool ok = conv_.read_prompt(ids.data() + from, n, from, err);
    last.on_chunk = nullptr;
    return ok;
}

bool PagedQwen::run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                         std::string& err) {
    int sessions[strata::kernels::kVerifyMaxT] = {};
    int32_t tokens[strata::kernels::kVerifyMaxT] = {};
    int64_t pos[strata::kernels::kVerifyMaxT] = {};
    const int S = (int) rows.size();
    if (S > strata::kernels::kVerifyMaxT) {
        err = "a step of " + std::to_string(S) + " rows exceeds the verify window";
        return false;
    }
    for (int r = 0; r < S; ++r) {
        const strata::spec::StepRow& row = rows[(size_t) r];
        // the verify windows take a conversation's rows as a chain (Verifier::run_slot_rows), not as a tree
        const bool follows = r > 0 && rows[(size_t) r - 1].session == row.session;
        if (row.parent != (follows ? r - 1 : -1)) {
            err = "strata-q38's verify windows take each conversation's rows as a chain";
            return false;
        }
        sessions[r] = row.session;
        tokens[r] = row.token;
        pos[r] = row.pos;
    }
    if (m_.o.vision) {
        std::vector<int32_t> coordinates((size_t) S * 3);
        for (int r = 0; r < S; ++r) {
            if ((r == 0 || sessions[r] != sessions[r - 1]) && !activate_images(sessions[r], err, false))
                return false;
            const auto& table = leads_[(size_t) sessions[r]]->image_positions;
            for (int axis = 0; axis < 3; ++axis)
                coordinates[(size_t) r * 3 + axis] = table.empty() ? (int32_t) pos[r]
                    : table[(size_t) pos[r] * 3 + axis];
        }
        for (size_t k = 0; k < cards_.size(); ++k) {
            const strata::core::OnDevice on(pcards_[k].dev);
            int32_t* table = k == 0 ? m_.d_mrope : cards_[k].stage->mrope;
            if (!strata::core::gpu_copy(table, coordinates.data(), coordinates.size() * sizeof(int32_t))) {
                err = "the step's image position upload failed";
                return false;
            }
        }
    }
    conv_.ver.set_batch_histories(strata::serve::get_penalty_histories(rows, histories_, history_lengths_));
    if (!conv_.run_rows(sessions, S, tokens, pos, picks.data(), err))
        return false;
    for (int r = 0; r < S; ++r)
        if (conv_.ver.nonfinite_in(r))
            picks[(size_t) r] = -1;
    picks_ = picks;
    return true;
}

bool PagedQwen::keep_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept, bool drafting,
                          std::string& err) {
    std::vector<int> keep((size_t) sessions_, 0);
    for (size_t r = 0, c = 0; r < rows.size(); ++r)
        if (r == 0 || rows[r].session != rows[r - 1].session)
            keep[(size_t) rows[r].session] = kept[c++];
    const bool all = rows.size() == kept.size();   // one row each, kept
    if (!(all ? conv_.ver.commit_slots(err) : conv_.ver.commit_slot_prefixes(keep.data(), err)))
        return false;
    if (drafting)
        drafter_->note_window(rows, kept, picks_.data());
    return true;
}

int PagedQwen::propose(strata::serve::PagedPart& last, int session, int32_t, int64_t, int n, int32_t* out,
                       std::string& err, strata::spec::DraftContext* draft) {
    if (!activate_images(session, err))
        return -1;
    Part& part = get_part(last);
    return drafter_->propose(session, part.pages, part.ss, conv_.ver.final_R_all(), n, out, err, draft);
}

bool PagedQwen::switch_drafter(bool on, std::string& err) {
    if (!on) {
        m_.unload_drafter();
        return true;
    }
    m_.mtp.set_kv_in_pages(true);
    if (!m_.load_drafter(err) || !drafter_->init(m_, sessions_, err)) {
        m_.unload_drafter();
        return false;
    }
    return true;
}

void PagedQwen::print_info(int slots, int spec) const {
    const Options& o = m_.o;
    int64_t resident = 0;
    uint64_t expert_bytes = 0;
    for (size_t k = 0; k < cards_.size(); ++k) {
        resident += cards_[k].cache->resident();
        expert_bytes += pcards_[k].experts.pool().report().experts.bytes;
    }
    std::printf("INFO context=%lld kv=%s expert_slots=%lld expert_cache_mib=%lld spec=%d prefill_cap=%lld%s engine="
                STRATA_VERSION "\n",
                (long long) o.max_context, o.kv.c_str(), (long long) resident, (long long) (expert_bytes >> 20), spec,
                (long long) o.prefill_chunk,
                slots > 0 ? (" batch_slots=" + std::to_string(slots) + " slot_cache=0").c_str() : "");
}

void PagedQwen::print_done(const strata::serve::RequestDone& d) const {
    // the fields of the old engine's DONE that this one has (no expert counts)
    std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld 0 0 0 0 0.0 %lld 0 chunk=%lld experts_streamed=0\n",
                (long long) d.produced, (long long) d.prompt, d.prompt_ms, d.decode_ms, d.finish,
                (long long) d.guessed_right, (long long) d.guessed, (long long) d.resumed, (long long) d.read,
                (long long) m_.o.prefill_chunk);
}

/// Gets why the cards of `o`'s layer split may not compute experts in hops, or an empty
/// string: card 0 runs on GPU 0 (a layer split takes no other primary device) and the later cards on --split-device's
/// GPUs, which without the option are GPUs of their own.
std::string get_split_hop_refusal(const Options& o) {
    std::vector<int64_t> stages;
    std::string err;
    if (o.layer_split.empty() || o.split_device.empty() ||
        !strata::serve::parse_i64_list(o.split_device.c_str(), stages, err))
        return {};
    std::vector<int> devices = {0};
    devices.insert(devices.end(), stages.begin(), stages.end());
    return strata::core::get_hop_refusal(devices);
}

/// Adds the card that runs layers [lb, le) of `m` to `cards` and its model side to `model_cards`: CUDA0's (card 0,
/// over the Model's own cache and session and `conv`'s windows) or the layer split's stage `stage`.  Its link to RAM
/// is measured before its pool takes the memory the measuring copies need, and its speed is predicted from its
/// multiprocessors and clock.
PagedCard& add_card(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, GpuStage* stage) {
    PagedCard& card = cards.emplace_back();
    ModelCard& mc = model_cards.emplace_back();
    card.index = cards.size() - 1;
    card.dev = stage != nullptr ? stage->dev : m.primary_device;
    card.lb = stage != nullptr ? stage->lb : 0;
    card.le = stage != nullptr ? stage->le : m.split_at.empty() ? m.g.n_layers : m.split_at[0];
    const bool shared =
        std::any_of(cards.begin(), cards.end() - 1, [&](const PagedCard& c) { return c.dev == card.dev; });
    card.id = "gpu" + std::to_string(card.dev) + (shared ? "." + std::to_string(card.index) : "");
    mc.stage = stage;
    mc.cache = stage != nullptr ? &stage->cache : &m.xcache;
    mc.own = stage != nullptr ? &stage->ss : &m.ss;
    mc.cs = stage != nullptr ? (void*) stage->stream : m.main_cs;
    mc.ver = stage != nullptr ? &stage->ver : &conv.ver;
    mc.sp = stage != nullptr ? &stage->sp : &conv.sp;
    card.weights_lb = m.stage_weights_trimmed ? card.lb : 0;
    card.weights_le = m.stage_weights_trimmed ? card.le : m.g.n_layers;
    strata::serve::measure_card(card, kProgram, strata::kernels::cpu::expert_layout().max_blob);
    return card;
}

/// Opens `card`'s expert cache (in `mc`) in the pages of its page pool, for at most `max_experts` experts
/// (Model::open_expert_pages, open_stage_pages); with a layer split, for those of the expert profile's first
/// `max_experts` that are of the card's layers, so that the cards hold the experts one GPU would.  The pool takes
/// `gpu_mib` (-1: what get_gpu_room gives, but no more than those experts (the profile's first) and a conversation at
/// --max-context take), in chunks of kChunkBytes; experts leave the pages of `tokens_min` tokens free.  With the
/// drafter (--mtp), the pool of the card that runs it (has_drafter) also keeps each conversation's drafter K/V, and
/// the drafter's head stays out of it.
bool open_pages(Model& m, PagedCard& card, ModelCard& mc, int64_t tokens_min, int64_t gpu_mib, int64_t max_experts,
                std::string& err) {
    const strata::core::OnDevice on(card.dev);
    const std::vector<strata::core::ExpertKey> ranking = get_card_ranking(m, card.lb, card.le, max_experts);
    const bool drafting = has_drafter(m, mc);
    const std::vector<uint64_t> blobs = get_blob_bytes(m);
    const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
    strata::core::StateLayout layout;
    if (!ConversationPages::get_layout(m.g, *mc.own, drafting ? &m.mtp.kv_state() : nullptr, layout, err))
        return false;
    uint64_t most = strata::serve::get_conversation_bytes(layout, m.o.max_context, page);
    for (size_t i = 0; i < ranking.size() && (int64_t) i < max_experts; ++i)
        most += (blobs[(size_t) ranking[i].layer] + page - 1) / page * page;
    uint64_t room = 0;
    const uint64_t keep = get_gpu_keep(m, mc, false);
    const uint64_t pool_b = strata::serve::get_gpu_pool_bytes(card.dev, keep, gpu_mib, most, room);
    const uint64_t chunk = strata::serve::get_chunk_bytes(pool_b, page);
    mc.cache->set_expert_bytes((int64_t) strata::kernels::cpu::expert_layout().max_blob);
    strata::core::PagePoolConfig config;
    config.tokens_min = std::clamp<int64_t>(tokens_min, 0, m.o.max_context);
    if (!mc.cache->open((int64_t) (chunk / page), m.g.n_layers, m.g.n_expert, (int64_t) page, err) ||
        !card.experts.open(*mc.cache, strata::serve::get_gpu_chunk_memory(card.dev), pool_b, layout, config, blobs,
                           m.g.n_expert, max_experts, ranking, err))
        return false;
    strata::serve::say_gpu_pool(kProgram, card, room, chunk);
    return true;
}

/// Opens the RAM's page pool `ram` beside the cards' (open_ram_pool) and gets whether an expert a GPU's pool evicts is
/// copied back to it (`copy_back`): first the experts of the profile that the GPUs' pools do not hold.  Its conversation
/// layout has the whole model's regions, so that a paused conversation is one conversation in it wherever the layer
/// cut is (PagedQwen::get_ram_span).
bool open_ram(Model& m, const PagedCards& cards, const ModelCards& model_cards, const PagedSettings& s,
              strata::core::RamExperts& ram, bool& copy_back, std::string& err) {
    std::vector<strata::core::ExpertKey> profile;
    for (const auto& [l, e] : m.profile_all.empty() ? m.profile : m.profile_all)
        profile.push_back({l, e});
    strata::core::StateLayout layout;
    const int ring = strata::kernels::kVerifyMaxT * (int) m.K;   // the experts of one layer of a step
    return get_card_layout(m, cards.back(), model_cards.back(), 0, m.g.n_layers, layout, err) &&
           strata::serve::open_ram_pool(kProgram, s, m.src, m.o.resident_headroom, ring, layout, cards[0],
                                        get_blob_bytes(m), m.g.n_expert, profile, m.host_res,
                                        m.xcache.device_slot(0), ram, copy_back, err);
}

/// Gets why strata-q38 cannot run the loaded model `m` (empty: it can).
std::string get_unsupported_reason(const Model& m) {
    if (!m.native_pack) return "it needs a native pack (--native), which decodes through verify windows";
    if (m.thits.d_res == nullptr || m.o.expert_cache <= 0)
        return "it needs the device residency table (--expert-profile, --expert-cache and the token graph)";
    return {};
}

/// Sets up each card's prompt path and verify windows over its own session, each handing its rows to the next card's
/// (through hand-offs in page-locked RAM, so that cards without peer access work), and the CPU expert pool's work in
/// the windows going to the card that runs each layer (`split`, which must outlive them).  The Model's stages run in
/// the order of their layers, as the old engine's layer split runs them.
bool init_card_paths(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, SplitDrive& split,
                     std::string& err) {
    const Options& o = m.o;
    const size_t n = cards.size();
    std::vector<float*> hand(n - 1, nullptr);
    if (!alloc_split_handoffs(m.g, strata::kernels::kVerifyMaxT, hand)) {
        err = "the layer split's hand-offs do not fit";
        return false;
    }
    for (size_t k = n; k-- > 0;) {   // a card's prompt path is set up after the next card's, which it hands to
        const PagedCard& card = cards[k];
        ModelCard& mc = model_cards[k];
        mc.hand_in = k == 0 ? nullptr : hand[k - 1];
        mc.hand_out = k + 1 == n ? nullptr : hand[k];
        set_stage(cards, model_cards, k);
        if (k == 0)
            break;
        const GpuStage& st = *mc.stage;
        const strata::core::OnDevice on(card.dev);
        if (!mc.sp->init(st.wt, m.g, *mc.own, m.srcp, mc.cache, m.host_res.data(), o.prefill_chunk, mc.cs, err) ||
            !mc.ver->init(st.wt, m.g, *mc.own, m.get_verify_hits(st), st.head.loaded() ? &st.head : nullptr, o.spec,
                          err)) {
            err = card.id + "'s prompt path or windows: " + err;
            return false;
        }
        mc.ver->set_split(o.spec_split);
        mc.ver->set_pcie_mode(get_pcie_mode_code(o));
    }
    if (!conv.init_prompt(m, m.ss, nullptr, 0, err) || !conv.init_decode(m, m.ss, o.spec, err))
        return false;
    if (n == 1)
        return true;
    split.base = &m.drive;
    split.n = (int) n;
    for (size_t k = 0; k < n; ++k) {
        const ModelCard& mc = model_cards[k];
        split.end[k] = cards[k].le;
        split.plan[k] = mc.ver->plan_sink();
        split.cache_base[k] = mc.stage != nullptr ? mc.cache->device_slot(0) : m.drive.d.cache_base;
        split.cache_slot_off[k] = mc.stage != nullptr ? mc.cache->slot_offsets() : m.drive.d.cache_slot_off;
        split.pcie_num[k] = get_pcie_num(mc.stage != nullptr ? mc.stage->pcie_frac : o.pcie_frac);
        if (k + 1 < n)
            mc.ver->set_next(model_cards[k + 1].ver, &split);
    }
    conv.set_split_drive(&split);
    return true;
}

/// Appends `n` to `bytes` (save_state's sizes).
void append_size(std::vector<uint8_t>& bytes, uint64_t n) {
    const auto* p = reinterpret_cast<const uint8_t*>(&n);
    bytes.insert(bytes.end(), p, p + sizeof n);
}

/// Reads a size append_size wrote at `at` in `bytes` and moves `at` past it.
uint64_t read_size(const std::vector<uint8_t>& bytes, size_t& at) {
    uint64_t n = 0;
    if (at > bytes.size() || bytes.size() - at < sizeof n)
        return UINT64_MAX;
    std::memcpy(&n, bytes.data() + at, sizeof n);
    at += sizeof n;
    return n;
}

}  // namespace
}  // namespace strata::qwen38

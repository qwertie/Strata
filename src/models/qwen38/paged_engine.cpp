// src/models/qwen38/paged_engine.cpp - strata-q38: conversations in a page pool, several decoding at once
// (paged_engine.hpp).
#include "strata/models/qwen38/paged_engine.hpp"

#include "strata/base/gpu/memory_info.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/common/experts/expert_pages.hpp"
#include "strata/common/experts/expert_swaps.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/experts/ram_tier.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/common/serve/loop.hpp"
#include "strata/common/serve/request.hpp"
#include "strata/common/spec/step.hpp"
#include "strata/common/state/pool_copy.hpp"
#include "strata/common/state/progress.hpp"
#include "strata/models/qwen38/conversation.hpp"
#include "strata/models/qwen38/conversation_drafter.hpp"
#include "strata/models/qwen38/conversation_pages.hpp"
#include "strata/models/qwen38/conversation_snapshot.hpp"
#include "strata/models/qwen38/kernels/verify_kernels.hpp"
#include "strata/models/qwen38/model.hpp"
#include "strata/models/qwen38/options.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strata::qwen38 {
namespace {

using Clock = std::chrono::steady_clock;

/// The conversation tokens whose pages experts leave free when --tokens-min is not given.
constexpr int64_t kDefaultTokensMin = 8192;
/// The SSD space paused conversations may take when --ssd-pause-mib is not given.
constexpr int64_t kDefaultSsdPauseMib = 8192;
/// The most evicted experts put back in one step of the serve loop, so that a request waits little for it.
constexpr int kRefillBatch = 16;
/// Between two chunks of a prompt, the conversations that decode take steps for this share of the time the chunk
/// took, so that a long prompt does not stop them (the old engine's STRATA_BATCH_DECODE_SHARE default).
constexpr double kDecodeShare = 0.5;
/// With drafting, the verify windows' graphs kept for the most recent row layouts (the old engine's --batch-mtp).
constexpr size_t kWindowGraphs = 64;
/// How many times faster than reading it from the SSD again copying an evicted expert back from the GPU must be for
/// --copy-back auto to copy it: when the two are close, dropping wins, as the expert may not be needed for a while.
constexpr double kCopyBackFactor = 2.0;
/// The step in which the pools take memory and give it back (PagePoolConfig::chunk_bytes).  An expert may not cross a
/// chunk's end, so each chunk ends in a run too short for any expert: 0.36% of the Strix Halo's GPU pool in chunks of
/// this size (docs/MEMORY_MANAGER_PLAN.md, M2f).
constexpr uint64_t kChunkBytes = 256ull << 20;

/// Gets the milliseconds from `t0` to now.
double get_ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// strata-q38's own options (the rest of its command line is `strata`'s).
struct PagedSettings {
    int64_t tokens_min = kDefaultTokensMin;        ///< --tokens-min
    int64_t tokens_max = -1;                       ///< --tokens-max; -1: --max-context
    int64_t gpu_mib = -1;                          ///< --gpu-mib; -1: what the GPU has free less what it keeps
    int64_t ram_mib = -1;                          ///< --ram-mib; -1: what the RAM has free less the headroom
    int64_t ssd_pause_mib = kDefaultSsdPauseMib;   ///< --ssd-pause-mib
    std::string pause_dir;                         ///< --pause-dir; empty: the temporary folder's strata-q38
    std::string copy_back = "auto";                ///< --copy-back
    bool drafter = true;                           ///< --drafter on|off: its weights loaded (with --lookahead)
    strata::spec::StepSettings step;               ///< --rows, --lookahead and --concurrent (default --batch)
};

/// CHANGE_SETTINGS' fields: names and values, in order.
using SettingValues = std::vector<std::pair<std::string, std::string>>;

/// A place for a conversation on the GPU: its state, in a session whose GDN state and K/V are page-pool pages, and the
/// tokens in it.  Its session is the Verifier's slot session of the same index.
struct Held {
    int session = 0;                    ///< its index: its verify-window session; its id in the GPU's pool - 1
    strata::core::SessionState ss;
    void* mem = nullptr;                ///< `ss`'s memory (without the state the pages hold)
    ConversationPages pages;            ///< given at the conversation's first request
    std::vector<int32_t> live;          ///< the tokens its state holds: the prompt, then the fed outputs
    strata::spec::Decoding dec;         ///< its decoding, while `decoding`
    bool decoding = false;
    int slot = -1;                      ///< while decoding: its batch slot (BT lines), or -1: it is the request on the
                                        ///< control lines (T lines)
    Clock::time_point t0;               ///< when it started decoding
    /// the request count when it last served a request: the oldest goes first, as an idle one to pause and among
    /// decoding ones of the same priority
    uint64_t used = 0;
    /// where its GDN state was when the verify windows' graphs last saw it (their graphs hold the address)
    const float* gdn_seen = nullptr;
    int64_t stepped = 0;                ///< the decode step it last took part in (decode_step's turns)

    /// Gets its place in the order in which conversations give way to others (smallest first): the lowest priority,
    /// then the least recently used.
    std::pair<int64_t, uint64_t> get_yield_order() const { return {dec.priority, used}; }

    ~Held() {
        strata::core::session_release(ss);
        (void) cudaFree(mem);
    }
    /// Forgets its tokens and gives back its growing pages.
    void forget() {
        live.clear();
        if (pages.is_ready())
            pages.truncate(0);
    }
};

/// A conversation paused out of the GPU's pool: its pages in the RAM's pool or in a file on the SSD, and what its
/// session kept beside them.
struct Paused {
    uint64_t serial = 0;              ///< its number (LIST_CONVERSATIONS' parked<serial>) and its id in the RAM's pool
    std::vector<int32_t> live;        ///< the tokens it holds
    std::vector<uint8_t> unpaged;     ///< its session's state beside the pages (conversation_unpaged_save)
    int64_t priority = 0;             ///< its last request's
    uint64_t used = 0;                ///< Held::used
    uint64_t bytes = 0;               ///< its pages' bytes
    std::string file;                 ///< on the SSD: its file; empty: in the RAM's pool
};

/// The engine once its model is up: its conversations, the pages they share with the experts, the step that decodes
/// them, and the requests' protocol.
class PagedEngine {
public:
    /// `ram`: the RAM's pool (experts and paused conversations); `copy_back`: an expert the GPU's pool evicts is
    /// copied to it.  `slots`: the batch slots (--batch; 0: one conversation at a time).  `drafter`: null when
    /// drafting is off.
    PagedEngine(Model& m, Conversation& conv, strata::core::ExpertPages& experts, strata::core::RamExperts& ram,
                bool copy_back, ConversationDrafter* drafter, const PagedSettings& settings, int slots);
    /// Deletes the paused conversations' files.
    ~PagedEngine();

    /// Carves each conversation's session and gives them to the verify windows; false (said on stderr) when one
    /// does not fit.
    bool init();
    /// Answers the server's commands until QUIT or the end of stdin; returns the exit status.
    int serve();

private:
    /// Changes the settings CHANGE_SETTINGS names (memory_report.hpp): the pools' sizes, the token limits, the SSD's
    /// budget, the drafter, the conversations at once, the rows and the lookahead.  Gets why not, having changed
    /// nothing, or an empty string.  A memory that gives less than asked (a failed allocation) is said on stderr only:
    /// the change is made as far as it goes, and the report shows the sizes it reached.
    std::string change_settings(const SettingValues& values);
    /// Takes or gives back chunks of the GPU's pool so that it holds `bytes`.  When the conversations' pages do not fit
    /// the smaller pool, conversations are paused until they do: those that do not decode first (those in the chunks
    /// it gives back before the others), then batch slots' decoding ones in those chunks, the lowest priority first,
    /// which end as preempted.  Gets why not (the memory had less to give), or an empty string.
    std::string resize_gpu(uint64_t bytes);
    /// Takes or gives back chunks of the RAM's pool so that it (and the staging ring) hold `bytes`: the conversations
    /// paused in the chunks it gives back move to the SSD.  Gets why not, or an empty string.
    std::string resize_ram(uint64_t bytes);
    void report_resize(const char* name, const strata::core::PagePoolReport& before,
                       const strata::core::PagePoolReport& after, Clock::time_point t0) const;
    /// Sets the conversation token limits: the GPU pool's minimum, and the maximum's budget of the conversations paused
    /// in RAM (moving those past it to the SSD).  Gets why not, or an empty string.
    std::string set_token_limits(int64_t min, int64_t max);
    /// Loads (`on`) or gives back the drafter's weights, and keeps settings_.drafter to whether they are loaded; false
    /// (and `err`) when loading failed.
    bool switch_drafter(bool on, std::string& err);
    /// Gets the least the GPU's pool may hold now: the pages of a conversation token minimum of `tokens_min`, and
    /// those each decoding conversation takes at the end of its request, in whole chunks.
    uint64_t get_smallest_gpu(int64_t tokens_min) const;
    /// Gets the most the GPU's pool may hold now: what it holds and what the GPU has free beyond what it keeps.
    uint64_t get_largest_gpu() const;
    /// Gets the least and the most the RAM's pool and its staging ring may hold now.
    uint64_t get_smallest_ram() const;
    uint64_t get_largest_ram() const;
    /// Runs one GEN or BGEN line; false on a failure that ends the engine (its ERR has been printed).
    bool serve_request(const std::string& line);
    /// Reads `ids[from, to)` into `h` (the request began at `t0`); `read_to` gets how far it got.  Conversations that
    /// decode meanwhile take steps between the prompt's chunks (kDecodeShare).  False with `err` "cancelled" after a
    /// STOP, else on a failure that ends the engine (`fatal`: a decode step failed, its ERR printed).
    bool read_prompt(Held& h, const std::vector<int64_t>& ids, int64_t from, int64_t to, Clock::time_point t0,
                     int64_t& read_to, bool& fatal, std::string& err);
    /// Runs one step of every decoding conversation and prints what it emitted; false (ERR printed) ends the engine.
    bool decode_step();
    /// Ends the request that batch slot `slot` decodes, so that one of priority `priority` takes the slot, when that is
    /// higher than its own: prints its BDONE with the finish "preempted".  The server sends it again later, its prompt
    /// and the tokens it got so far, which resumes its conversation (take_conversation pauses it).  False when the
    /// request's priority is as high.
    bool preempt(size_t slot, int64_t priority);
    /// Puts the conversation that holds the longest start of `ids` (an idle one on the GPU, or a paused one, which
    /// comes back) in place `target` of `at_`, pausing the one there unless that is it.  Gets how many of `ids` it
    /// holds.
    int64_t take_conversation(size_t target, const std::vector<int64_t>& ids);
    /// Gives `h` the pages for `tokens` tokens, pausing idle conversations (the lowest priority, then the least
    /// recently used, first) while the pool has no room, and tells the kernels about the experts that left those
    /// pages.
    bool make_room(Held& h, int64_t tokens, std::string& err);
    /// Pauses `h`'s conversation, or forgets it when that fails (said on stderr).
    void pause_or_forget(Held& h);
    /// Ends the request that batch slot `slot` decodes, its finish "preempted" (preempt).
    void end_preempted(size_t slot);
    /// Gets the conversation that holds pages, does not decode and goes first (the lowest priority, then the least
    /// recently used), among those `can_go` accepts besides `keep` and the one whose prompt is read; null when none.
    Held* find_idle(const Held* keep, const std::function<bool(const Held&)>& can_go = nullptr);
    /// Gives `h` its fixed region in the GPU's pool, and drops the verify windows' graphs when its GDN state moved.
    bool init_pages(Held& h, std::string& err);
    /// Drops the verify windows' graphs when `h`'s GDN state is elsewhere than when they last saw it.
    bool note_gdn_place(Held& h, std::string& err);
    /// Moves `h`'s conversation out of the GPU's pool into the RAM's (first moving the RAM's paused conversations
    /// down to the SSD while they would take more than --tokens-max allows), or else into a file; `h` holds nothing
    /// after.  False (and `err`) when a copy failed: then `h` keeps its conversation.
    bool pause(Held& h, std::string& err);
    /// Puts paused conversation `p` back in `h`, which holds nothing: its pages, then its session's state.
    bool resume(const Paused& p, Held& h, std::string& err);
    /// Moves the lowest-priority conversation paused in RAM to the SSD (forgetting it when the copy fails); false when
    /// none is there.
    bool move_down();
    /// Moves paused conversation `p`, in RAM, to the SSD (forgetting it when the copy fails).
    void move_to_ssd(std::vector<Paused>::iterator p);
    /// Moves the lowest-priority conversations paused in RAM to the SSD while those in RAM take more than the budget.
    void trim_ram();
    /// Forgets the lowest-priority paused conversations while those on the SSD take more than --ssd-pause-mib.
    void trim_ssd();
    /// Gives back a paused conversation's pages in RAM, or deletes its file.
    void drop_pages(const Paused& p);
    /// Gets the bytes of the paused conversations in RAM (`in_ram`) or on the SSD.
    uint64_t get_paused_bytes(bool in_ram) const;
    /// Gets the bytes the conversations paused in RAM may take: one conversation of --tokens-max tokens.
    uint64_t get_ram_pause_budget() const;
    /// Gets the file on the SSD of the paused conversation of Paused::serial `serial`.
    std::string get_pause_file(uint64_t serial) const;
    /// Gets the paused conversation that goes first, the lowest priority then the least recently used, among those in
    /// RAM (`in_ram`) or on the SSD; paused_.end() when there is none.
    std::vector<Paused>::iterator find_lowest(bool in_ram);
    /// Every --adapt-every steps (the old engine's adaptive tier, with its options): counts a use, in the GPU's pool,
    /// of the experts there that the steps since the last round routed to; then the experts decoding routed to most
    /// that the pool does not hold take the pages of the least-routed ones of their layer (at most --adapt-swaps,
    /// each by --adapt-min-gain more routings).  False (ERR printed) when a copy failed.
    bool adapt_experts();
    /// Puts back a batch of the experts the conversations evicted (the GPU's first, then the RAM's), or waits until
    /// pages open to them: the serve loop's work between requests.  False when a copy failed (said on stderr).
    bool refill_experts();
    /// Uploads the device's residency table when experts came or left since the last upload.
    bool publish_residency(std::string& err);
    /// Ends batch slot `slot`'s decoding at its next step (BSTOP).
    void stop_slot(int slot);
    /// Takes the BSTOP lines that came while a request runs (BYIELD lines are not taken: a read here never gives way).
    void take_slot_controls();
    /// Tells whether any conversation decodes.
    bool is_decoding() const;
    /// Gets the drafter while it is switched on; null when it is off or the engine has none.
    ConversationDrafter* get_drafter() const { return settings_.drafter ? drafter_ : nullptr; }
    /// Gets the most conversations that may decode at once: the batch slots, or 1 without any.
    int get_most_concurrent() const { return (int) std::max<size_t>(at_.size() - 1, 1); }
    /// Gives the step settings_' step settings, the lookahead 0 while the drafter is off.
    void apply_step_settings();
    /// Gets the conversation in place `place` of `at_`.
    Held& get_held(size_t place) const { return *held_[at_[place]]; }
    /// Describes the conversations for LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS.
    std::vector<strata::core::HeldConversation> list_conversations() const;
    /// Describes the memories for GET_MEMORY_USE.
    strata::serve::MemoryReport describe_memory() const;
    /// Gets the step's view of the model: the verify windows over the conversations' sessions.
    strata::spec::StepModel get_step_model();

    Model& m_;
    Conversation& conv_;   ///< the prompt path and the verify windows, shared by every conversation
    strata::core::ExpertPages& experts_;
    strata::core::RamExperts& ram_;
    /// The MTP drafter, whose K/V the conversations' pages have (it can be switched on); null without --lookahead
    ConversationDrafter* const drafter_;
    PagedSettings settings_;
    strata::serve::Input input_;
    strata::spec::Step step_;
    std::vector<std::unique_ptr<Held>> held_;   ///< by session index
    /// held_'s index of the conversation in each batch slot, then of the one a GEN uses ("main")
    std::vector<size_t> at_;
    std::vector<int32_t> picks_;       ///< the last step's token after each row, for the drafter
    std::vector<Paused> paused_;       ///< the conversations paused out of the GPU's pool
    uint64_t serial_ = 0;              ///< the last paused conversation's Paused::serial
    std::string pause_prefix_;         ///< a paused conversation's file is this, its serial and ".bin"
    uint64_t sessions_bytes_ = 0;      ///< the device memory of every session (the engine's own included)
    uint64_t requests_ = 0;            ///< the requests served so far (Held::used)
    const Held* reading_ = nullptr;    ///< the conversation whose prompt read_prompt reads, which make_room keeps
    bool residency_changed_ = false;   ///< `m_.host_res` changed since the device's table was uploaded
    int64_t steps_ = 0;                ///< the decode steps so far (adapt_experts' rounds)
    bool grew_ = false;                ///< a pool grew: decode_step refills experts too, until none are wanted
    std::vector<float> seen_;          ///< `m_.drive.d.usage` when adapt_experts last decayed it
    int64_t swapped_ = 0;              ///< the experts adapt_experts moved into the GPU's pool, since the last report
    double adapt_ms_ = 0;              ///< the time those took
    int64_t copied_back_ = 0;          ///< the experts the GPU's pool evicted that were copied to RAM, since then
    double copy_back_ms_ = 0;          ///< the time the copies back took (all evictions', failed copies too)
};

/// Removes strata-q38's own options from `argv` (the rest is `strata`'s command line).  -1: go on, else exit status.
int take_own_options(int& argc, char** argv, PagedSettings& s) {
    struct Own {
        const char* name;
        int64_t* value;
        std::string* text = nullptr;   ///< instead of `value`, for a word
    };
    int64_t rows = s.step.rows, lookahead = s.step.lookahead, concurrent = -1;
    std::string drafter = "on";
    const Own own[] = {{"--tokens-min", &s.tokens_min},       {"--tokens-max", &s.tokens_max},
                       {"--gpu-mib", &s.gpu_mib},             {"--ram-mib", &s.ram_mib},
                       {"--ssd-pause-mib", &s.ssd_pause_mib}, {"--rows", &rows},
                       {"--lookahead", &lookahead},           {"--concurrent", &concurrent},
                       {"--pause-dir", nullptr, &s.pause_dir}, {"--copy-back", nullptr, &s.copy_back},
                       {"--drafter", nullptr, &drafter}};
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const Own* o = std::find_if(std::begin(own), std::end(own),
                                    [&](const Own& x) { return std::strcmp(argv[i], x.name) == 0; });
        if (o == std::end(own)) {
            argv[out++] = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "strata-q38: %s needs a value\n", o->name);
            return 2;
        }
        if (o->text != nullptr)
            *o->text = argv[++i];
        else
            *o->value = std::atoll(argv[++i]);
    }
    argc = out;
    s.step.rows = (int) std::clamp<int64_t>(rows, 0, INT32_MAX);
    s.step.lookahead = (int) std::clamp<int64_t>(lookahead, -1, INT32_MAX);
    s.step.concurrent = (int) std::clamp<int64_t>(concurrent, -1, INT32_MAX);   // -1: --batch, set once it is read
    s.drafter = drafter == "on";
    if (s.copy_back != "auto" && s.copy_back != "on" && s.copy_back != "off") {
        std::fprintf(stderr, "strata-q38: --copy-back takes auto, on or off\n");
        return 2;
    }
    if (drafter != "on" && drafter != "off") {
        std::fprintf(stderr, "strata-q38: --drafter takes on or off\n");
        return 2;
    }
    return -1;
}

/// Gets the bytes of one expert of each of `m`'s layers.
std::vector<uint64_t> get_blob_bytes(const Model& m) {
    std::vector<uint64_t> blobs;
    for (int64_t l = 0; l < m.g.n_layers; ++l)
        blobs.push_back(strata::kernels::cpu::expert_layout().blob_bytes(l));
    return blobs;
}

uint64_t get_gpu_room(const Model& m, bool within_share);

/// Opens `m`'s expert cache in the pages of the GPU's page pool, for at most `max_experts` experts
/// (Model::open_expert_pages).  The pool takes `gpu_mib` (-1: what get_gpu_room gives, but no more than those
/// experts (the profile's first) and a conversation at --max-context take), in chunks of kChunkBytes; experts leave
/// the pages of `tokens_min` tokens free.  With the drafter (--mtp), each conversation also keeps the drafter's K/V
/// in the pool, and the drafter's head stays out of it.  From here on the experts are read from the files without the
/// file cache.
bool open_pages(Model& m, strata::core::ExpertPages& experts, int64_t tokens_min, int64_t gpu_mib, int64_t max_experts,
                std::string& err) {
    if (std::string why; !m.src.read_unbuffered(why))
        std::fprintf(stderr, "strata-q38: WARNING: the experts are read through the file cache (%s)\n", why.c_str());
    const bool drafting = !m.o.mtp.empty();
    const std::vector<uint64_t> blobs = get_blob_bytes(m);
    const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
    strata::core::StateLayout layout;
    if (!ConversationPages::get_layout(m.g, m.ss, drafting ? &m.mtp.kv_state() : nullptr, layout, err))
        return false;
    uint64_t most = layout.bytes_for(m.o.max_context);
    for (const strata::core::StateRegion& r : layout.regions)   // + each region's last block
        most += strata::core::choose_block_pages(r, page) * page;
    for (size_t i = 0; i < m.profile.size() && (int64_t) i < max_experts; ++i)
        most += (blobs[(size_t) m.profile[i].first] + page - 1) / page * page;
    const uint64_t room = get_gpu_room(m, true);
    const uint64_t pool_b = gpu_mib >= 0 ? std::min((uint64_t) gpu_mib << 20, room) : std::min(room, most);
    const uint64_t chunk = std::max<uint64_t>(std::min(kChunkBytes, pool_b) / page, 1) * page;
    m.xcache.set_expert_bytes((int64_t) strata::kernels::cpu::expert_layout().max_blob);
    std::vector<strata::core::ExpertKey> ranking;
    for (const auto& [layer, expert] : m.profile)
        ranking.push_back({layer, expert});
    strata::core::PagePoolConfig config;
    config.tokens_min = std::clamp<int64_t>(tokens_min, 0, m.o.max_context);
    strata::core::ChunkMemory memory;
    memory.alloc = [](uint64_t n) { return (uint8_t*) strata::core::gpu_alloc_zeroed((size_t) n); };
    memory.free = [](uint8_t* p) { strata::core::gpu_free(p); };
    memory.copy = [](uint8_t* to, const uint8_t* from, uint64_t n) {
        return strata::core::gpu_copy(to, from, (size_t) n);
    };
    if (!m.xcache.open((int64_t) (chunk / page), m.g.n_layers, m.g.n_expert, (int64_t) page, err) ||
        !experts.open(m.xcache, std::move(memory), pool_b, layout, config, blobs, m.g.n_expert, max_experts, ranking,
                      err))
        return false;
    std::fprintf(stderr, "strata-q38: the page pool: %.2f GiB (of %.2f GiB it could take) in chunks of %.0f MiB, "
                         "pages of %llu bytes, for the experts and the conversations, %lld tokens of it kept for the "
                         "conversations\n",
                 (double) experts.pool().report().total.bytes / 1073741824.0, (double) room / 1073741824.0,
                 (double) chunk / 1048576.0, (unsigned long long) page, (long long) config.tokens_min);
    return true;
}

/// Gets the GPU memory a pool may take besides what it holds: what is free less --vram-reserve-mib and the prompt
/// path's own buffers, and the drafter's head when there is a drafter.  `within_share`: on Windows, for a GPU whose
/// memory is the system RAM, no more than is free of the share the BIOS gives it.  That holds only before the RAM's
/// page-locked pool is made, which Windows counts in the share's use too (on the Strix Halo 81 GiB used of a 63.8 GiB
/// share, with a 21.75 GiB RAM pool); what a pool takes beyond the share the server counts against the RAM.
uint64_t get_gpu_room(const Model& m, bool within_share) {
    const uint64_t keep = ((uint64_t) m.o.vram_reserve_mib << 20) +
        strata::prefill::Prefill::bytes_needed_owned(m.g, m.ss, m.o.prefill_chunk, m.srcp != nullptr) +
        (!m.o.mtp.empty() ? m.mtp.bind_bytes(m.native_head.row_bytes(), m.n_vocab) : 0);
    uint64_t free_b = strata::core::device_free_bytes();
    if (uint64_t room = 0; within_share && strata::core::get_share_room(m.primary_device, room))
        free_b = std::min(free_b, room);
    return free_b > keep ? free_b - keep : 0;
}

/// Opens the RAM's page pool `ram` beside the GPU's `experts` and gets whether an expert the GPU's pool evicts is
/// copied back to it (`copy_back`, by --copy-back and the two speeds measured here).  The pool takes --ram-mib, by
/// default what the RAM and the commit limit have free less the resident headroom, but no more than every expert and
/// the staging ring take.  It holds first the experts of the profile the GPU's pool does not, in the profile's order,
/// then copies of the GPU's, then the experts outside the profile, as far as they fit.
bool open_ram(Model& m, const strata::core::ExpertPages& experts, const PagedSettings& s, strata::core::RamExperts& ram,
              bool& copy_back, std::string& err) {
    using strata::core::ExpertKey;
    const std::vector<uint64_t> blobs = get_blob_bytes(m);
    const auto at = [&](int64_t l, int64_t e) { return (size_t) (l * m.g.n_expert + e); };
    std::vector<ExpertKey> ranking, on_gpu;
    std::vector<uint8_t> ranked(blobs.size() * (size_t) m.g.n_expert, 0);
    for (const auto& [l, e] : m.profile) {
        (m.host_res[at(l, e)] >= 0 ? on_gpu : ranking).push_back({l, e});
        ranked[at(l, e)] = 1;
    }
    ranking.insert(ranking.end(), on_gpu.begin(), on_gpu.end());
    uint64_t every = 0;
    for (int32_t l = 0; l < (int32_t) blobs.size(); ++l)
        for (int32_t e = 0; e < (int32_t) m.g.n_expert; ++e) {
            every += blobs[(size_t) l];
            if (!ranked[at(l, e)])
                ranking.push_back({l, e});
        }
    const int ring = strata::kernels::kVerifyMaxT * (int) m.K;   // the experts of one layer of a step
    uint64_t bytes = (uint64_t) s.ram_mib << 20;
    if (s.ram_mib < 0) {
        strata::core::detail::HostMemory hm;
        const uint64_t free_b = strata::core::detail::host_available_memory(hm) ? std::min(hm.available, hm.commit) : 0;
        bytes = std::min(free_b > m.o.resident_headroom ? free_b - m.o.resident_headroom : 0,
                         every + (uint64_t) ring * *std::max_element(blobs.begin(), blobs.end()));
    }
    strata::core::PagePoolConfig config;
    config.page_bytes = experts.pool().page_bytes();
    config.chunk_bytes = kChunkBytes / config.page_bytes * config.page_bytes;
    const Clock::time_point t0 = Clock::now();
    if (!ram.open(m.src, bytes, ring, experts.pool().layout(), config, blobs, m.g.n_expert, ranking, err))
        return false;
    const double fill_s = get_ms_since(t0) / 1000.0;
    const uint64_t held = ram.pages().pool().report().experts.bytes;
    double ssd = 0, back = 0;
    // the best of three: on the Strix Halo single copies measured 3.3 to 57 GB/s, single SSD reads 0.3 to 2.4 GB/s
    for (int round = 0; round < 3; ++round) {
        ssd = std::max(ssd, ram.time_ssd_reads(round));
        back = std::max(back, ram.time_copy_in(m.xcache.device_slot(0), ram.ring_bytes()));
    }
    copy_back = s.copy_back == "on" || (s.copy_back == "auto" && back >= kCopyBackFactor * ssd);
    std::fprintf(stderr, "strata-q38: the RAM's page pool: %.2f GiB%s, %.2f GiB of experts read in %.1f s (%.1f GB/s); "
                         "the other experts are read from the SSD without the file cache.  An expert the GPU evicts "
                         "is %s (copying it back: %.1f GB/s, reading the SSD: %.1f GB/s; --copy-back %s)\n",
                 (double) (ram.pages().pool().report().total.bytes) / 1073741824.0,
                 ram.is_page_locked() ? ", page-locked" : "", (double) held / 1073741824.0,
                 fill_s, fill_s > 0 ? (double) held / 1e9 / fill_s : 0.0,
                 copy_back ? "copied back to RAM" : "dropped", back, ssd, s.copy_back.c_str());
    return true;
}

/// Gets `bytes` rounded to the nearest whole number of `pool`'s chunks (a size given in MiB is seldom one).
uint64_t round_to_chunks(const strata::core::PagePool& pool, uint64_t bytes) {
    return (bytes + pool.chunk_bytes() / 2) / pool.chunk_bytes() * pool.chunk_bytes();
}

/// Gets why strata-q38 cannot run the loaded model `m` (empty: it can).
std::string get_unsupported_reason(const Model& m) {
    if (!m.native_pack) return "it needs a native pack (--native), which decodes through verify windows";
    if (m.multi_gpu || !m.stages.empty()) return "it runs on one GPU (no layer split)";
    if (m.thits.d_res == nullptr || m.o.expert_cache <= 0)
        return "it needs the device residency table (--expert-profile, --expert-cache and the token graph)";
    return {};
}

/// Gets why strata-q38 cannot decode with `s` (empty: it can).
std::string get_step_settings_problem(const strata::spec::StepSettings& s) {
    if (s.rows < 1 || s.rows > strata::kernels::kVerifyMaxT)
        return "--rows must be 1 to " + std::to_string(strata::kernels::kVerifyMaxT);
    if (s.concurrent > s.rows)
        return "the conversations at once (" + std::to_string(s.concurrent) + ") must not exceed the rows (" +
               std::to_string(s.rows) + ")";
    if (s.lookahead < 0)
        return "--lookahead must be 0 or more";
    return {};
}

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // each protocol line reaches the server as it is printed
    PagedSettings settings;
    if (const int status = take_own_options(argc, argv, settings); status >= 0) return status;
    prepare_process(argc, argv);
    prime_gpu_queues();
    Options o;
    bool have_tokens = false;
    if (const int status = parse_options(argc, argv, o, have_tokens); status >= 0) return status;
    if (const int status = apply_device_defaults(o); status >= 0) return status;
    const int slots = std::max(o.batch, 0);
    if (settings.step.concurrent < 0 || settings.step.concurrent > std::max(slots, 1))
        settings.step.concurrent = std::max(slots, 1);
    if (const std::string why = get_step_settings_problem(settings.step); !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    const bool drafting = settings.step.lookahead > 0;
    settings.drafter = settings.drafter && drafting;
    if (drafting && o.mtp.empty()) {
        std::fprintf(stderr, "strata-q38: --lookahead %d needs the MTP drafter (--mtp DIR)\n", settings.step.lookahead);
        return 2;
    }
    // the MTP drafter only, its weights loaded only with a lookahead; no K/V streaming or elastic K/V (the pool
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
    if (o.prefill_chunk <= 0) o.prefill_chunk = 2048;   // open_pages leaves room for the prompt path's own buffers
    if (o.resident_budget > 0 && settings.ram_mib < 0)
        settings.ram_mib = (int64_t) (o.resident_budget >> 20);
    if (settings.tokens_max < 0)
        settings.tokens_max = o.max_context;
    o.mmap_experts = true;
    o.resident_cpu_experts = o.resident_pin = o.resident_soft = false;
    o.resident_budget = 0;
    if (!o.layer_split.empty() || o.expert_cache_per_layer) {
        std::fprintf(stderr, "strata-q38: %s\n", o.expert_cache_per_layer
                                                      ? "--expert-cache-per-layer: its experts share the page pool"
                                                      : "--layer-split: it runs on one GPU");
        return 2;
    }
    strata::core::session_set_pool_state(true);
    Model m(o);
    m.mtp.set_kv_in_pages(true);
    strata::core::ExpertPages experts;
    m.open_expert_pages = [&](int64_t max_experts, std::string& err) {
        return open_pages(m, experts, settings.tokens_min, settings.gpu_mib, max_experts, err);
    };
    if (const int status = m.load(); status != 0) return status;
    if (const int status = m.build_hit_tables(false); status != 0) return status;
    if (const std::string why = get_unsupported_reason(m); !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    m.finish_expert_tiers();
    if (o.adapt_every > 0 && o.adapt_swaps > 0)   // the routings adapt_experts counts
        m.drive.d.usage.assign((size_t) (m.g.n_layers * m.g.n_expert), 0.0f);
    std::string err;
    strata::core::RamExperts ram;
    bool copy_back = false;
    if (!open_ram(m, experts, settings, ram, copy_back, err)) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    m.use_expert_source(ram);
    m.set_prompt_pinned_share();
    m.arm_prompt_cpu_share();
    Conversation conv;
    ConversationDrafter drafter;
    if (!conv.init_prompt(m, m.ss, nullptr, 0, err) || !conv.init_decode(m, m.ss, o.spec, err) ||
        (drafting && !drafter.init(m, slots + 1, err))) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    if (drafting)   // the conversations' guesses make many row layouts: the verify windows keep the recent ones' graphs
        conv.ver.set_batch_graph_limit(kWindowGraphs);
    if (drafting && !settings.drafter)   // loaded for the pool's layout of the drafter's K/V, switched on later
        m.unload_drafter();
    PagedEngine engine(m, conv, experts, ram, copy_back, drafting ? &drafter : nullptr, settings, slots);
    return engine.init() ? engine.serve() : 1;
}

namespace {

PagedEngine::PagedEngine(Model& m, Conversation& conv, strata::core::ExpertPages& experts,
                         strata::core::RamExperts& ram, bool copy_back, ConversationDrafter* drafter,
                         const PagedSettings& settings, int slots)
    : m_(m), conv_(conv), experts_(experts), ram_(ram), drafter_(drafter), settings_(settings),
      step_(get_step_model(), m.o.eos_ids) {
    apply_step_settings();
    for (int i = 0; i <= slots; ++i) {
        held_.push_back(std::make_unique<Held>());
        held_.back()->session = i;
        at_.push_back((size_t) i);
    }
    std::error_code ec;   // a folder that cannot be found or made fails each pause's save, which says so
    const std::filesystem::path dir = settings.pause_dir.empty()
                                          ? std::filesystem::temp_directory_path(ec) / "strata-q38"
                                          : std::filesystem::path(settings.pause_dir);
    std::filesystem::create_directories(dir, ec);
    pause_prefix_ = strata::core::get_pause_file_prefix(dir.string());
    if (const int stale = strata::core::remove_stale_pause_files(dir.string()); stale > 0)
        std::fprintf(stderr, "strata-q38: deleted %d files of paused conversations that engines no longer running "
                             "left in %s\n", stale, dir.string().c_str());
    strata::core::ExpertPages::Hooks hooks;
    hooks.load = [this](strata::core::ExpertKey key, int32_t slot, std::string& err) {
        const uint8_t* blob = m_.srcp->blob_stable(key.layer, key.expert);
        if (blob == nullptr)
            err = "no bytes for layer " + std::to_string(key.layer) + " expert " + std::to_string(key.expert);
        return blob != nullptr &&
               m_.xcache.fill_slot_blocking(slot, blob, err,
                                            (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(key.layer));
    };
    hooks.moved = [this](strata::core::ExpertKey key, int32_t slot) {
        m_.host_res[(size_t) key.layer * (size_t) m_.g.n_expert + (size_t) key.expert] = slot;
        residency_changed_ = true;
    };
    if (copy_back)
        hooks.leaving = [this](strata::core::ExpertKey key, const uint8_t* bytes) {
            const Clock::time_point t0 = Clock::now();
            copied_back_ += ram_.keep_copy(key, bytes);
            copy_back_ms_ += get_ms_since(t0);
        };
    experts_.set_hooks(std::move(hooks));
}

PagedEngine::~PagedEngine() {
    for (const Paused& p : paused_)
        drop_pages(p);
}

bool PagedEngine::init() {
    using namespace strata::core;
    const QsaState* rope = &m_.ss.qsa_states[m_.ss.qsa_primary()];   // every session reads the engine's own table
    const uint64_t bytes = session_bytes(m_.g, m_.o.max_context, m_.K, 0, -1, false);
    sessions_bytes_ = session_bytes(m_.g, m_.o.max_context, m_.K) + held_.size() * bytes;
    std::vector<SessionState*> sessions;
    for (const std::unique_ptr<Held>& h : held_) {
        if (cudaMalloc(&h->mem, bytes) != cudaSuccess ||
            session_init(m_.g, m_.o.max_context, m_.K, h->mem, h->ss, 0, -1, rope) == 0) {
            std::fprintf(stderr, "strata-q38: no room for conversation %zu's session (%.1f MiB)\n", sessions.size(),
                         (double) bytes / 1048576.0);
            return false;
        }
        PleRun& ple = h->ss.ple;   // the engine's PLE weights and buffers, with this conversation's history
        ple = m_.ss.ple;
        ple.token = &h->ss.ple_token;
        ple.prev = h->ss.ple_prev;
        ple.hist = h->ss.ple_hist;
        session_zero(h->ss, m_.g, nullptr, m_.main_cs);
        sessions.push_back(&h->ss);
    }
    std::string err;
    if (cudaStreamSynchronize(m_.main_stream) != cudaSuccess || !conv_.ver.init_slots(sessions, err)) {
        std::fprintf(stderr, "strata-q38: the conversations' sessions: %s\n", err.c_str());
        return false;
    }
    return true;
}

int PagedEngine::serve() {
    const Options& o = m_.o;
    const int slots = (int) at_.size() - 1;
    std::printf("INFO context=%lld kv=%s expert_slots=%lld expert_cache_mib=%lld spec=%d prefill_cap=%lld%s engine="
                STRATA_VERSION "\n",
                (long long) o.max_context, o.kv.c_str(), (long long) m_.xcache.resident(),
                (long long) (experts_.pool().report().experts.bytes >> 20),
                get_drafter() != nullptr ? settings_.step.lookahead + 1 : 0, (long long) o.prefill_chunk,
                slots > 0 ? (" batch_slots=" + std::to_string(slots) + " slot_cache=0").c_str() : "");
    std::printf("READY %lld stop conversations memory preempt settings\n", (long long) o.max_context);
    strata::serve::EngineHooks hooks;
    hooks.is_busy = [this] {
        return is_decoding() || experts_.is_refill_wanted() || ram_.pages().is_refill_wanted();
    };
    hooks.step = [this] { return is_decoding() ? decode_step() : refill_experts(); };
    hooks.stop_slot = [this](int slot) { stop_slot(slot); };
    hooks.list_conversations = [this] { return list_conversations(); };
    hooks.describe_memory = [this] { return describe_memory(); };
    hooks.change_settings = [this](const SettingValues& values) { return change_settings(values); };
    strata::serve::Loop loop(input_, hooks);
    std::string line;
    while (loop.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
            if (!serve_request(line)) return 1;
            continue;
        }
        std::printf("ERR strata-q38 does not take %s (no session files)\n", line.substr(0, line.find(' ')).c_str());
    }
    return loop.is_failed() ? 1 : 0;
}

bool PagedEngine::serve_request(const std::string& line) {
    const Options& o = m_.o;
    strata::serve::RequestLine req;
    strata::serve::Sampling smp;
    std::vector<int64_t> ids;
    std::string err;
    const size_t slots = at_.size() - 1;
    if (!strata::serve::parse_request_line(line, req) || req.max_new < 1 ||
        !parse_i64_list(req.ids.c_str(), ids, err)) {
        std::printf("ERR %s\n", err.empty() ? "malformed request (or max_new < 1)" : err.c_str());
        return true;
    }
    if (req.pictures) {
        std::printf("ERR strata-q38 reads no pictures\n");
        return true;
    }
    if (req.batch && (req.slot < 0 || (size_t) req.slot >= slots)) {
        std::printf("ERR BGEN: no such slot (--batch %zu)\n", slots);
        return true;
    }
    if (!req.batch && slots > 0 &&
        (size_t) std::count_if(held_.begin(), held_.end(), [](const auto& h) { return h->decoding; }) >= slots) {
        std::printf("ERR strata-q38 decodes %zu conversations at once, and they all are\n", slots);
        return true;
    }
    int64_t priority = 0;
    for (const auto& [key, value] : req.keys)
        if (key == "priority")
            priority = std::atoll(value.c_str());
        else
            (void) strata::serve::read_sampling_key(key, value, smp);   // the other engines' keys are not this one's
    const int64_t n = (int64_t) ids.size();
    if (n >= o.max_context) {
        std::printf("ERR the prompt (%lld tokens) does not fit the context (%lld)\n", (long long) n,
                    (long long) o.max_context);
        return true;
    }
    if (req.batch && get_held((size_t) req.slot).decoding && !preempt((size_t) req.slot, priority)) {
        std::printf("ERR BGEN: slot %ld decodes a request of priority %lld, as high as this one's\n", req.slot,
                    (long long) get_held((size_t) req.slot).dec.priority);
        return true;
    }
    input_.clear_stop();   // a STOP between requests is stale
    const Clock::time_point t0 = Clock::now();
    const size_t target = req.batch ? (size_t) req.slot : slots;
    const int64_t resume = take_conversation(target, ids);
    Held& h = get_held(target);
    const int session = h.session;
    h.used = ++requests_;
    h.dec = strata::spec::Decoding{};
    h.dec.priority = priority;
    if (drafter_ != nullptr)
        drafter_->forget(session);
    if (!h.pages.is_ready() && !init_pages(h, err)) {
        std::printf("ERR %s\n", err.c_str());
        return true;
    }
    if (resume == 0) {
        h.forget();
        strata::core::session_zero(h.ss, m_.g, nullptr, m_.main_cs);
        (void) cudaStreamSynchronize(m_.main_stream);
    }
    std::printf("RESUME %lld\n", (long long) resume);
    strata::kernels::SamplerParams sp = get_sampler_params(smp);
    sp.penalty_last_n = 0;   // the penalties need a history per window (the old engine's set_history): not here
    conv_.ver.set_slot_sampling(session, sp);
    const char* finish = "length";
    int64_t read_to = resume;   // the prompt tokens the session holds
    if (!make_room(h, n - 1, err)) {   // refused: the engine stays up
        std::printf("ERR %s\n", err.c_str());
        return true;
    }
    if (n - 1 > resume) {
        bool fatal = false;
        const bool ok = read_prompt(h, ids, resume, n - 1, t0, read_to, fatal, err);
        if (fatal)
            return false;
        if (!ok && err != "cancelled") {
            std::printf("ERR %s\n", err.c_str());
            return false;
        }
        if (!ok) finish = "cancel";
    }
    h.live.assign(ids.begin(), ids.begin() + read_to);
    const double prompt_ms = get_ms_since(t0);
    std::printf("REUSED %lld\n", (long long) resume);
    const Clock::time_point d0 = Clock::now();
    if (read_to == n - 1) {
        h.dec.session = session;
        h.dec.next = (int32_t) ids[(size_t) (n - 1)];
        h.dec.pos = n - 1;
        h.dec.max_new = std::min<int64_t>(req.max_new, o.max_context - n);
        h.decoding = true;
        h.slot = -1;
        h.t0 = d0;
        // a BGEN's request ends with its first token; the slot decodes the rest beside the others
        while (h.decoding && !(req.batch && h.dec.produced > 0)) {
            h.dec.stop |= input_.is_stop_requested();
            take_slot_controls();
            if (!decode_step())
                return false;
        }
        finish = h.decoding ? "length" : h.dec.finish;
    }
    // the fields of the old engine's DONE that this one has (no expert counts)
    std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld 0 0 0 0 0.0 %lld 0 chunk=%lld experts_streamed=0\n",
                (long long) h.dec.produced, (long long) n, prompt_ms, get_ms_since(d0), finish,
                (long long) h.dec.guessed_right, (long long) h.dec.guessed, (long long) resume,
                (long long) (read_to - resume), (long long) o.prefill_chunk);
    if (req.batch) {
        if (h.decoding)
            h.slot = (int) req.slot;
        std::printf("BADM %ld %d\n", req.slot, h.decoding ? 1 : 0);
    }
    return true;
}

bool PagedEngine::read_prompt(Held& h, const std::vector<int64_t>& ids, int64_t from, int64_t to,
                              Clock::time_point t0, int64_t& read_to, bool& fatal, std::string& err) {
    conv_.sp.set_session(h.ss);
    reading_ = &h;
    std::vector<int32_t> next;   // the drafter's: the token after each of the chunk's
    conv_.sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t pos0, std::string& e) -> bool {
        if (ConversationDrafter* drafter = get_drafter()) {
            next.assign(ids.begin() + pos0 + 1, ids.begin() + pos0 + T + 1);
            if (!drafter->read_prompt(conv_.sp, h.pages, h.ss, R_rows, next.data(), T, pos0, (int64_t) ids.size(), e))
                return false;
        }
        read_to = pos0 + T;
        const double ms = get_ms_since(t0);
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) read_to, (long long) ids.size(), ms,
                    ms > 0 ? 1000.0 * (double) (read_to - from) / ms : 0.0);
        strata::core::progress_beat();
        if (!input_.is_stop_requested()) return true;
        e = "cancelled";
        return false;
    };
    // beside conversations that decode, a chunk at a time, and they decode between two chunks
    const int64_t piece = is_decoding() ? std::max<int64_t>(conv_.sp.chunk(), 1) : to - from;
    bool ok = true;
    for (int64_t q = from; ok && q < to; q += piece) {
        const Clock::time_point tq = Clock::now();
        ok = conv_.read_prompt(ids.data() + q, std::min(piece, to - q), q, err);
        const double budget_ms = kDecodeShare * get_ms_since(tq);
        const Clock::time_point td = Clock::now();
        while (ok && q + piece < to && is_decoding() && get_ms_since(td) < budget_ms) {
            take_slot_controls();
            fatal = !decode_step();
            ok = !fatal;
        }
    }
    conv_.sp.on_chunk = nullptr;
    reading_ = nullptr;
    return ok;
}

bool PagedEngine::decode_step() {
    std::vector<Held*> going;
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding)
            going.push_back(h.get());
    // at most `concurrent` of them: the highest priority, then the one that waited longest
    std::stable_sort(going.begin(), going.end(), [](const Held* a, const Held* b) {
        return a->dec.priority != b->dec.priority ? a->dec.priority > b->dec.priority
               : a->stepped != b->stepped         ? a->stepped < b->stepped
                                                  : a->used < b->used;
    });
    if (going.size() > (size_t) settings_.step.concurrent)
        going.resize((size_t) settings_.step.concurrent);
    ++steps_;
    for (Held* h : going)
        h->stepped = steps_;
    std::vector<strata::spec::Decoding*> active;
    for (Held* h : going)
        active.push_back(&h->dec);
    std::string err;
    if (!step_.run(active, err)) {
        std::printf("ERR %s\n", err.c_str());
        return false;
    }
    for (Held* h : going) {
        const strata::spec::Decoding& d = h->dec;
        h->live.insert(h->live.end(), d.fed.begin(), d.fed.end());
        for (const int32_t t : d.emitted)
            if (h->slot < 0)
                std::printf("T %d\n", (int) t);
            else
                std::printf("BT %d %d\n", h->slot, (int) t);
        if (d.finish == nullptr)
            continue;
        h->decoding = false;
        if (drafter_ != nullptr && d.guessed > 0)
            std::fprintf(stderr, "strata-q38: %lld of %lld guesses right (%.0f%%) over %lld tokens\n",
                         (long long) d.guessed_right, (long long) d.guessed,
                         100.0 * (double) d.guessed_right / (double) d.guessed, (long long) d.produced);
        if (swapped_ > 0 || copy_back_ms_ > 0) {
            std::fprintf(stderr, "strata-q38: %lld experts moved into the GPU's pool (%.0f ms in all); %lld it evicted "
                                 "copied back to RAM (%.1f ms, all evictions)\n",
                         (long long) swapped_, adapt_ms_, (long long) copied_back_, copy_back_ms_);
            swapped_ = copied_back_ = 0;
            adapt_ms_ = copy_back_ms_ = 0;
        }
        if (std::strcmp(d.finish, "nonfinite") == 0) {   // #879: its state may carry the poison: start over next time
            std::fprintf(stderr, "strata-q38: non-finite logits (#879) at position %lld: the conversation's next "
                                 "request reads its prompt from token 0\n", (long long) d.pos);
            h->forget();
        }
        if (h->slot >= 0)
            std::printf("BDONE %d %lld %s %.1f\n", h->slot, (long long) d.produced, d.finish, get_ms_since(h->t0));
    }
    strata::core::progress_beat();
    if (grew_ && !experts_.is_refill_wanted() && !ram_.pages().is_refill_wanted())
        grew_ = false;
    return adapt_experts() && (!grew_ || refill_experts());
}

bool PagedEngine::preempt(size_t slot, int64_t priority) {
    Held& h = get_held(slot);
    if (priority <= h.dec.priority)
        return false;
    std::fprintf(stderr, "strata-q38: a request of priority %lld preempts slot %zu's (priority %lld, %lld tokens)\n",
                 (long long) priority, slot, (long long) h.dec.priority, (long long) h.dec.produced);
    end_preempted(slot);
    return true;
}

void PagedEngine::end_preempted(size_t slot) {
    Held& h = get_held(slot);
    h.decoding = false;
    std::printf("BDONE %zu %lld preempted %.1f\n", slot, (long long) h.dec.produced, get_ms_since(h.t0));
}

int64_t PagedEngine::take_conversation(size_t target, const std::vector<int64_t>& ids) {
    const auto get_prefix = [&](const std::vector<int32_t>& live) -> int64_t {
        const size_t n = live.size();
        return n > 0 && n < ids.size() && std::equal(live.begin(), live.end(), ids.begin()) ? (int64_t) n : 0;
    };
    size_t best = target;
    int64_t most = get_prefix(get_held(target).live);
    for (size_t i = 0; i < at_.size(); ++i) {
        const Held& h = get_held(i);
        if (const int64_t prefix = get_prefix(h.live); !h.decoding && prefix > most) {
            best = i;
            most = prefix;
        }
    }
    auto back = paused_.end();
    for (auto p = paused_.begin(); p != paused_.end(); ++p)
        if (const int64_t prefix = get_prefix(p->live); prefix > most) {
            back = p;
            most = prefix;
        }
    if (back == paused_.end())
        std::swap(at_[target], at_[best]);
    // out of the list first: pausing the conversation in its place changes the list
    std::optional<Paused> p;
    if (back != paused_.end()) {
        p = std::move(*back);
        paused_.erase(back);
    }
    Held& h = get_held(target);
    if ((p || most == 0) && !h.live.empty())
        pause_or_forget(h);
    if (!p)
        return most;
    std::string err;
    const bool ok = resume(*p, h, err);
    drop_pages(*p);
    if (ok)
        return most;
    std::fprintf(stderr, "strata-q38: resuming a paused conversation failed, so its prompt is read again: %s\n",
                 err.c_str());
    h.live.clear();
    return 0;
}

bool PagedEngine::make_room(Held& h, int64_t tokens, std::string& err) {
    while (!h.pages.reserve(tokens, err)) {
        Held* idle = find_idle(&h);
        if (idle == nullptr)
            return false;
        pause_or_forget(*idle);
    }
    return publish_residency(err);
}

Held* PagedEngine::find_idle(const Held* keep, const std::function<bool(const Held&)>& can_go) {
    Held* idle = nullptr;
    for (const std::unique_ptr<Held>& x : held_)
        if (x.get() != keep && x.get() != reading_ && !x->decoding && !x->live.empty() && (!can_go || can_go(*x)) &&
            (idle == nullptr || x->get_yield_order() < idle->get_yield_order()))
            idle = x.get();
    return idle;
}

void PagedEngine::pause_or_forget(Held& h) {
    std::string err;
    if (pause(h, err))
        return;
    std::fprintf(stderr, "strata-q38: pausing a conversation failed, so it is forgotten: %s\n", err.c_str());
    h.forget();
}

bool PagedEngine::init_pages(Held& h, std::string& err) {
    return h.pages.init(m_.g, h.ss, experts_.pool(), (strata::core::ConversationId) h.session + 1, err) &&
           note_gdn_place(h, err);
}

bool PagedEngine::note_gdn_place(Held& h, std::string& err) {
    const bool moved = h.gdn_seen != nullptr && h.gdn_seen != h.ss.gdn_state;
    h.gdn_seen = h.ss.gdn_state;
    return !moved || conv_.ver.drop_batch_graphs(err);
}

bool PagedEngine::pause(Held& h, std::string& err) {
    const Clock::time_point t0 = Clock::now();
    Paused p;
    p.serial = ++serial_;
    p.priority = h.dec.priority;
    p.used = h.used;
    if (!strata::core::conversation_unpaged_save(h.ss, m_.g, p.unpaged, err))
        return false;
    strata::core::PagePool& ram = ram_.pages().pool();
    const int64_t tokens = h.pages.get_pool_tokens((int64_t) h.live.size());
    p.bytes = ram.pages_for(tokens) * ram.page_bytes();
    const uint64_t budget = get_ram_pause_budget();
    while (p.bytes <= budget && get_paused_bytes(true) + p.bytes > budget)
        if (!move_down())
            break;
    bool ok = false;
    if (get_paused_bytes(true) + p.bytes <= budget && ram.grow_conversation(p.serial, tokens).empty()) {
        ram.set_paused(p.serial, true);
        ok = strata::core::copy_conversation(experts_.pool(), h.pages.id(), ram, p.serial, err);
    } else {   // the file takes every block the GPU's pool gave the conversation
        p.file = get_pause_file(p.serial);
        p.bytes = h.pages.bytes();
        ok = strata::core::save_conversation(experts_.pool(), h.pages.id(), p.file, err);
    }
    if (!ok) {
        drop_pages(p);
        return false;
    }
    std::fprintf(stderr, "strata-q38: a conversation of %zu tokens is paused %s (%.0f MiB in %.0f ms)\n",
                 h.live.size(), p.file.empty() ? "in RAM" : "on the SSD", (double) p.bytes / 1048576.0,
                 get_ms_since(t0));
    p.live = std::move(h.live);
    h.live.clear();
    h.pages.release();
    paused_.push_back(std::move(p));
    trim_ssd();
    return true;
}

bool PagedEngine::resume(const Paused& p, Held& h, std::string& err) {
    const Clock::time_point t0 = Clock::now();
    if ((!h.pages.is_ready() && !init_pages(h, err)) || !make_room(h, (int64_t) p.live.size(), err))
        return false;
    const strata::core::PagePool& gpu = experts_.pool();
    if (!(p.file.empty() ? strata::core::copy_conversation(ram_.pages().pool(), p.serial, gpu, h.pages.id(), err)
                         : strata::core::load_conversation(p.file, gpu, h.pages.id(), err)) ||
        !strata::core::conversation_unpaged_restore(p.unpaged, h.ss, m_.g, p.live, err))
        return false;
    h.live = p.live;
    std::fprintf(stderr, "strata-q38: a conversation of %zu tokens is back from %s (%.0f ms)\n", p.live.size(),
                 p.file.empty() ? "RAM" : "the SSD", get_ms_since(t0));
    return true;
}

bool PagedEngine::move_down() {
    const auto p = find_lowest(true);
    if (p == paused_.end())
        return false;
    move_to_ssd(p);
    return true;
}

void PagedEngine::move_to_ssd(std::vector<Paused>::iterator p) {
    const std::string file = get_pause_file(p->serial);
    std::string err;
    const bool ok = strata::core::save_conversation(ram_.pages().pool(), p->serial, file, err);
    drop_pages(*p);
    if (ok) {
        p->file = file;
        trim_ssd();
        return;
    }
    std::fprintf(stderr, "strata-q38: moving a paused conversation to the SSD failed, so it is forgotten: %s\n",
                 err.c_str());
    paused_.erase(p);
}

void PagedEngine::trim_ram() {
    while (get_paused_bytes(true) > get_ram_pause_budget() && move_down()) {
    }
}

void PagedEngine::trim_ssd() {
    const uint64_t budget = (uint64_t) settings_.ssd_pause_mib << 20;
    while (get_paused_bytes(false) > budget) {
        const auto p = find_lowest(false);
        std::fprintf(stderr, "strata-q38: a paused conversation of %zu tokens is forgotten (--ssd-pause-mib %lld)\n",
                     p->live.size(), (long long) settings_.ssd_pause_mib);
        drop_pages(*p);
        paused_.erase(p);
    }
}

void PagedEngine::drop_pages(const Paused& p) {
    std::error_code ec;
    if (!p.file.empty())
        std::filesystem::remove(p.file, ec);
    else
        ram_.pages().pool().remove_conversation(p.serial);
}

uint64_t PagedEngine::get_paused_bytes(bool in_ram) const {
    uint64_t bytes = 0;
    for (const Paused& p : paused_)
        if (p.file.empty() == in_ram)
            bytes += p.bytes;
    return bytes;
}

uint64_t PagedEngine::get_ram_pause_budget() const {
    const strata::core::PagePool& ram = ram_.pages().pool();
    return ram.pages_for(settings_.tokens_max) * ram.page_bytes();
}

std::string PagedEngine::get_pause_file(uint64_t serial) const {
    return pause_prefix_ + std::to_string(serial) + ".bin";
}

std::vector<Paused>::iterator PagedEngine::find_lowest(bool in_ram) {
    auto lowest = paused_.end();
    for (auto p = paused_.begin(); p != paused_.end(); ++p)
        if (p->file.empty() == in_ram &&
            (lowest == paused_.end() ||
             std::make_pair(p->priority, p->used) < std::make_pair(lowest->priority, lowest->used)))
            lowest = p;
    return lowest;
}

bool PagedEngine::adapt_experts() {
    const Options& o = m_.o;
    std::vector<float>& usage = m_.drive.d.usage;
    if (usage.empty() || steps_ % o.adapt_iv(steps_) != 0)
        return true;
    const Clock::time_point t0 = Clock::now();
    const int64_t n_expert = m_.g.n_expert;
    seen_.resize(usage.size(), 0.0f);
    for (size_t i = 0; i < usage.size(); ++i)
        if (usage[i] > seen_[i] && m_.host_res[i] >= 0)
            experts_.note_use({(int32_t) ((int64_t) i / n_expert), (int32_t) ((int64_t) i % n_expert)});
    std::vector<strata::core::ExpertSwap> swaps = strata::core::choose_expert_swaps(
        usage, m_.host_res, m_.g.n_layers, n_expert, o.adapt_min_gain, [](int64_t, int32_t) { return false; });
    if (swaps.size() > (size_t) o.adapt_swaps)
        swaps.resize((size_t) o.adapt_swaps);
    std::string err;
    for (const strata::core::ExpertSwap& s : swaps)
        if (!experts_.replace(s.layer, s.out, s.in, err)) {
            std::printf("ERR moving an expert into the GPU's pool: %s\n", err.c_str());
            return false;
        }
    for (float& u : usage)
        u *= o.adapt_decay;
    seen_ = usage;
    if (!publish_residency(err)) {
        std::printf("ERR %s\n", err.c_str());
        return false;
    }
    swapped_ += (int64_t) swaps.size();
    adapt_ms_ += get_ms_since(t0);
    return true;
}

bool PagedEngine::refill_experts() {
    std::string err;
    uint64_t wait_ms = 0;
    int loaded = experts_.is_refill_wanted() ? experts_.refill(kRefillBatch, wait_ms, err) : 0;
    if (loaded == 0 && ram_.pages().is_refill_wanted()) {   // the GPU's first: its experts are the hotter ones
        uint64_t ram_wait_ms = 0;
        loaded = ram_.pages().refill(kRefillBatch, ram_wait_ms, err);
        if (ram_wait_ms > 0 && (wait_ms == 0 || ram_wait_ms < wait_ms))
            wait_ms = ram_wait_ms;
    }
    if (loaded < 0 || !publish_residency(err)) {
        std::fprintf(stderr, "strata-q38: putting experts back in their pools: %s\n", err.c_str());
        return false;
    }
    if (wait_ms > 0)
        input_.wait_for_line(std::chrono::milliseconds(wait_ms), [] { return false; });
    return true;
}

bool PagedEngine::publish_residency(std::string& err) {
    if (!residency_changed_)
        return true;
    if (const cudaError_t e = m_.put_residency(m_.d_res); e != cudaSuccess) {
        err = std::string("the expert residency table: ") + cudaGetErrorString(e);
        return false;
    }
    residency_changed_ = false;
    return true;
}

void PagedEngine::stop_slot(int slot) {
    if (slot < 0 || slot + 1 >= (int) at_.size())
        return;
    Held& h = get_held((size_t) slot);
    if (h.decoding && h.slot == slot)
        h.dec.stop = true;
}

void PagedEngine::take_slot_controls() {
    input_.take_slot_controls([this](int slot) { stop_slot(slot); }, [](int) { return false; });
}

bool PagedEngine::is_decoding() const {
    return std::any_of(held_.begin(), held_.end(), [](const std::unique_ptr<Held>& h) { return h->decoding; });
}

std::vector<strata::core::HeldConversation> PagedEngine::list_conversations() const {
    std::vector<strata::core::HeldConversation> list;
    for (size_t i = 0; i < at_.size(); ++i) {
        const size_t place = (i + at_.size() - 1) % at_.size();   // "main" first, then the slots
        const Held& h = get_held(place);
        strata::core::HeldConversation c;
        c.id = place + 1 == at_.size() ? "main" : "slot" + std::to_string(place);
        c.state = h.decoding ? "decoding" : h.live.empty() ? "empty" : "holding";
        c.ids = h.live.data();
        c.n = h.live.size();
        c.capacity = (uint64_t) m_.o.max_context;
        c.bytes = c.used = h.pages.bytes();
        list.push_back(c);
    }
    for (const Paused& p : paused_) {
        strata::core::HeldConversation c;
        c.id = "parked" + std::to_string(p.serial);
        c.place = p.file.empty() ? "ram" : "disk";
        c.state = "parked";
        c.ids = p.live.data();
        c.n = c.capacity = p.live.size();
        c.bytes = c.used = p.bytes;
        list.push_back(c);
    }
    return list;
}

strata::serve::MemoryReport PagedEngine::describe_memory() const {
    const strata::core::PagePoolReport r = experts_.pool().report();
    const strata::core::PagePoolReport rr = ram_.pages().pool().report();
    strata::serve::MemoryReport report;
    strata::serve::Memory& gpu = report.gpu(m_.primary_device);
    strata::core::read_gpu_memory(m_.primary_device, gpu.name, gpu.total, gpu.free, gpu.unified);
    gpu.add("weights", m_.pool_bytes + m_.native_dense.allocated_bytes() + m_.native_head.weight_bytes());
    gpu.add("experts", r.experts.bytes);
    gpu.add("conversations", r.active.bytes + r.fixed.bytes);
    gpu.add("pool_free", r.free.bytes);
    gpu.add("scratch", sessions_bytes_);
    gpu.pool_pages = r.total.pages;
    gpu.allocated_pages = r.allocated.pages;
    gpu.fragmented_pages = r.fragmented.pages;
    gpu.setting = r.total.bytes;
    gpu.smallest = get_smallest_gpu(settings_.tokens_min);
    gpu.largest = get_largest_gpu();
    strata::serve::Memory& ram = report.at("ram");
    ram.add("experts", rr.experts.bytes);
    ram.add("pool_free", rr.free.bytes);
    ram.add("scratch", ram_.ring_bytes());
    ram.paused_budget = get_ram_pause_budget();
    ram.pool_pages = rr.total.pages;
    ram.allocated_pages = rr.allocated.pages;
    ram.fragmented_pages = rr.fragmented.pages;
    ram.setting = rr.total.bytes + ram_.ring_bytes();
    ram.smallest = get_smallest_ram();
    ram.largest = get_largest_ram();
    uint64_t on_ssd = 0;   // the experts neither pool holds
    for (int64_t l = 0; l < m_.g.n_layers; ++l)
        for (int64_t e = 0; e < m_.g.n_expert; ++e)
            if (m_.host_res[(size_t) (l * m_.g.n_expert + e)] < 0 && !ram_.holds(l, e))
                on_ssd += strata::kernels::cpu::expert_layout().blob_bytes(l);
    strata::serve::Memory& ssd = report.at("ssd");
    ssd.add("experts", on_ssd);
    ssd.paused_budget = (uint64_t) settings_.ssd_pause_mib << 20;
    const strata::core::PagePool& pool = experts_.pool();
    const int64_t probe = std::max<int64_t>(m_.o.max_context, 1);   // tokens whose pages give the average
    const uint64_t fixed = pool.pages_for(0) * pool.page_bytes();
    report.settings = {{"context", m_.o.max_context},
                       {"concurrent", settings_.step.concurrent},
                       {"slots", get_most_concurrent()},
                       {"rows", settings_.step.rows},
                       {"lookahead", settings_.step.lookahead},
                       {"tokens_min", settings_.tokens_min},
                       {"tokens_max", settings_.tokens_max},
                       {"ssd_pause_mib", settings_.ssd_pause_mib},
                       {"memory_mib.gpu" + std::to_string(m_.primary_device), (int64_t) (r.total.bytes >> 20)},
                       {"memory_mib.ram", (int64_t) ((rr.total.bytes + ram_.ring_bytes()) >> 20)},
                       {"token_bytes", (int64_t) ((pool.pages_for(probe) * pool.page_bytes() - fixed) / probe)},
                       {"conversation_bytes", (int64_t) fixed}};
    if (drafter_ != nullptr)
        report.settings.emplace_back("drafter", settings_.drafter ? 1 : 0);
    report.per_conversation = true;
    return report;
}

std::string PagedEngine::change_settings(const SettingValues& values) {
    PagedSettings s = settings_;
    const std::string gpu_name = "memory_mib.gpu" + std::to_string(m_.primary_device);
    std::optional<uint64_t> gpu_bytes, ram_bytes;
    for (const auto& [name, text] : values) {
        if (name == "drafter") {
            if (text != "on" && text != "off")
                return "drafter takes on or off, not " + text;
            s.drafter = text == "on";
            continue;
        }
        char* end = nullptr;
        const long long v = std::strtoll(text.c_str(), &end, 10);
        if (text.empty() || *end != '\0' || v < 0 || v > INT32_MAX)
            return name + " takes a whole number from 0 to " + std::to_string(INT32_MAX) + ", not " + text;
        if (name == gpu_name)
            gpu_bytes = round_to_chunks(experts_.pool(), (uint64_t) v << 20);
        else if (name == "memory_mib.ram")
            ram_bytes = round_to_chunks(ram_.pages().pool(), std::max((uint64_t) v << 20, ram_.ring_bytes()) -
                                                              ram_.ring_bytes()) + ram_.ring_bytes();
        else if (name == "tokens_min")
            s.tokens_min = v;
        else if (name == "tokens_max")
            s.tokens_max = v;
        else if (name == "ssd_pause_mib")
            s.ssd_pause_mib = v;
        else if (name == "concurrent")
            s.step.concurrent = (int) v;
        else if (name == "rows")
            s.step.rows = (int) v;
        else if (name == "lookahead")
            s.step.lookahead = (int) v;
        else
            return "strata-q38 has no setting " + name;
    }
    // every check first, so that a refusal changes nothing
    const auto gib = [](uint64_t b) {
        char text[32];
        std::snprintf(text, sizeof text, "%.2f GiB", (double) b / 1073741824.0);
        return std::string(text);
    };
    const int slots = get_most_concurrent();
    if (s.step.concurrent < 1 || s.step.concurrent > slots)
        return "the conversations at once may be 1 to " + std::to_string(slots) +
               " (the batch slots it started with; more need a restart with a larger --batch)";
    if (const std::string why = get_step_settings_problem(s.step); !why.empty())
        return why;
    if (s.drafter && drafter_ == nullptr)
        return "this engine has no drafter: it needs --mtp and a --lookahead above 0 at its start";
    if (s.tokens_min > s.tokens_max)
        return "the conversation token minimum (" + std::to_string(s.tokens_min) + ") is above the maximum (" +
               std::to_string(s.tokens_max) + ")";
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding && (int64_t) h->live.size() > s.tokens_max)
            return "a conversation decoding now holds " + std::to_string(h->live.size()) +
                   " tokens, above the maximum asked for";
    const uint64_t gpu_now = experts_.pool().report().total.bytes, smallest = get_smallest_gpu(s.tokens_min);
    if (gpu_bytes.value_or(gpu_now) < smallest)
        return "the biggest active conversation and the token minimum need " + gib(smallest) +
               "; the GPU's pool would keep " + gib(gpu_bytes.value_or(gpu_now));
    if (gpu_bytes && *gpu_bytes > get_largest_gpu())
        return "the GPU has room for " + gib(get_largest_gpu()) + " of pool, not " + gib(*gpu_bytes);
    if (ram_bytes && (*ram_bytes < get_smallest_ram() || *ram_bytes > get_largest_ram()))
        return "the RAM's pool may be " + gib(get_smallest_ram()) + " to " + gib(get_largest_ram()) + ", not " +
               gib(*ram_bytes);
    // then the changes: those that free memory first, so that those taking memory find it; a higher token minimum
    // waits for the GPU's pool to grow, as the pool may not hold it before
    const int64_t min_first = std::min(s.tokens_min, settings_.tokens_min);
    if (const std::string why = set_token_limits(min_first, s.tokens_max); !why.empty())
        return why;
    std::string err;
    if (!s.drafter && settings_.drafter)
        (void) switch_drafter(false, err);
    settings_.ssd_pause_mib = s.ssd_pause_mib;
    trim_ssd();
    settings_.step = s.step;
    std::string short_of;   // what a memory could not give
    const auto note = [&short_of](const char* what, const std::string& why) {
        if (!why.empty())
            short_of += (short_of.empty() ? "" : "; ") + (what + why);
    };
    const uint64_t ram_now = ram_.pages().pool().report().total.bytes + ram_.ring_bytes();
    const auto resize = [&](bool shrink) {
        if (gpu_bytes && *gpu_bytes != gpu_now && (*gpu_bytes < gpu_now) == shrink)
            note("the GPU's pool: ", resize_gpu(*gpu_bytes));
        if (ram_bytes && *ram_bytes != ram_now && (*ram_bytes < ram_now) == shrink)
            note("the RAM's pool: ", resize_ram(*ram_bytes));
    };
    resize(true);
    if (s.drafter && !settings_.drafter && !switch_drafter(true, err))
        note("", err);
    resize(false);
    if (s.tokens_min != min_first)
        note("", set_token_limits(s.tokens_min, s.tokens_max));
    apply_step_settings();
    // the report the loop answers with shows what the memories gave; the shortfall is said here
    if (!short_of.empty())
        std::fprintf(stderr, "strata-q38: CHANGE_SETTINGS changed less than asked: %s\n", short_of.c_str());
    return {};
}

std::string PagedEngine::resize_gpu(uint64_t bytes) {
    strata::core::PagePool& pool = experts_.pool();
    const Clock::time_point t0 = Clock::now();
    const strata::core::PagePoolReport before = pool.report();
    const uint64_t keep = bytes / pool.chunk_bytes() * pool.chunk_bytes() / pool.page_bytes();   // its pages after
    // the conversations that do not decode and would not fit in the smaller pool at all
    while (Held* h = find_idle(nullptr, [&](const Held& x) { return x.pages.bytes() / pool.page_bytes() > keep; }))
        pause_or_forget(*h);
    for (;;) {
        const strata::core::PoolResize r = experts_.resize(bytes);
        if (r.refusal.empty())
            break;
        if (r.to_pause.empty())
            return r.refusal;
        const auto in_the_way = [&](const Held& x) {
            return std::find(r.to_pause.begin(), r.to_pause.end(), (strata::core::ConversationId) x.session + 1) !=
                   r.to_pause.end();
        };
        // pausing any conversation makes room for the ones in the way: those that do not decode first
        if (Held* h = find_idle(nullptr, in_the_way); h != nullptr || (h = find_idle(nullptr)) != nullptr) {
            pause_or_forget(*h);
            continue;
        }
        Held* slot = nullptr;   // a batch slot's decoding conversation, the lowest priority
        for (size_t i = 0; i + 1 < at_.size(); ++i)
            if (Held& h = get_held(i);
                h.decoding && in_the_way(h) && (slot == nullptr || h.get_yield_order() < slot->get_yield_order()))
                slot = &h;
        if (slot == nullptr)
            return r.refusal;
        end_preempted((size_t) slot->slot);
        pause_or_forget(*slot);
    }
    grew_ = grew_ || pool.report().total.pages > before.total.pages;
    std::string err;
    for (const std::unique_ptr<Held>& h : held_)   // a shrink moved conversation blocks
        if (h->pages.is_ready() && (!h->pages.refresh(h->ss, err) || !note_gdn_place(*h, err)))
            return err;
    if (!publish_residency(err))
        return err;
    report_resize("GPU", before, pool.report(), t0);
    return {};
}

std::string PagedEngine::resize_ram(uint64_t bytes) {
    const Clock::time_point t0 = Clock::now();
    const strata::core::PagePoolReport before = ram_.pages().pool().report();
    for (;;) {
        const strata::core::PoolResize r = ram_.resize(bytes);
        if (r.refusal.empty())
            break;
        const auto p = std::find_if(paused_.begin(), paused_.end(), [&](const Paused& x) {
            return x.file.empty() &&
                   std::find(r.to_pause.begin(), r.to_pause.end(), (strata::core::ConversationId) x.serial) !=
                       r.to_pause.end();
        });
        if (p == paused_.end())
            return r.refusal;
        move_to_ssd(p);
    }
    grew_ = grew_ || ram_.pages().pool().report().total.pages > before.total.pages;
    trim_ram();
    report_resize("RAM", before, ram_.pages().pool().report(), t0);
    return {};
}

/// Says on stderr how the pool of memory `name` changed, from `before` to `after`, in the time since `t0`.
void PagedEngine::report_resize(const char* name, const strata::core::PagePoolReport& before,
                                const strata::core::PagePoolReport& after, Clock::time_point t0) const {
    const auto gib = [](const strata::core::PagePoolReport::Use& u) { return (double) u.bytes / 1073741824.0; };
    std::fprintf(stderr, "strata-q38: the %s's pool: %.2f -> %.2f GiB in %.0f ms (experts %.2f -> %.2f GiB)\n", name,
                 gib(before.total), gib(after.total), get_ms_since(t0), gib(before.experts), gib(after.experts));
}

std::string PagedEngine::set_token_limits(int64_t min, int64_t max) {
    // the conversations that do not decode and hold more tokens than the new maximum
    while (Held* h = find_idle(nullptr, [&](const Held& x) { return (int64_t) x.live.size() > max; }))
        pause_or_forget(*h);
    strata::core::PagePool& pool = experts_.pool();
    if (const std::string why = pool.set_token_limits(min, pool.tokens_max()); !why.empty())
        return why;
    settings_.tokens_min = min;
    settings_.tokens_max = max;
    trim_ram();
    return {};
}

bool PagedEngine::switch_drafter(bool on, std::string& err) {
    if (!on) {
        m_.unload_drafter();
        settings_.drafter = false;
        std::fprintf(stderr, "strata-q38: the drafter is off, its memory given back\n");
        return true;
    }
    m_.mtp.set_kv_in_pages(true);
    if (!m_.load_drafter(err) || !drafter_->init(m_, (int) held_.size(), err)) {
        m_.unload_drafter();
        err = "the drafter could not be loaded: " + err;
        return false;
    }
    for (const std::unique_ptr<Held>& h : held_)
        drafter_->forget(h->session);
    settings_.drafter = true;
    std::fprintf(stderr, "strata-q38: the drafter is on\n");
    return true;
}

void PagedEngine::apply_step_settings() {
    step_.settings = settings_.step;
    if (get_drafter() == nullptr)
        step_.settings.lookahead = 0;
}

uint64_t PagedEngine::get_smallest_gpu(int64_t tokens_min) const {
    const strata::core::PagePool& pool = experts_.pool();
    uint64_t pages = pool.pages_for(tokens_min);
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding) {
            const int64_t end = h->dec.pos + h->dec.max_new - h->dec.produced + settings_.step.lookahead + 1;
            pages = std::max(pages, pool.pages_for(h->pages.get_pool_tokens(std::min(end, m_.o.max_context))));
        }
    const uint64_t chunk = pool.chunk_bytes();
    return std::max<uint64_t>((pages * pool.page_bytes() + chunk - 1) / chunk, 1) * chunk;
}

uint64_t PagedEngine::get_largest_gpu() const {
    const strata::core::PagePool& pool = experts_.pool();
    return (pool.report().total.bytes + get_gpu_room(m_, false)) / pool.chunk_bytes() * pool.chunk_bytes();
}

uint64_t PagedEngine::get_smallest_ram() const {
    return ram_.pages().pool().chunk_bytes() + ram_.ring_bytes();
}

uint64_t PagedEngine::get_largest_ram() const {
    strata::core::detail::HostMemory hm;
    const uint64_t free_b = strata::core::detail::host_available_memory(hm) ? std::min(hm.available, hm.commit) : 0;
    const strata::core::PagePool& pool = ram_.pages().pool();
    const uint64_t chunk = pool.chunk_bytes();
    return (pool.report().total.bytes + (free_b > m_.o.resident_headroom ? free_b - m_.o.resident_headroom : 0)) /
               chunk * chunk + ram_.ring_bytes();
}

strata::spec::StepModel PagedEngine::get_step_model() {
    strata::spec::StepModel model;
    model.make_room = [this](int session, int64_t tokens) {
        std::string err;
        if (make_room(*held_[(size_t) session], tokens, err))
            return true;
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return false;
    };
    model.run = [this](const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
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
        if (!conv_.run_rows(sessions, S, tokens, pos, picks.data(), err))
            return false;
        for (int r = 0; r < S; ++r)
            if (conv_.ver.nonfinite_in(r))
                picks[(size_t) r] = -1;
        picks_ = picks;
        return true;
    };
    model.keep = [this](const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                        std::string& err) {
        std::vector<int> keep(held_.size(), 0);
        for (size_t r = 0, c = 0; r < rows.size(); ++r)
            if (r == 0 || rows[r].session != rows[r - 1].session)
                keep[(size_t) rows[r].session] = kept[c++];
        const bool all = rows.size() == kept.size();   // one row each, kept
        if (!(all ? conv_.ver.commit_slots(err) : conv_.ver.commit_slot_prefixes(keep.data(), err)))
            return false;
        if (ConversationDrafter* drafter = get_drafter())
            drafter->note_window(rows, kept, picks_.data());
        return true;
    };
    if (drafter_ != nullptr)
        model.propose = [this](int session, int32_t, int64_t pos, int n, int32_t* out) {
            ConversationDrafter* drafter = get_drafter();
            if (drafter == nullptr)
                return 0;
            Held& h = *held_[(size_t) session];
            std::string err;
            if (n > 0 && !make_room(h, pos + 1 + n, err))   // (the drafter writes the guesses' cells)
                n = 0;
            const int got = drafter->propose(session, h.pages, h.ss, conv_.ver.final_R_all(), n, out, err);
            if (got >= 0)
                return got;
            std::fprintf(stderr, "strata-q38: the drafter: %s\n", err.c_str());
            return 0;
        };
    return model;
}

}  // namespace
}  // namespace strata::qwen38

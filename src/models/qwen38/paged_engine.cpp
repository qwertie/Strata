// src/models/qwen38/paged_engine.cpp - strata-q38: conversations in a page pool, several decoding at once
// (paged_engine.hpp).
#include "strata/models/qwen38/paged_engine.hpp"

#include "strata/base/gpu/links.hpp"
#include "strata/base/gpu/memory_info.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/env.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/expert_pages.hpp"
#include "strata/common/experts/expert_swaps.hpp"
#include "strata/common/experts/layer_split.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/experts/ram_tier.hpp"
#include "strata/common/experts/split_rules.hpp"
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
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace strata::qwen38 {
namespace {

using Clock = std::chrono::steady_clock;

/// The conversation tokens whose pages experts leave free when --tokens-min is not given.
constexpr int64_t kDefaultTokensMin = 8192;
/// The SSD space paused conversations may take when --ssd-pause-mib is not given.
constexpr int64_t kDefaultSsdPauseMib = 8192;
/// The lookahead when --mtp is given and --lookahead is not: on the Strix Halo one conversation decoded fastest with 3
/// guesses a step (4 rows) or 2, and several conversations with 8 rows (docs/DETAILS.md, rows per step).
constexpr int kDefaultLookahead = 3;
/// The most evicted experts put back in one step of the serve loop, so that a request waits little for it.
constexpr int kRefillBatch = 16;
/// The time the experts put back into the GPU's pool in one step of the serve loop may take on its link: a slow link
/// puts back fewer (get_refill_batch).
constexpr double kRefillBudgetMs = 10.0;
/// The milliseconds the CPU pool takes for one routed expert of a decode step, from RAM, against which a hop to
/// another card is weighed (--remote-experts auto, card_rules' is_hop_cheaper): the Strix Halo's pool computed 70,349
/// experts in 2,449 ms and 43,808 in 1,550 ms (two stages, 6 requests each, the reads from the SSD not counted).
constexpr double kCpuExpertMs = 0.035;
/// The bytes each way that measure the GPU's link to RAM at start.
constexpr size_t kLinkProbeBytes = 64ull << 20;
/// Between two chunks of a prompt, the conversations that decode take steps for this share of the time the chunk
/// took, so that a long prompt does not stop them (the old engine's STRATA_BATCH_DECODE_SHARE default).
constexpr double kDecodeShare = 0.5;
/// With drafting, the verify windows' graphs kept for the most recent row layouts (the old engine's --batch-mtp).
constexpr size_t kWindowGraphs = 64;
/// The step in which the pools take memory and give it back (PagePoolConfig::chunk_bytes).  An expert may not cross a
/// chunk's end, so each chunk ends in a run too short for any expert: 0.36% of the Strix Halo's GPU pool in chunks of
/// this size (docs/MEMORY_MANAGER_PLAN.md, M2f).
constexpr uint64_t kChunkBytes = 256ull << 20;
/// The share of Windows' commit limit (RAM and page file) that the pools' default sizes leave uncommitted, and the
/// least they leave: the engine's other allocations and the other programs need it.  On the Strix Halo the GPU's
/// memory counts in the commit charge too (8 GiB of hipMalloc raised it by 8.3 GiB), and pools sized to the commit
/// left less 4 GiB took Windows to its limit.
constexpr double kCommitMarginShare = 0.10;
constexpr uint64_t kCommitMarginMin = 8ull << 30;

/// Gets the milliseconds from `t0` to now.
double get_ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// strata-q38's own options (the rest of its command line is `strata`'s).
struct PagedSettings {
    int64_t tokens_min = kDefaultTokensMin;        ///< --tokens-min
    int64_t tokens_max = -1;                       ///< --tokens-max; -1: --max-context
    /// --gpu-mib, per card of a layer split (one value: every card's); empty: what each GPU has free less what it keeps
    std::vector<int64_t> gpu_mib;
    int64_t ram_mib = -1;                          ///< --ram-mib; -1: get_ram_room
    int64_t ssd_pause_mib = kDefaultSsdPauseMib;   ///< --ssd-pause-mib
    std::string pause_dir;                         ///< --pause-dir; empty: the temporary folder's strata-q38
    std::string copy_back = "auto";                ///< --copy-back
    std::string remote_experts = "auto";           ///< --remote-experts
    bool drafter = true;                           ///< --drafter on|off: its weights loaded (with --lookahead)
    strata::spec::StepSettings step;               ///< --rows, --lookahead and --concurrent (default --batch)
};

/// CHANGE_SETTINGS' fields: names and values, in order.
using SettingValues = std::vector<std::pair<std::string, std::string>>;

/// One card of a layer split, or the GPU without one: the layers it runs, its page pool with their experts and the
/// conversations' state for them, and its prompt path and verify windows, which hand their rows to the next card's.
/// Card 0 runs on the Model's own cache and session and the Conversation's windows, a later card on one of
/// Model::stages.
struct Card {
    size_t index = 0;                             ///< its place among the cards
    int dev = 0;
    int64_t lb = 0, le = 0;                       ///< it runs layers [lb, le)
    /// its memory's id in GET_MEMORY_USE and CHANGE_SETTINGS: gpu<dev>, and gpu<dev>.<index> for a card on the GPU of
    /// an earlier card (the test of a split on one GPU)
    std::string id;
    GpuStage* stage = nullptr;                    ///< null for card 0
    /// the layers whose dense weights it holds, which the cut keeps it within: every layer, or with
    /// --trim-stage-weights its own at the start
    int64_t weights_lb = 0, weights_le = 0;
    strata::core::LinkSpeed link;                 ///< its link to RAM, measured at start
    double layer_ms = 0;                          ///< the predicted ms of one layer of a decode window on it
    /// the hand-offs its verify windows take their rows from (null for card 0) and give them to (null for the last)
    const float* hand_in = nullptr;
    float* hand_out = nullptr;
    strata::core::ExpertCache* cache = nullptr;   ///< the expert cache whose slots are the pool's pages
    strata::core::ExpertPages experts;            ///< the page pool: its experts and the conversations' pages
    strata::core::SessionState* own = nullptr;    ///< the engine's own session on it (the rope table and PLE it lends)
    void* cs = nullptr;                           ///< its stream, as the session functions take it
    strata::core::Verifier* ver = nullptr;
    strata::prefill::Prefill* sp = nullptr;
    int32_t* d_res = nullptr;                     ///< the residency table on its device
    uint64_t sessions_bytes = 0;                  ///< the device memory of the sessions on it (`own`'s included)
    int refill_batch = kRefillBatch;              ///< the most experts one refill puts into its pool (kRefillBudgetMs)
    /// per expert (layer * n_expert + expert) of another card's layers that its pool holds: its slot, else
    /// kNotResident (empty unless the cards lend pages: PagedEngine::place_card_experts)
    std::vector<int32_t> lent;
    /// computes the `lent` experts that a verify window of their own card routes to
    strata::core::PeerExperts hop;
};
using Cards = std::deque<Card>;   ///< (a deque: a card stays where it is as the next one is added)

/// A conversation's state on one card: a session carved for the card's layers, whose GDN state and K/V are pages of
/// the card's pool.
struct Part {
    int dev = 0;
    strata::core::SessionState ss;
    void* mem = nullptr;                ///< `ss`'s memory (without the state the pages hold)
    ConversationPages pages;            ///< given at the conversation's first request
    /// where its GDN state was when the card's verify windows' graphs last saw it (their graphs hold the address)
    const float* gdn_seen = nullptr;

    ~Part() {
        const strata::core::OnDevice on(dev);
        strata::core::session_release(ss);
        (void) cudaFree(mem);
    }
};

/// A place for a conversation on the GPUs: its state on each card and the tokens in it.  Its session on each card is
/// that card's Verifier's slot session of the same index.
struct Held {
    int session = 0;                    ///< its index: its verify-window session; its id in each card's pool - 1
    std::vector<std::unique_ptr<Part>> parts;   ///< its state on each card, in the cards' order
    std::vector<int32_t> live;          ///< the tokens its state holds: the prompt, then the fed outputs
    strata::spec::Decoding dec;         ///< its decoding, while `decoding`
    bool decoding = false;
    int slot = -1;                      ///< while decoding: its batch slot (BT lines), or -1: it is the request on the
                                        ///< control lines (T lines)
    Clock::time_point t0;               ///< when it started decoding
    /// the request count when it last served a request: the oldest goes first, as an idle one to pause and among
    /// decoding ones of the same priority
    uint64_t used = 0;
    int64_t stepped = 0;                ///< the decode step it last took part in (decode_step's turns)

    /// Gets its place in the order in which conversations give way to others (smallest first): the lowest priority,
    /// then the least recently used.
    std::pair<int64_t, uint64_t> get_yield_order() const { return {dec.priority, used}; }

    /// Forgets its tokens and gives back its growing pages.
    void forget() {
        live.clear();
        for (const std::unique_ptr<Part>& p : parts)
            if (p->pages.is_ready())
                p->pages.truncate(0);
    }
    /// Tells whether it has its pages on every card (ConversationPages::init).
    bool has_pages() const {
        return std::all_of(parts.begin(), parts.end(),
                           [](const std::unique_ptr<Part>& p) { return p->pages.is_ready(); });
    }
    /// Gets the bytes of the pool pages it holds on every card.
    uint64_t get_bytes() const {
        uint64_t bytes = 0;
        for (const std::unique_ptr<Part>& p : parts)
            bytes += p->pages.bytes();
        return bytes;
    }
};

/// A conversation paused out of the GPU's pool: its pages in the RAM's pool or in a file on the SSD, and what its
/// session kept beside them.
struct Paused {
    uint64_t serial = 0;              ///< its number (LIST_CONVERSATIONS' parked<serial>) and its id in the RAM's pool
    std::vector<int32_t> live;        ///< the tokens it holds
    /// its sessions' state beside the pages (conversation_unpaged_save), apart from the layer cut: each card's PLE
    /// history, and every QSA layer's indexer state in the layers' order
    std::vector<std::vector<uint8_t>> ple;
    std::vector<uint8_t> qsa;
    int64_t priority = 0;             ///< its last request's
    uint64_t used = 0;                ///< Held::used
    uint64_t bytes = 0;               ///< its pages' bytes
    std::string file;                 ///< on the SSD: card 0's file (get_pause_file); empty: in the RAM's pool
    std::vector<int64_t> file_cut;    ///< on the SSD: the layer cut its files have the cards' parts at (get_cut)
};

/// The engine once its model is up: its conversations, the pages they share with the experts, the step that decodes
/// them, and the requests' protocol.
class PagedEngine {
public:
    /// `cards`: the GPUs' pools, prompt paths and windows, and `split` the CPU pool's routing of their layers
    /// (init_card_paths); `ram`: the RAM's pool (experts and paused conversations), whose conversation layout has the
    /// whole model's regions (open_ram); `copy_back`: an expert a GPU's pool evicts is copied to it.  `slots`: the
    /// batch slots (--batch; 0: one conversation at a time).  `drafter`: null when drafting is off.
    PagedEngine(Model& m, Conversation& conv, Cards& cards, SplitDrive& split, strata::core::RamExperts& ram,
                bool copy_back, ConversationDrafter* drafter, const PagedSettings& settings, int slots);
    /// Deletes the paused conversations' files.
    ~PagedEngine();

    /// Carves each conversation's session on each card and gives them to the cards' verify windows; false (said on
    /// stderr) when one does not fit.
    bool init();
    /// Answers the server's commands until QUIT or the end of stdin; returns the exit status.
    int serve();

private:
    /// Changes the settings CHANGE_SETTINGS names (memory_report.hpp): the pools' sizes, the token limits, the SSD's
    /// budget, the drafter, the conversations at once, the rows and the lookahead.  Gets why not, having changed
    /// nothing, or an empty string.  When a memory gives less than asked (a failed allocation), the change is made as
    /// far as it goes, the report shows the sizes it reached, and `shortfall` says what was missing.
    std::string change_settings(const SettingValues& values, std::string& shortfall);
    /// Takes or gives back chunks of `card`'s pool so that it holds `bytes`.  When the conversations' pages do not fit
    /// the smaller pool, conversations are paused until they do: those that do not decode first (those in the chunks
    /// it gives back before the others), then batch slots' decoding ones in those chunks, the lowest priority first,
    /// which end as preempted.  Gets why not (the memory had less to give), or an empty string.
    std::string resize_gpu(Card& card, uint64_t bytes);
    /// Takes or gives back chunks of the RAM's pool so that it (and the staging ring) hold `bytes`: the conversations
    /// paused in the chunks it gives back move to the SSD.  Gets why not, or an empty string.
    std::string resize_ram(uint64_t bytes);
    void report_resize(const char* name, const strata::core::PagePoolReport& before,
                       const strata::core::PagePoolReport& after, Clock::time_point t0) const;
    /// Runs the layer split's planner on the cards' pool sizes (their memory settings), for the conversations that
    /// decode and the token minimum; gets the cut it chose (get_cut's form), or today's when no other can run (a card
    /// would lack a layer's dense weights, a GDN layer or a QSA layer).
    std::vector<int64_t> plan_cut() const;
    /// Moves the layer cut to `cut` between two steps: every conversation is paused into the RAM's pool, each card
    /// takes its new layers (the pool's layout for them and their experts, the sessions' carve, the verify windows'
    /// and the prompt paths' layers), the decoding conversations come back, and each card fills its pool with its
    /// experts before the next step, so that the steps compute the same experts on the GPUs as before.  False (and
    /// `err`) when a conversation did not fit in the RAM's pool, or a pool cannot have the new layout: the cut stays.
    bool move_cut(const std::vector<int64_t>& cut, std::string& err);
    /// Gives each card's pool the experts it wants that fit (refill), waiting for pages that open soon; then uploads
    /// the residency tables.
    bool fill_cards(std::string& err);
    /// Gives each card's pool the experts it is to hold (ExpertPages::set_ranking): its layers' share of the profile's
    /// first --expert-cache (get_card_ranking), or with `lend`, while the cards lend pages (holder_), those that
    /// place_experts puts on it, which may be of another card's layers that their card has no room for, where a hop
    /// beats the CPU (is_hop_wanted).  The experts it no longer holds leave now; refills bring in the others.
    void place_card_experts(bool lend);
    /// Tells whether a card's pool may hold the experts of another card's layers, which `holder` then computes in one
    /// hop: always with --remote-experts on; with auto, when one hop over its link costs less than the CPU computing
    /// one expert (kCpuExpertMs).
    bool is_hop_wanted(const Card& holder) const;
    /// Gets the layer cut: the first layer of each card after the first.
    std::vector<int64_t> get_cut() const;
    /// Gets the regions of `card`'s pool a paused conversation's copy takes: its GDN state, then its QSA layers' (and
    /// the drafter's, last).
    strata::core::RegionSpan get_gpu_span(const Card& card) const;
    /// Gets where the part of a conversation of a card that runs layers [lb, le) is in the RAM pool's layout, which has
    /// the whole model's regions: its share of the GDN state, and its QSA layers' regions (and the drafter's, last).
    strata::core::RegionSpan get_ram_span(int64_t lb, int64_t le) const;
    /// Copies `h`'s running state beside its pages to `p`, apart from the layer cut (Paused::ple, Paused::qsa).
    bool save_unpaged(const Held& h, Paused& p, std::string& err) const;
    /// Puts `p`'s running state beside the pages back into `h`, each card its layers'.
    bool restore_unpaged(const Paused& p, Held& h, std::string& err) const;
    /// Sets the conversation token limits: the GPU pools' minimum, and the maximum's budget of the conversations paused
    /// in RAM (moving those past it to the SSD).  Gets why not, having changed no pool, or an empty string.
    std::string set_token_limits(int64_t min, int64_t max);
    /// Loads (`on`) or gives back the drafter's weights, and keeps settings_.drafter to whether they are loaded; false
    /// (and `err`) when loading failed.
    bool switch_drafter(bool on, std::string& err);
    /// Gets the least `card`'s pool may hold now: the pages of a conversation token minimum of `tokens_min`, and
    /// those each decoding conversation takes there at the end of its request, in whole chunks.
    uint64_t get_smallest_gpu(const Card& card, int64_t tokens_min) const;
    /// Gets the most `card`'s pool may hold now: what it holds and what its GPU has free beyond what it keeps.
    uint64_t get_largest_gpu(const Card& card) const;
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
    /// Gives `h` the pages for `tokens` tokens on every card, pausing idle conversations (the lowest priority, then
    /// the least recently used, first) while a pool has no room, and tells the kernels about the experts that left
    /// those pages.
    bool make_room(Held& h, int64_t tokens, std::string& err);
    /// Pauses `h`'s conversation, or forgets it when that fails (said on stderr).
    void pause_or_forget(Held& h);
    /// Ends the request that batch slot `slot` decodes, its finish "preempted" (preempt).
    void end_preempted(size_t slot);
    /// Gets the conversation that holds pages, does not decode and goes first (the lowest priority, then the least
    /// recently used), among those `can_go` accepts besides `keep` and the one whose prompt is read; null when none.
    Held* find_idle(const Held* keep, const std::function<bool(const Held&)>& can_go = nullptr);
    /// Gives `h` its fixed regions in the pools of the cards where it has none, and drops a card's verify windows'
    /// graphs when its GDN state there moved.
    bool init_pages(Held& h, std::string& err);
    /// Drops card `card`'s verify windows' graphs when `h`'s GDN state there is elsewhere than when they last saw it.
    bool note_gdn_place(Held& h, size_t card, std::string& err);
    /// Moves `h`'s conversation out of the cards' pools into the RAM's (first moving the RAM's paused conversations
    /// down to the SSD while they would take more than --tokens-max allows), or else into files, one per card; `h`
    /// holds nothing after.  `ram_only`: into the RAM's pool beyond that budget, or not at all (move_cut).  False (and
    /// `err`) when a copy failed or the RAM's pool had no room for it: then `h` keeps its conversation.
    bool pause(Held& h, std::string& err, bool ram_only = false);
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
    /// Gives back a paused conversation's pages in RAM, or deletes its files.
    void drop_pages(const Paused& p);
    /// Gets the bytes of the paused conversations in RAM (`in_ram`) or on the SSD.
    uint64_t get_paused_bytes(bool in_ram) const;
    /// Gets the bytes the conversations paused in RAM may take: one conversation of --tokens-max tokens.
    uint64_t get_ram_pause_budget() const;
    /// Gets the file on the SSD of card `card`'s part of the paused conversation of Paused::serial `serial`.
    std::string get_pause_file(uint64_t serial, size_t card = 0) const;
    /// Gets the paused conversation that goes first, the lowest priority then the least recently used, among those in
    /// RAM (`in_ram`) or on the SSD; paused_.end() when there is none.
    std::vector<Paused>::iterator find_lowest(bool in_ram);
    /// Every --adapt-every steps (the old engine's adaptive tier, with its options): counts a use, in the GPUs' pools,
    /// of the experts there that the steps since the last round routed to; then the experts decoding routed to most
    /// that the pools do not hold take the pages of the least-routed ones of their layer (at most --adapt-swaps,
    /// each by --adapt-min-gain more routings).  False (ERR printed) when a copy failed.
    bool adapt_experts();
    /// Puts back a batch of the experts the conversations evicted (the GPUs' first, then the RAM's), or waits until
    /// pages open to them: the serve loop's work between requests.  False when a copy failed (said on stderr).
    bool refill_experts();
    /// Tells whether a pool wants experts back (ExpertPages::is_refill_wanted).
    bool is_refill_wanted() const;
    /// Uploads the cards' residency tables when experts came or left since the last upload.
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
    /// Gets the card that runs layer `layer`.
    Card& get_card_of(int64_t layer) const {
        return *std::find_if(cards_.begin(), cards_.end(), [&](const Card& c) { return layer < c.le; });
    }
    /// Tells whether a card other than the one that runs `layer` holds its expert `expert` (Card::lent).
    bool is_lent(int64_t layer, int64_t expert) const {
        const int32_t holder = holder_.empty() ? -1 : holder_[(size_t) layer];
        return holder >= 0 && cards_[(size_t) holder].lent[(size_t) (layer * m_.g.n_expert + expert)] >= 0;
    }
    /// Describes the conversations for LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS.
    std::vector<strata::core::HeldConversation> list_conversations() const;
    /// Describes the memories for GET_MEMORY_USE.
    strata::serve::MemoryReport describe_memory() const;
    /// Gets the step's view of the model: the verify windows over the conversations' sessions.
    strata::spec::StepModel get_step_model();

    Model& m_;
    Conversation& conv_;   ///< card 0's prompt path and verify windows, shared by every conversation
    Cards& cards_;
    SplitDrive& split_;
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
    uint64_t requests_ = 0;            ///< the requests served so far (Held::used)
    const Held* reading_ = nullptr;    ///< the conversation whose prompt read_prompt reads, which make_room keeps
    bool residency_changed_ = false;   ///< `m_.host_res` changed since the device's table was uploaded
    /// Per layer, the card whose pool holds the layer's experts that its own card's does not (SplitDrive::holder), or
    /// -1; empty while the cards lend no pages (one card, or --remote-experts off)
    std::vector<int32_t> holder_;
    int64_t steps_ = 0;                ///< the decode steps so far (adapt_experts' rounds)
    /// decode_step refills experts too, until none are wanted: a pool grew, or is filling after the start
    /// (ExpertPages::open)
    bool refilling_ = true;
    std::vector<float> seen_;          ///< `m_.drive.d.usage` when adapt_experts last decayed it
    int64_t swapped_ = 0;              ///< the experts adapt_experts moved into the GPU's pool, since the last report
    double adapt_ms_ = 0;              ///< the time those took
    int64_t copied_back_ = 0;          ///< the experts the GPU's pool evicted that were copied to RAM, since then
    double copy_back_ms_ = 0;          ///< the time the copies back took (all evictions', failed copies too)
    /// `m_.drive.d.usage` before the step that runs now: the experts whose count grows were routed to in the step
    std::vector<float> routed_before_;
    int step_rows_ = 0;                ///< the rows the last step ran
    /// Since the last report: the decode steps that routed on the CPU's side (not every expert on the GPU), their
    /// rows, and the experts they routed to, each counted once a step (docs/DETAILS.md, rows and guesses)
    int64_t routed_steps_ = 0, routed_rows_ = 0, routed_experts_ = 0;
};

/// Removes strata-q38's own options from `argv` (the rest is `strata`'s command line).  -1: go on, else exit status.
int take_own_options(int& argc, char** argv, PagedSettings& s) {
    struct Own {
        const char* name;
        int64_t* value;
        std::string* text = nullptr;   ///< instead of `value`, for a word
    };
    int64_t rows = s.step.rows, lookahead = -1, concurrent = -1;   // -1: chosen once the options are read
    std::string drafter = "on", gpu_mib;
    const Own own[] = {{"--tokens-min", &s.tokens_min},       {"--tokens-max", &s.tokens_max},
                       {"--gpu-mib", nullptr, &gpu_mib},      {"--ram-mib", &s.ram_mib},
                       {"--ssd-pause-mib", &s.ssd_pause_mib}, {"--rows", &rows},
                       {"--lookahead", &lookahead},           {"--concurrent", &concurrent},
                       {"--pause-dir", nullptr, &s.pause_dir}, {"--copy-back", nullptr, &s.copy_back},
                       {"--drafter", nullptr, &drafter},      {"--remote-experts", nullptr, &s.remote_experts}};
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
    if (s.remote_experts != "auto" && s.remote_experts != "on" && s.remote_experts != "off") {
        std::fprintf(stderr, "strata-q38: --remote-experts takes auto, on or off\n");
        return 2;
    }
    if (drafter != "on" && drafter != "off") {
        std::fprintf(stderr, "strata-q38: --drafter takes on or off\n");
        return 2;
    }
    if (std::string err; !gpu_mib.empty() && !parse_i64_list(gpu_mib.c_str(), s.gpu_mib, err)) {
        std::fprintf(stderr, "strata-q38: --gpu-mib takes MiB, one per card of a layer split: %s\n", err.c_str());
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

/// Gets whether `card` runs `m`'s MTP drafter (--mtp), which Model::load_drafter puts on a layer split's last card.
bool has_drafter(const Model& m, const Card& card) {
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

/// Gets the regions a conversation keeps in the pool of `card` (whose pool is open) when the card runs layers
/// [lb, le): ConversationPages::get_layout's for its session carved for them, and the drafter's K/V region of its
/// pool when it runs the drafter (has_drafter).
bool get_card_layout(const Model& m, const Card& card, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                     std::string& err) {
    strata::core::SessionState carve = *card.own;   // (a copy of the description only: the memory stays the session's)
    strata::core::session_set_layers(carve, m.g, lb, le);
    if (!ConversationPages::get_layout(m.g, carve, nullptr, layout, err))
        return false;
    if (has_drafter(m, card))
        layout.regions.push_back(card.experts.pool().layout().regions.back());
    return true;
}

uint64_t get_gpu_room(const Model& m, const Card& card, uint64_t mapped);
bool init_card_paths(Model& m, Conversation& conv, Cards& cards, SplitDrive& split, std::string& err);
void set_stage(const Cards& cards, size_t k);
uint64_t get_ram_room(uint64_t headroom);
uint64_t get_commit_room();

/// Gets the layers [lb, le) card `k` runs at layer cut `cut` (PagedEngine::get_cut's form) of a model of `n_layers`.
std::pair<int64_t, int64_t> get_card_layers(const std::vector<int64_t>& cut, size_t k, int64_t n_layers) {
    return {k == 0 ? 0 : cut[k - 1], k < cut.size() ? cut[k] : n_layers};
}

/// Adds the card that runs layers [lb, le) of `m` to `cards`: CUDA0's (card 0, over the Model's own cache and
/// session and `conv`'s windows) or the layer split's stage `stage`.  Its link to RAM is measured before its pool
/// takes the memory the measuring copies need, and its speed is predicted from its multiprocessors and clock.
Card& add_card(Model& m, Conversation& conv, Cards& cards, GpuStage* stage) {
    Card& card = cards.emplace_back();
    card.index = cards.size() - 1;
    card.dev = stage != nullptr ? stage->dev : m.primary_device;
    card.lb = stage != nullptr ? stage->lb : 0;
    card.le = stage != nullptr ? stage->le : m.split_at.empty() ? m.g.n_layers : m.split_at[0];
    const bool shared = std::any_of(cards.begin(), cards.end() - 1, [&](const Card& c) { return c.dev == card.dev; });
    card.id = "gpu" + std::to_string(card.dev) + (shared ? "." + std::to_string(card.index) : "");
    card.stage = stage;
    card.cache = stage != nullptr ? &stage->cache : &m.xcache;
    card.own = stage != nullptr ? &stage->ss : &m.ss;
    card.cs = stage != nullptr ? (void*) stage->stream : m.main_cs;
    card.ver = stage != nullptr ? &stage->ver : &conv.ver;
    card.sp = stage != nullptr ? &stage->sp : &conv.sp;
    card.weights_lb = m.stage_weights_trimmed ? card.lb : 0;
    card.weights_le = m.stage_weights_trimmed ? card.le : m.g.n_layers;
    const strata::core::LinkSpeed link = card.link = strata::core::measure_host_link(card.dev, kLinkProbeBytes);
    card.refill_batch = strata::core::get_refill_batch(link, strata::kernels::cpu::expert_layout().max_blob,
                                                       kRefillBudgetMs, kRefillBatch);
    std::fprintf(stderr, "strata-q38: %s's link to RAM: %.1f GB/s to the GPU, %.1f GB/s back, %.3f ms a round trip\n",
                 card.id.c_str(), link.to_gbps, link.from_gbps, link.latency_ms);
    int sms = 0, khz = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, card.dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, card.dev) != cudaSuccess || khz <= 0)
        khz = 1800000;   // (the old engine's guess)
    (void) cudaGetLastError();
    card.layer_ms = strata::program::split_rules::get_layer_ms(std::max(1.0, sms * (khz / 1e6)));
    return card;
}

/// Opens `card`'s expert cache in the pages of its page pool, for at most `max_experts` experts
/// (Model::open_expert_pages, open_stage_pages); with a layer split, for those of the expert profile's first
/// `max_experts` that are of the card's layers, so that the cards hold the experts one GPU would.  The pool takes
/// `gpu_mib` (-1: what get_gpu_room gives, but no more than those experts (the profile's first) and a conversation at
/// --max-context take), in chunks of kChunkBytes; experts leave the pages of `tokens_min` tokens free.  With the
/// drafter (--mtp), the pool of the card that runs it (has_drafter) also keeps each conversation's drafter K/V, and
/// the drafter's head stays out of it.
bool open_pages(Model& m, Card& card, int64_t tokens_min, int64_t gpu_mib, int64_t max_experts, std::string& err) {
    const strata::core::OnDevice on(card.dev);
    const std::vector<strata::core::ExpertKey> ranking = get_card_ranking(m, card.lb, card.le, max_experts);
    const bool drafting = has_drafter(m, card);
    const std::vector<uint64_t> blobs = get_blob_bytes(m);
    const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
    strata::core::StateLayout layout;
    if (!ConversationPages::get_layout(m.g, *card.own, drafting ? &m.mtp.kv_state() : nullptr, layout, err))
        return false;
    uint64_t most = layout.bytes_for(m.o.max_context);
    for (const strata::core::StateRegion& r : layout.regions)   // + each region's last block
        most += strata::core::choose_block_pages(r, page) * page;
    for (size_t i = 0; i < ranking.size() && (int64_t) i < max_experts; ++i)
        most += (blobs[(size_t) ranking[i].layer] + page - 1) / page * page;
    const uint64_t room = get_gpu_room(m, card, 0);
    const uint64_t pool_b = gpu_mib >= 0 ? std::min((uint64_t) gpu_mib << 20, room) : std::min(room, most);
    const uint64_t chunk = std::max<uint64_t>(std::min(kChunkBytes, pool_b) / page, 1) * page;
    card.cache->set_expert_bytes((int64_t) strata::kernels::cpu::expert_layout().max_blob);
    strata::core::PagePoolConfig config;
    config.tokens_min = std::clamp<int64_t>(tokens_min, 0, m.o.max_context);
    strata::core::ChunkMemory memory;   // (called later from any card's work: each on its own GPU)
    memory.alloc = [dev = card.dev](uint64_t n) {
        const strata::core::OnDevice on_card(dev);
        return (uint8_t*) strata::core::gpu_alloc_zeroed((size_t) n);
    };
    memory.free = [dev = card.dev](uint8_t* p) {
        const strata::core::OnDevice on_card(dev);
        strata::core::gpu_free(p);
    };
    memory.copy = [dev = card.dev](uint8_t* to, const uint8_t* from, uint64_t n) {
        const strata::core::OnDevice on_card(dev);
        return strata::core::gpu_copy(to, from, (size_t) n);
    };
    if (!card.cache->open((int64_t) (chunk / page), m.g.n_layers, m.g.n_expert, (int64_t) page, err) ||
        !card.experts.open(*card.cache, std::move(memory), pool_b, layout, config, blobs, m.g.n_expert, max_experts,
                           ranking, err))
        return false;
    std::fprintf(stderr, "strata-q38: the page pool of %s (layers %lld-%lld): %.2f GiB (of %.2f GiB it could take) in "
                         "chunks of %.0f MiB, pages of %llu bytes, for the experts and the conversations, %lld tokens "
                         "of it kept for the conversations\n",
                 card.id.c_str(), (long long) card.lb, (long long) card.le - 1,
                 (double) card.experts.pool().report().total.bytes / 1073741824.0, (double) room / 1073741824.0,
                 (double) chunk / 1048576.0, (unsigned long long) page, (long long) config.tokens_min);
    return true;
}

/// Gets the memory `card`'s pool may take besides what it holds: what its GPU has free less --vram-reserve-mib and the
/// prompt path's own buffers, and the drafter's head when the card runs it (has_drafter).  On Windows, for a GPU whose
/// memory is the system RAM, what is free is what is free of the share the BIOS gives it (get_share_room, `mapped` the
/// RAM's page-locked pool), not the runtime's free memory: that is the share and what the GPU may borrow beyond it,
/// and HIP takes page-locked host memory off it twice (on the Strix Halo, 8 GiB of hipHostMalloc took 16 GiB off
/// it).  Its memory counts in Windows' commit charge, so no more than get_commit_room either.  Cards on one GPU each
/// see what is free of it.
uint64_t get_gpu_room(const Model& m, const Card& card, uint64_t mapped) {
    const uint64_t keep = ((uint64_t) m.o.vram_reserve_mib << 20) +
        strata::prefill::Prefill::bytes_needed_owned(m.g, *card.own, m.o.prefill_chunk, m.srcp != nullptr) +
        (has_drafter(m, card)
             ? m.mtp.bind_bytes((card.stage != nullptr ? card.stage->head : m.native_head).row_bytes(), m.n_vocab)
             : 0);
    const strata::core::OnDevice on(card.dev);
    uint64_t free_b = 0;
    if (strata::core::get_share_room(card.dev, mapped, free_b))
        free_b = std::min(free_b, get_commit_room());
    else
        free_b = strata::core::device_free_bytes();
    return free_b > keep ? free_b - keep : 0;
}

/// Gets the RAM a pool may take by default besides what the pools hold: what is available less `headroom`, and no more
/// than get_commit_room.
uint64_t get_ram_room(uint64_t headroom) {
    strata::core::detail::HostMemory hm;
    if (!strata::core::detail::host_available_memory(hm))
        return 0;
    return std::min(hm.available > headroom ? hm.available - headroom : 0, get_commit_room());
}

/// Gets how much more may be committed while a tenth of Windows' commit limit, at least 8 GiB, stays uncommitted
/// (kCommitMarginShare, kCommitMarginMin); UINT64_MAX where the commit charge is not reported (elsewhere than
/// Windows).
uint64_t get_commit_room() {
    strata::core::detail::HostMemory hm;
    if (!strata::core::detail::host_available_memory(hm) || hm.commit == ~uint64_t{0})
        return UINT64_MAX;
    const uint64_t share = hm.commit_limit == ~uint64_t{0} ? 0 : (uint64_t) (kCommitMarginShare * hm.commit_limit);
    const uint64_t margin = std::max(kCommitMarginMin, share);
    return hm.commit > margin ? hm.commit - margin : 0;
}

/// Opens the RAM's page pool `ram` beside the cards' and gets whether an expert a GPU's pool evicts is copied back to
/// it (`copy_back`, by --copy-back and the two speeds measured here).  The pool takes --ram-mib, by default
/// get_ram_room (the RAM free less the resident headroom, within the commit margin), but no more than every expert
/// and the staging ring take.  It holds first the experts of the profile the GPUs' pools do not, in the profile's
/// order, then copies of the GPUs', then the experts outside the profile, as far as they fit.  Its conversation layout
/// has the whole model's regions, so that a paused conversation is one conversation in it wherever the layer cut is
/// (PagedEngine::get_ram_span).
bool open_ram(Model& m, const Cards& cards, const PagedSettings& s, strata::core::RamExperts& ram, bool& copy_back,
              std::string& err) {
    using strata::core::ExpertKey;
    const std::vector<uint64_t> blobs = get_blob_bytes(m);
    const auto at = [&](int64_t l, int64_t e) { return (size_t) (l * m.g.n_expert + e); };
    std::vector<ExpertKey> ranking, on_gpu;
    std::vector<uint8_t> ranked(blobs.size() * (size_t) m.g.n_expert, 0);
    for (const auto& [l, e] : m.profile_all.empty() ? m.profile : m.profile_all) {
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
    if (s.ram_mib < 0)
        bytes = std::min(get_ram_room(m.o.resident_headroom),
                         every + (uint64_t) ring * *std::max_element(blobs.begin(), blobs.end()));
    strata::core::PagePoolConfig config;
    config.page_bytes = cards[0].experts.pool().page_bytes();
    config.chunk_bytes = kChunkBytes / config.page_bytes * config.page_bytes;
    strata::core::StateLayout layout;
    if (!get_card_layout(m, cards.back(), 0, m.g.n_layers, layout, err))
        return false;
    const Clock::time_point t0 = Clock::now();
    if (!ram.open(m.src, bytes, ring, layout, config, blobs, m.g.n_expert, ranking, err))
        return false;
    const double fill_s = get_ms_since(t0) / 1000.0;
    const uint64_t held = ram.pages().pool().report().experts.bytes;
    double ssd = 0, back = 0;
    // the best of three: on the Strix Halo single copies measured 3.3 to 57 GB/s, single SSD reads 0.3 to 2.4 GB/s
    for (int round = 0; round < 3; ++round) {
        ssd = std::max(ssd, ram.time_ssd_reads(round));
        back = std::max(back, ram.time_copy_in(m.xcache.device_slot(0), ram.ring_bytes()));
    }
    copy_back = s.copy_back == "on" || (s.copy_back == "auto" && strata::core::is_copy_back_worth(back, ssd));
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
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    // ROCclr on Windows keeps freed memory in its resource cache and gives none of it back (on the Strix Halo, 8 GiB
    // freed with hipHostFree or hipFree stayed taken from Windows' RAM), so the pools could not shrink; an explicit
    // setting wins
    strata::platform::set_env("GPU_RESOURCE_CACHE_SIZE", "0", false);
#endif
    // a card computing another card's experts (--remote-experts) runs them on a stream of its own while that card's
    // verify window spins on the GPU for them; the runtime spreads its streams over a few hardware queues (HIP 4 a
    // priority, CUDA 8 connections), and a hop that landed behind a spinning kernel waited for ever (two stages on the
    // Strix Halo's GPU, 2 of 2 runs; 0 of 1 with 16 queues, which a stream of the highest priority did not fix).  An
    // explicit setting wins.
    if (settings.remote_experts != "off" &&
        std::any_of(argv + 1, argv + argc, [](const char* a) { return std::strcmp(a, "--layer-split") == 0; })) {
        strata::platform::set_env("GPU_MAX_HW_QUEUES", "16", false);
        strata::platform::set_env("CUDA_DEVICE_MAX_CONNECTIONS", "32", false);
    }
    prepare_process(argc, argv);
    prime_gpu_queues();
    Options o;
    bool have_tokens = false;
    if (const int status = parse_options(argc, argv, o, have_tokens); status >= 0) return status;
    if (const int status = apply_device_defaults(o); status >= 0) return status;
    const int slots = std::max(o.batch, 0);
    if (settings.step.concurrent < 0 || settings.step.concurrent > std::max(slots, 1))
        settings.step.concurrent = std::max(slots, 1);
    if (settings.step.lookahead < 0)
        settings.step.lookahead = o.mtp.empty() ? 0 : kDefaultLookahead;
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
    // strata-q38 does not pipeline verify windows across cards; 0 also keeps the drafter on the last card (mtp_first)
    o.pipeline_windows = 0;
    if (o.prefill_chunk <= 0) o.prefill_chunk = 2048;   // open_pages leaves room for the prompt path's own buffers
    if (o.resident_budget > 0 && settings.ram_mib < 0)
        settings.ram_mib = (int64_t) (o.resident_budget >> 20);
    if (settings.tokens_max < 0)
        settings.tokens_max = o.max_context;
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
    Cards cards;
    const auto get_gpu_mib = [&](size_t card) {
        return settings.gpu_mib.empty() ? -1 : settings.gpu_mib[std::min(card, settings.gpu_mib.size() - 1)];
    };
    m.open_expert_pages = [&](int64_t max_experts, std::string& err) {
        // from here on the experts are read from the files without the file cache
        if (std::string why; !m.src.read_unbuffered(why))
            std::fprintf(stderr, "strata-q38: WARNING: the experts are read through the file cache (%s)\n",
                         why.c_str());
        return open_pages(m, add_card(m, conv, cards, nullptr), settings.tokens_min, get_gpu_mib(0), max_experts,
                          err);
    };
    m.open_stage_pages = [&](GpuStage& st, std::string& err) {
        Card& card = add_card(m, conv, cards, &st);
        return open_pages(m, card, settings.tokens_min, get_gpu_mib(card.index), o.expert_cache, err);
    };
    if (const int status = m.load(); status != 0) return status;
    if (const int status = m.build_hit_tables(false); status != 0) return status;
    for (Card& card : cards)
        card.d_res = card.stage != nullptr ? card.stage->d_res : m.d_res;
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
    if (!open_ram(m, cards, settings, ram, copy_back, err)) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    m.use_expert_source(ram);
    m.set_prompt_pinned_share();
    m.arm_prompt_cpu_share();
    ConversationDrafter drafter;
    SplitDrive split;
    if (!init_card_paths(m, conv, cards, split, err) || (drafting && !drafter.init(m, slots + 1, err))) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    if (drafting)   // the conversations' guesses make many row layouts: the verify windows keep the recent ones' graphs
        for (const Card& card : cards)
            card.ver->set_batch_graph_limit(kWindowGraphs);
    if (drafting && !settings.drafter)   // loaded for the pool's layout of the drafter's K/V, switched on later
        m.unload_drafter();
    PagedEngine engine(m, conv, cards, split, ram, copy_back, drafting ? &drafter : nullptr, settings, slots);
    return engine.init() ? engine.serve() : 1;
}

namespace {

/// Sets up each card's prompt path and verify windows over its own session, each handing its rows to the next card's
/// (through hand-offs in page-locked RAM, so that cards without peer access work), and the CPU expert pool's work in
/// the windows going to the card that runs each layer (`split`, which must outlive them).  The Model's stages run in
/// the order of their layers, as the old engine's layer split runs them.
bool init_card_paths(Model& m, Conversation& conv, Cards& cards, SplitDrive& split, std::string& err) {
    const Options& o = m.o;
    const size_t n = cards.size();
    std::vector<float*> hand(n - 1, nullptr);
    if (!alloc_split_handoffs(m.g, strata::kernels::kVerifyMaxT, hand)) {
        err = "the layer split's hand-offs do not fit";
        return false;
    }
    for (size_t k = n; k-- > 0;) {   // a card's prompt path is set up after the next card's, which it hands to
        Card& card = cards[k];
        card.hand_in = k == 0 ? nullptr : hand[k - 1];
        card.hand_out = k + 1 == n ? nullptr : hand[k];
        set_stage(cards, k);
        if (k == 0)
            break;
        const GpuStage& st = *card.stage;
        const strata::core::OnDevice on(card.dev);
        if (!card.sp->init(st.wt, m.g, *card.own, m.srcp, card.cache, m.host_res.data(), o.prefill_chunk, card.cs,
                           err) ||
            !card.ver->init(st.wt, m.g, *card.own, m.get_verify_hits(st), st.head.loaded() ? &st.head : nullptr, o.spec,
                            err)) {
            err = card.id + "'s prompt path or windows: " + err;
            return false;
        }
        card.ver->set_split(o.spec_split);
        card.ver->set_pcie_mode(get_pcie_mode_code(o));
    }
    if (!conv.init_prompt(m, m.ss, nullptr, 0, err) || !conv.init_decode(m, m.ss, o.spec, err))
        return false;
    if (n == 1)
        return true;
    split.base = &m.drive;
    split.n = (int) n;
    for (size_t k = 0; k < n; ++k) {
        const Card& card = cards[k];
        split.end[k] = card.le;
        split.plan[k] = card.ver->plan_sink();
        split.cache_base[k] = card.stage != nullptr ? card.cache->device_slot(0) : m.drive.d.cache_base;
        split.cache_slot_off[k] = card.stage != nullptr ? card.cache->slot_offsets() : m.drive.d.cache_slot_off;
        split.pcie_num[k] = get_pcie_num(card.stage != nullptr ? card.stage->pcie_frac : o.pcie_frac);
        if (k + 1 < n)
            card.ver->set_next(cards[k + 1].ver, &split);
    }
    conv.set_split_drive(&split);
    return true;
}

/// Gives card `k` of `cards` its layers in its prompt path and verify windows (Card::lb and le), as their stage of the
/// layer split, handing their rows on to the next card's.
void set_stage(const Cards& cards, size_t k) {
    const Card& card = cards[k];
    const bool last = k + 1 == cards.size();
    card.sp->set_stage(card.lb, card.le, last ? nullptr : cards[k + 1].sp);
    card.ver->set_stage(card.lb, card.le, card.hand_in, card.hand_out);
}

PagedEngine::PagedEngine(Model& m, Conversation& conv, Cards& cards, SplitDrive& split, strata::core::RamExperts& ram,
                         bool copy_back, ConversationDrafter* drafter, const PagedSettings& settings, int slots)
    : m_(m), conv_(conv), cards_(cards), split_(split), ram_(ram), drafter_(drafter), settings_(settings),
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
    for (Card& card : cards_) {
        strata::core::ExpertPages::Hooks hooks;
        hooks.load = [this, &c = card](strata::core::ExpertKey key, int32_t slot, std::string& err) {
            const uint8_t* blob = m_.srcp->blob_stable(key.layer, key.expert);
            if (blob == nullptr)
                err = "no bytes for layer " + std::to_string(key.layer) + " expert " + std::to_string(key.expert);
            const strata::core::OnDevice on(c.dev);
            return blob != nullptr &&
                   c.cache->fill_slot_blocking(slot, blob, err,
                                               (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(key.layer));
        };
        // an expert of another card's layers goes to the card's `lent`, not to the residency table its layer's card
        // reads; an evicted one leaves whichever holds it
        hooks.moved = [this, &c = card](strata::core::ExpertKey key, int32_t slot) {
            const size_t i = (size_t) key.layer * (size_t) m_.g.n_expert + (size_t) key.expert;
            const bool own = c.lent.empty() || (slot >= 0 ? key.layer >= c.lb && key.layer < c.le : c.lent[i] < 0);
            (own ? m_.host_res[i] : c.lent[i]) = slot;
            residency_changed_ = residency_changed_ || own;
        };
        if (copy_back)
            hooks.leaving = [this](strata::core::ExpertKey key, const uint8_t* bytes) {
                const Clock::time_point t0 = Clock::now();
                copied_back_ += ram_.keep_copy(key, bytes);
                copy_back_ms_ += get_ms_since(t0);
            };
        card.experts.set_hooks(std::move(hooks));
    }
}

PagedEngine::~PagedEngine() {
    for (const Paused& p : paused_)
        drop_pages(p);
}

bool PagedEngine::init() {
    using namespace strata::core;
    for (Card& card : cards_) {
        const OnDevice on(card.dev);
        // every session on the card reads the engine's own session's rope table there
        const QsaState* rope = &card.own->qsa_states[card.own->qsa_primary()];
        // (carved for every layer, as the card's own session: move_cut changes the layers they are for)
        const uint64_t bytes = session_bytes(m_.g, m_.o.max_context, m_.K, 0, -1, false);
        card.sessions_bytes = session_bytes(m_.g, m_.o.max_context, m_.K) + held_.size() * bytes;
        std::vector<SessionState*> sessions;
        for (const std::unique_ptr<Held>& h : held_) {
            Part& part = *h->parts.emplace_back(std::make_unique<Part>());
            part.dev = card.dev;
            if (cudaMalloc(&part.mem, bytes) != cudaSuccess ||
                session_init(m_.g, m_.o.max_context, m_.K, part.mem, part.ss, 0, -1, rope) == 0) {
                std::fprintf(stderr, "strata-q38: no room for conversation %zu's session on %s (%.1f MiB)\n",
                             sessions.size(), card.id.c_str(), (double) bytes / 1048576.0);
                return false;
            }
            session_set_layers(part.ss, m_.g, card.lb, card.le);
            PleRun& ple = part.ss.ple;   // the engine's PLE weights and buffers, with this conversation's history
            ple = card.own->ple;
            ple.token = &part.ss.ple_token;
            ple.prev = part.ss.ple_prev;
            ple.hist = part.ss.ple_hist;
            session_zero(part.ss, m_.g, nullptr, card.cs);
            sessions.push_back(&part.ss);
        }
        std::string err;
        if (cudaStreamSynchronize((cudaStream_t) card.cs) != cudaSuccess || !card.ver->init_slots(sessions, err)) {
            std::fprintf(stderr, "strata-q38: the conversations' sessions on %s: %s\n", card.id.c_str(), err.c_str());
            return false;
        }
    }
    if (cards_.size() < 2 || settings_.remote_experts == "off" || m_.profile_all.empty())
        return true;
    const ModelGeometry& g = m_.g;
    for (Card& card : cards_) {
        card.lent.assign((size_t) (g.n_layers * g.n_expert), kNotResident);
        if (std::string err; !card.hop.attach(card.dev, *card.cache, card.lent.data(), g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata-q38: %s cannot compute other cards' experts: %s\n", card.id.c_str(),
                         err.c_str());
            for (Card& c : cards_)
                c.lent.clear();
            return settings_.remote_experts != "on";   // (auto: the cards lend no pages)
        }
        split_.hop[card.index] = &card.hop;
    }
    holder_.assign((size_t) g.n_layers, -1);
    split_.holder = holder_.data();
    place_card_experts(true);
    return true;
}

int PagedEngine::serve() {
    const Options& o = m_.o;
    const int slots = (int) at_.size() - 1;
    int64_t resident = 0;
    uint64_t expert_bytes = 0;
    for (const Card& card : cards_) {
        resident += card.cache->resident();
        expert_bytes += card.experts.pool().report().experts.bytes;
    }
    std::printf("INFO context=%lld kv=%s expert_slots=%lld expert_cache_mib=%lld spec=%d prefill_cap=%lld%s engine="
                STRATA_VERSION "\n",
                (long long) o.max_context, o.kv.c_str(), (long long) resident, (long long) (expert_bytes >> 20),
                get_drafter() != nullptr ? settings_.step.lookahead + 1 : 0, (long long) o.prefill_chunk,
                slots > 0 ? (" batch_slots=" + std::to_string(slots) + " slot_cache=0").c_str() : "");
    std::printf("READY %lld stop conversations memory preempt settings\n", (long long) o.max_context);
    strata::serve::EngineHooks hooks;
    hooks.is_busy = [this] { return is_decoding() || is_refill_wanted(); };
    hooks.step = [this] { return is_decoding() ? decode_step() : refill_experts(); };
    hooks.stop_slot = [this](int slot) { stop_slot(slot); };
    hooks.list_conversations = [this] { return list_conversations(); };
    hooks.describe_memory = [this] { return describe_memory(); };
    hooks.change_settings = [this](const SettingValues& values, std::string& shortfall) {
        return change_settings(values, shortfall);
    };
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
    if (!h.has_pages() && !init_pages(h, err)) {
        std::printf("ERR %s\n", err.c_str());
        return true;
    }
    if (resume == 0) {
        h.forget();
        for (const Card& card : cards_) {
            const strata::core::OnDevice on(card.dev);
            strata::core::session_zero(h.parts[card.index]->ss, m_.g, nullptr, card.cs);
            (void) cudaStreamSynchronize((cudaStream_t) card.cs);
        }
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
    for (const Card& card : cards_)
        card.sp->set_session(h.parts[card.index]->ss);
    reading_ = &h;
    std::vector<int32_t> next;   // the drafter's: the token after each of the chunk's
    strata::prefill::Prefill& last = *cards_.back().sp;   // which has the head
    last.on_chunk = [&](const float* R_rows, int64_t T, int64_t pos0, std::string& e) -> bool {
        if (ConversationDrafter* drafter = get_drafter()) {
            next.assign(ids.begin() + pos0 + 1, ids.begin() + pos0 + T + 1);
            Part& part = *h.parts.back();
            if (!drafter->read_prompt(last, part.pages, part.ss, R_rows, next.data(), T, pos0, (int64_t) ids.size(),
                                      e))
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
    last.on_chunk = nullptr;
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
    const std::vector<float>& usage = m_.drive.d.usage;
    routed_before_.assign(usage.begin(), usage.end());
    std::string err;
    if (!step_.run(active, err)) {
        std::printf("ERR %s\n", err.c_str());
        return false;
    }
    if (const int64_t routed = std::inner_product(usage.begin(), usage.end(), routed_before_.begin(), (int64_t) 0,
                                                  std::plus<>(), std::greater<>());
        routed > 0) {
        routed_experts_ += routed;
        routed_rows_ += step_rows_;
        ++routed_steps_;
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
        if (routed_steps_ > 0) {
            std::fprintf(stderr, "strata-q38: %lld decode steps of %lld rows in all routed to %lld experts (each "
                                 "counted once a step)\n",
                         (long long) routed_steps_, (long long) routed_rows_, (long long) routed_experts_);
            routed_steps_ = routed_rows_ = routed_experts_ = 0;
        }
        if (swapped_ > 0 || copy_back_ms_ > 0) {
            std::fprintf(stderr, "strata-q38: %lld experts moved into the GPU's pool (%.0f ms in all); %lld it evicted "
                                 "copied back to RAM (%.1f ms, all evictions)\n",
                         (long long) swapped_, adapt_ms_, (long long) copied_back_, copy_back_ms_);
            swapped_ = copied_back_ = 0;
            adapt_ms_ = copy_back_ms_ = 0;
        }
        for (const Card& card : cards_)
            if (card.hop.experts() > 0)
                std::fprintf(stderr, "strata-q38: so far %s computed %lld experts of other cards' layers (%lld rows) "
                                     "in one hop each layer, %.1f ms waited for\n",
                             card.id.c_str(), (long long) card.hop.experts(), (long long) card.hop.entries(),
                             card.hop.ms_wait);
        if (std::strcmp(d.finish, "nonfinite") == 0) {   // #879: its state may carry the poison: start over next time
            std::fprintf(stderr, "strata-q38: non-finite logits (#879) at position %lld: the conversation's next "
                                 "request reads its prompt from token 0\n", (long long) d.pos);
            h->forget();
        }
        if (h->slot >= 0)
            std::printf("BDONE %d %lld %s %.1f\n", h->slot, (long long) d.produced, d.finish, get_ms_since(h->t0));
    }
    strata::core::progress_beat();
    if (refilling_ && !is_refill_wanted())
        refilling_ = false;
    return adapt_experts() && (!refilling_ || refill_experts());
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
    for (const std::unique_ptr<Part>& part : h.parts) {
        const strata::core::OnDevice on(part->dev);
        while (!part->pages.reserve(tokens, err)) {
            Held* idle = find_idle(&h);
            if (idle == nullptr)
                return false;
            pause_or_forget(*idle);
        }
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
    for (Card& card : cards_) {
        Part& part = *h.parts[card.index];
        if (part.pages.is_ready())   // an earlier call failed on a later card
            continue;
        const strata::core::OnDevice on(card.dev);
        if (!part.pages.init(m_.g, part.ss, card.experts.pool(), (strata::core::ConversationId) h.session + 1, err) ||
            !note_gdn_place(h, card.index, err))
            return false;
    }
    return true;
}

bool PagedEngine::note_gdn_place(Held& h, size_t card, std::string& err) {
    Part& part = *h.parts[card];
    const bool moved = part.gdn_seen != nullptr && part.gdn_seen != part.ss.gdn_state;
    part.gdn_seen = part.ss.gdn_state;
    const strata::core::OnDevice on(part.dev);
    return !moved || cards_[card].ver->drop_batch_graphs(err);
}

bool PagedEngine::pause(Held& h, std::string& err, bool ram_only) {
    const Clock::time_point t0 = Clock::now();
    Paused p;
    p.serial = ++serial_;
    p.priority = h.dec.priority;
    p.used = h.used;
    if (!save_unpaged(h, p, err))
        return false;
    strata::core::PagePool& ram = ram_.pages().pool();
    const int64_t tokens = h.parts[0]->pages.get_pool_tokens((int64_t) h.live.size());
    p.bytes = ram.pages_for(tokens) * ram.page_bytes();
    const uint64_t budget = ram_only ? UINT64_MAX : get_ram_pause_budget();
    while (p.bytes <= budget && get_paused_bytes(true) + p.bytes > budget)
        if (!move_down())
            break;
    const bool in_ram = get_paused_bytes(true) + p.bytes <= budget && ram.grow_conversation(p.serial, tokens).empty();
    if (in_ram) {
        ram.set_paused(p.serial, true);
    } else if (ram_only) {
        err = "the RAM's pool has no room for a conversation of " + std::to_string(h.live.size()) + " tokens";
        return false;
    } else {   // the files take every block the cards' pools gave the conversation
        p.file = get_pause_file(p.serial);
        p.file_cut = get_cut();
        p.bytes = h.get_bytes();
    }
    bool ok = true;
    for (size_t k = 0; ok && k < h.parts.size(); ++k) {
        const strata::core::OnDevice on(h.parts[k]->dev);
        const Card& card = cards_[k];
        const strata::core::ConversationId id = h.parts[k]->pages.id();
        ok = in_ram ? strata::core::copy_conversation(card.experts.pool(), id, ram, p.serial, err, get_gpu_span(card),
                                                      get_ram_span(card.lb, card.le))
                    : strata::core::save_conversation(card.experts.pool(), id, get_pause_file(p.serial, k), err,
                                                      get_gpu_span(card));
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
    for (const std::unique_ptr<Part>& part : h.parts)
        part->pages.release();
    paused_.push_back(std::move(p));
    trim_ssd();
    return true;
}

bool PagedEngine::resume(const Paused& p, Held& h, std::string& err) {
    const Clock::time_point t0 = Clock::now();
    if ((!h.has_pages() && !init_pages(h, err)) || !make_room(h, (int64_t) p.live.size(), err))
        return false;
    strata::core::PagePool& ram = ram_.pages().pool();
    // files of the cards' parts at another cut come together in the RAM's pool, whose layout has every layer
    const bool through_ram = !p.file.empty() && p.file_cut != get_cut();
    if (through_ram) {
        const int64_t tokens = h.parts[0]->pages.get_pool_tokens((int64_t) p.live.size());
        if (const std::string why = ram.grow_conversation(p.serial, tokens); !why.empty()) {
            err = "the RAM's pool has no room for a conversation paused before the layer cut moved: " + why;
            return false;
        }
        for (size_t k = 0; k < cards_.size(); ++k) {
            const auto [lb, le] = get_card_layers(p.file_cut, k, m_.g.n_layers);
            if (!strata::core::load_conversation(get_pause_file(p.serial, k), ram, p.serial, err,
                                                 get_ram_span(lb, le))) {
                ram.remove_conversation(p.serial);
                return false;
            }
        }
    }
    bool ok = true;
    for (size_t k = 0; ok && k < h.parts.size(); ++k) {
        const Card& card = cards_[k];
        const strata::core::ConversationId id = h.parts[k]->pages.id();
        const strata::core::OnDevice on(card.dev);
        ok = p.file.empty() || through_ram
                 ? strata::core::copy_conversation(ram, p.serial, card.experts.pool(), id, err,
                                                   get_ram_span(card.lb, card.le), get_gpu_span(card))
                 : strata::core::load_conversation(get_pause_file(p.serial, k), card.experts.pool(), id, err,
                                                   get_gpu_span(card));
    }
    if (through_ram)
        ram.remove_conversation(p.serial);
    if (!ok || !restore_unpaged(p, h, err))
        return false;
    h.live = p.live;
    std::fprintf(stderr, "strata-q38: a conversation of %zu tokens is back from %s (%.0f ms)\n", p.live.size(),
                 p.file.empty() ? "RAM" : "the SSD", get_ms_since(t0));
    return true;
}

bool PagedEngine::save_unpaged(const Held& h, Paused& p, std::string& err) const {
    strata::core::ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(m_.g, z, err))
        return false;
    for (const std::unique_ptr<Part>& part : h.parts) {
        const strata::core::OnDevice on(part->dev);
        std::vector<uint8_t> state;   // the PLE history, then each of the card's QSA layers' (tail, dead, block_pos)
        if (!strata::core::conversation_unpaged_save(part->ss, m_.g, state, err))
            return false;
        const size_t ple = state.size() - (size_t) part->ss.qsa_alloc * (z.tail + z.dead + z.block_pos);
        p.ple.emplace_back(state.begin(), state.begin() + (std::ptrdiff_t) ple);
        p.qsa.insert(p.qsa.end(), state.begin() + (std::ptrdiff_t) ple, state.end());
    }
    return true;
}

bool PagedEngine::restore_unpaged(const Paused& p, Held& h, std::string& err) const {
    strata::core::ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(m_.g, z, err))
        return false;
    const size_t per_layer = z.tail + z.dead + z.block_pos;
    for (size_t k = 0; k < h.parts.size(); ++k) {
        strata::core::SessionState& ss = h.parts[k]->ss;
        const strata::core::OnDevice on(h.parts[k]->dev);
        std::vector<uint8_t> state = p.ple[k];
        const auto first = p.qsa.begin() + (std::ptrdiff_t) ((size_t) ss.qsa_ord0 * per_layer);
        state.insert(state.end(), first, first + (std::ptrdiff_t) ((size_t) ss.qsa_alloc * per_layer));
        if (!strata::core::conversation_unpaged_restore(state, ss, m_.g, p.live, err))
            return false;
    }
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
    std::string err;
    bool ok = true;
    for (size_t k = 0; ok && k < cards_.size(); ++k)
        ok = strata::core::save_conversation(ram_.pages().pool(), p->serial, get_pause_file(p->serial, k), err,
                                             get_ram_span(cards_[k].lb, cards_[k].le));
    drop_pages(*p);
    p->file = get_pause_file(p->serial);
    p->file_cut = get_cut();
    if (ok) {
        trim_ssd();
        return;
    }
    drop_pages(*p);   // the files written before the one that failed
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
    for (size_t k = 0; !p.file.empty() && k < cards_.size(); ++k)
        std::filesystem::remove(get_pause_file(p.serial, k), ec);
    if (p.file.empty())
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

std::string PagedEngine::get_pause_file(uint64_t serial, size_t card) const {
    return pause_prefix_ + std::to_string(serial) + (card > 0 ? "-" + std::to_string(card) : "") + ".bin";
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
        if (const int64_t l = (int64_t) i / n_expert, e = (int64_t) i % n_expert; usage[i] > seen_[i]) {
            Card* const holder = m_.host_res[i] >= 0 ? &get_card_of(l)
                                 : is_lent(l, e)     ? &cards_[(size_t) holder_[(size_t) l]]
                                                     : nullptr;
            if (holder != nullptr)
                holder->experts.note_use({(int32_t) l, (int32_t) e});
        }
    std::vector<strata::core::ExpertSwap> swaps = strata::core::choose_expert_swaps(
        usage, m_.host_res, m_.g.n_layers, n_expert, o.adapt_min_gain,
        [this](int64_t l, int32_t e) { return is_lent(l, e); });
    if (swaps.size() > (size_t) o.adapt_swaps)
        swaps.resize((size_t) o.adapt_swaps);
    std::string err;
    for (const strata::core::ExpertSwap& s : swaps)
        if (!get_card_of(s.layer).experts.replace(s.layer, s.out, s.in, err)) {
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
    int loaded = 0;
    for (Card& card : cards_) {
        uint64_t card_wait_ms = 0;
        const int got = card.experts.is_refill_wanted() ? card.experts.refill(card.refill_batch, card_wait_ms, err) : 0;
        loaded = got < 0 ? -1 : loaded + got;
        if (card_wait_ms > 0 && (wait_ms == 0 || card_wait_ms < wait_ms))
            wait_ms = card_wait_ms;
        if (loaded < 0)
            break;
    }
    if (loaded == 0 && ram_.pages().is_refill_wanted()) {   // the GPUs' first: their experts are the hotter ones
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

bool PagedEngine::is_refill_wanted() const {
    return ram_.pages().is_refill_wanted() ||
           std::any_of(cards_.begin(), cards_.end(), [](const Card& c) { return c.experts.is_refill_wanted(); });
}

bool PagedEngine::publish_residency(std::string& err) {
    if (!residency_changed_)
        return true;
    for (const Card& card : cards_) {
        const strata::core::OnDevice on(card.dev);
        if (const cudaError_t e = m_.put_residency(card.d_res); e != cudaSuccess) {
            err = "the expert residency table of " + card.id + ": " + cudaGetErrorString(e);
            return false;
        }
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
        c.bytes = c.used = h.get_bytes();
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
    const strata::core::PagePoolReport rr = ram_.pages().pool().report();
    strata::serve::MemoryReport report;
    for (const Card& card : cards_) {
        const strata::core::PagePoolReport r = card.experts.pool().report();
        strata::serve::Memory& gpu = report.at(card.id);
        strata::core::read_gpu_memory(card.dev, gpu.name, gpu.total, gpu.free, gpu.unified);
        gpu.add("weights", card.stage != nullptr ? card.stage->dense.allocated_bytes() + card.stage->head.weight_bytes()
                                                 : m_.pool_bytes + m_.native_dense.allocated_bytes() +
                                                       m_.native_head.weight_bytes());
        gpu.add("experts", r.experts.bytes);
        gpu.add("conversations", r.active.bytes + r.fixed.bytes);
        gpu.add("pool_free", r.free.bytes);
        gpu.add("scratch", card.sessions_bytes);
        gpu.pool_pages = r.total.pages;
        gpu.allocated_pages = r.allocated.pages;
        gpu.fragmented_pages = r.fragmented.pages;
        gpu.setting = r.total.bytes;
        gpu.smallest = get_smallest_gpu(card, settings_.tokens_min);
        gpu.largest = get_largest_gpu(card);
    }
    strata::serve::Memory& ram = report.at("ram");
    ram.add("experts", rr.experts.bytes);
    ram.add("pool_free", rr.free.bytes);
    ram.add("scratch", ram_.ring_bytes());
    ram.paused_budget = get_ram_pause_budget();
    ram.pool_pages = rr.total.pages;
    ram.allocated_pages = rr.allocated.pages;
    ram.fragmented_pages = rr.fragmented.pages;
    ram.setting = rr.total.bytes + ram_.ring_bytes();
    ram.mapped = ram_.get_mapped_bytes();
    ram.smallest = get_smallest_ram();
    ram.largest = get_largest_ram();
    uint64_t on_ssd = 0;   // the experts neither pool holds
    for (int64_t l = 0; l < m_.g.n_layers; ++l)
        for (int64_t e = 0; e < m_.g.n_expert; ++e)
            if (m_.host_res[(size_t) (l * m_.g.n_expert + e)] < 0 && !is_lent(l, e) && !ram_.holds(l, e))
                on_ssd += strata::kernels::cpu::expert_layout().blob_bytes(l);
    strata::serve::Memory& ssd = report.at("ssd");
    ssd.add("experts", on_ssd);
    ssd.paused_budget = (uint64_t) settings_.ssd_pause_mib << 20;
    const int64_t probe = std::max<int64_t>(m_.o.max_context, 1);   // tokens whose pages give the average
    uint64_t fixed = 0, grown = 0;   // a conversation's pages on every card: at no token, at `probe` tokens
    for (const Card& card : cards_) {
        const strata::core::PagePool& pool = card.experts.pool();
        fixed += pool.pages_for(0) * pool.page_bytes();
        grown += pool.pages_for(probe) * pool.page_bytes();
    }
    report.settings = {{"context", m_.o.max_context},
                       {"concurrent", settings_.step.concurrent},
                       {"slots", get_most_concurrent()},
                       {"rows", settings_.step.rows},
                       {"lookahead", settings_.step.lookahead},
                       {"tokens_min", settings_.tokens_min},
                       {"tokens_max", settings_.tokens_max},
                       {"ssd_pause_mib", settings_.ssd_pause_mib}};
    for (const Card& card : cards_)
        report.settings.emplace_back("memory_mib." + card.id,
                                     (int64_t) (card.experts.pool().report().total.bytes >> 20));
    report.settings.insert(report.settings.end(),
                           {{"memory_mib.ram", (int64_t) ((rr.total.bytes + ram_.ring_bytes()) >> 20)},
                            {"token_bytes", (int64_t) ((grown - fixed) / probe)},
                            {"conversation_bytes", (int64_t) fixed}});
    if (drafter_ != nullptr)
        report.settings.emplace_back("drafter", settings_.drafter ? 1 : 0);
    report.per_conversation = true;
    return report;
}

std::string PagedEngine::change_settings(const SettingValues& values, std::string& shortfall) {
    PagedSettings s = settings_;
    std::vector<std::optional<uint64_t>> gpu_bytes(cards_.size());   // per card
    std::optional<uint64_t> ram_bytes;
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
        const auto card = std::find_if(cards_.begin(), cards_.end(),
                                       [&](const Card& c) { return name == "memory_mib." + c.id; });
        if (card != cards_.end())
            gpu_bytes[card->index] = round_to_chunks(card->experts.pool(), (uint64_t) v << 20);
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
    std::vector<uint64_t> gpu_now;
    for (const Card& card : cards_) {
        const uint64_t now = gpu_now.emplace_back(card.experts.pool().report().total.bytes);
        const uint64_t smallest = get_smallest_gpu(card, s.tokens_min), asked = gpu_bytes[card.index].value_or(now);
        if (asked < smallest)
            return "the biggest active conversation and the token minimum need " + gib(smallest) + "; the pool of " +
                   card.id + " would keep " + gib(asked);
        if (gpu_bytes[card.index] && asked > get_largest_gpu(card))
            return card.id + " has room for " + gib(get_largest_gpu(card)) + " of pool, not " + gib(asked);
    }
    // (each range takes all of get_commit_room: on the Strix Halo, where the GPU's memory is commit too, one change
    // that grows both pools to their largest can take the commit margin)
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
    const auto note = [&shortfall](const char* what, const std::string& why) {
        if (!why.empty())
            shortfall += (shortfall.empty() ? "" : "; ") + (what + why);
    };
    const uint64_t ram_now = ram_.pages().pool().report().total.bytes + ram_.ring_bytes();
    const auto resize = [&](bool shrink) {
        for (Card& card : cards_)
            if (const std::optional<uint64_t> asked = gpu_bytes[card.index];
                asked && *asked != gpu_now[card.index] && (*asked < gpu_now[card.index]) == shrink)
                note(("the pool of " + card.id + ": ").c_str(), resize_gpu(card, *asked));
        if (ram_bytes && *ram_bytes != ram_now && (*ram_bytes < ram_now) == shrink)
            note("the RAM's pool: ", resize_ram(*ram_bytes));
    };
    resize(true);
    if (s.drafter && !settings_.drafter && !switch_drafter(true, err))
        note("", err);
    resize(false);
    if (std::any_of(gpu_bytes.begin(), gpu_bytes.end(), [](const auto& b) { return b.has_value(); })) {
        const std::vector<int64_t> cut = plan_cut();
        const bool move = cut != get_cut();
        if (move && !move_cut(cut, err))
            note("moving the layer cut: ", err);
        if ((!move || cut != get_cut()) && !holder_.empty()) {   // (a cut move places them for its new cut)
            place_card_experts(true);
            refilling_ = true;
        }
    }
    if (s.tokens_min != min_first)
        note("", set_token_limits(s.tokens_min, s.tokens_max));
    apply_step_settings();
    if (!shortfall.empty())
        std::fprintf(stderr, "strata-q38: CHANGE_SETTINGS changed less than asked: %s\n", shortfall.c_str());
    return {};
}

std::string PagedEngine::resize_gpu(Card& card, uint64_t bytes) {
    const strata::core::OnDevice on(card.dev);
    strata::core::PagePool& pool = card.experts.pool();
    const Clock::time_point t0 = Clock::now();
    const strata::core::PagePoolReport before = pool.report();
    const uint64_t keep = bytes / pool.chunk_bytes() * pool.chunk_bytes() / pool.page_bytes();   // its pages after
    // the conversations that do not decode and would not fit in the smaller pool at all
    while (Held* h = find_idle(nullptr, [&](const Held& x) {
               return x.parts[card.index]->pages.bytes() / pool.page_bytes() > keep;
           }))
        pause_or_forget(*h);
    for (;;) {
        const strata::core::PoolResize r = card.experts.resize(bytes);
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
    refilling_ = refilling_ || pool.report().total.pages > before.total.pages;
    std::string err;
    for (const std::unique_ptr<Held>& h : held_)   // a shrink moved conversation blocks
        if (Part& part = *h->parts[card.index];
            part.pages.is_ready() && (!part.pages.refresh(part.ss, err) || !note_gdn_place(*h, card.index, err)))
            return err;
    if (!publish_residency(err))
        return err;
    report_resize(card.id.c_str(), before, pool.report(), t0);
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
    refilling_ = refilling_ || ram_.pages().pool().report().total.pages > before.total.pages;
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

std::vector<int64_t> PagedEngine::plan_cut() const {
    using namespace strata::core;
    const ModelGeometry& g = m_.g;
    if (cards_.size() < 2)
        return {};
    const std::vector<StateRegion>& whole = ram_.pages().pool().layout().regions;   // (every layer's: open_ram)
    const uint64_t page = cards_[0].experts.pool().page_bytes();
    LayerSizes sizes;
    for (int64_t l = 0, qsa = 0; l < g.n_layers; ++l) {
        const bool is_qsa = is_qsa_layer(g, l);
        sizes.dense.push_back(0);   // (outside the pools)
        sizes.expert.push_back((strata::kernels::cpu::expert_layout().blob_bytes(l) + page - 1) / page * page);
        sizes.fixed_state.push_back(is_qsa ? 0 : whole[0].unit_bytes / (uint64_t) g.n_gdn_layers());
        sizes.token_bytes.push_back(is_qsa ? whole[(size_t) (1 + qsa++)].unit_bytes / whole[1].unit_tokens() : 0);
    }
    int64_t decoding = 0, tokens = settings_.tokens_min;   // the cards keep room for these conversations' state
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding) {
            ++decoding;
            tokens = std::max(tokens, h->dec.pos + h->dec.max_new - h->dec.produced + settings_.step.lookahead + 1);
        }
    std::vector<uint64_t> pools;
    std::vector<double> links;
    for (const Card& card : cards_) {
        pools.push_back(card.experts.pool().report().total.bytes);
        links.push_back(card.link.to_gbps);
    }
    SplitProblem problem = make_paged_split_problem(sizes, pools, std::max<int64_t>(decoding, 1),
                                                    std::min(tokens, m_.o.max_context));
    const std::vector<double> miss_scale = strata::program::split_rules::link_miss_scale(links);
    for (size_t k = 0; k < cards_.size(); ++k)
        problem.cards[k] = {cards_[k].layer_ms, miss_scale[k]};
    for (size_t i = 0; i < m_.profile_all.size() && (int64_t) i < m_.o.expert_cache; ++i)
        problem.ranked_layers.push_back(m_.profile_all[i].first);
    const int64_t interval = std::max<int64_t>(g.qsa_interval, 1);
    problem.can_start = [&](const std::vector<SplitFill>& fills) {
        for (size_t k = 0; k < fills.size(); ++k) {
            const int64_t lb = fills[k].lb, le = fills[k].le, qsa = le / interval - lb / interval;
            if (lb < cards_[k].weights_lb || le > cards_[k].weights_le || qsa == 0 || qsa == le - lb)
                return false;
        }
        return true;
    };
    const SplitPlan plan = plan_layer_split(problem, "strata-q38: the layer cut: ");
    return plan.cuts.empty() || plan.start_unchecked ? get_cut() : plan.cuts;
}

bool PagedEngine::move_cut(const std::vector<int64_t>& cut, std::string& err) {
    using namespace strata::core;
    const Clock::time_point t0 = Clock::now();
    const std::vector<int64_t> before = get_cut();
    std::vector<std::pair<Held*, Paused>> out;   // the conversations paused here
    // puts back those that decode (`all`: every one), the others staying paused
    const auto bring_back = [&](bool all) {
        for (auto& [h, p] : out) {
            if (!all && !h->decoding) {
                paused_.push_back(std::move(p));
                continue;
            }
            std::string why;
            const bool ok = resume(p, *h, why);
            drop_pages(p);
            if (ok)
                continue;
            std::fprintf(stderr, "strata-q38: a conversation could not come back after the layer cut moved, so it is "
                                 "forgotten: %s\n", why.c_str());
            h->forget();
            if (h->decoding)   // (a batch slot's: CHANGE_SETTINGS comes between a request's steps only there)
                end_preempted((size_t) h->slot);
        }
    };
    for (const std::unique_ptr<Held>& h : held_) {
        if (h->live.empty()) {
            for (const std::unique_ptr<Part>& part : h->parts)
                if (part->pages.is_ready())
                    part->pages.release();
            continue;
        }
        if (!pause(*h, err, true)) {
            bring_back(true);
            return false;
        }
        out.emplace_back(h.get(), std::move(paused_.back()));
        paused_.pop_back();
    }
    std::vector<StateLayout> old_layouts;
    for (size_t k = 0; k < cards_.size(); ++k) {
        PagePool& pool = cards_[k].experts.pool();
        const auto [lb, le] = get_card_layers(cut, k, m_.g.n_layers);
        old_layouts.push_back(pool.layout());
        StateLayout layout;
        std::string why;
        if (!get_card_layout(m_, cards_[k], lb, le, layout, why) || !(why = pool.set_layout(layout)).empty()) {
            err = cards_[k].id + "'s pool: " + why;
            for (size_t j = 0; j < k; ++j)
                (void) cards_[j].experts.pool().set_layout(old_layouts[j]);
            bring_back(true);
            return false;
        }
    }
    // the experts lent to other cards leave first: one of a layer its holder takes would stay in `lent`, where the
    // holder's own windows do not look
    if (!holder_.empty())
        place_card_experts(false);
    bool ok = true;
    for (size_t k = 0; k < cards_.size(); ++k) {
        Card& card = cards_[k];
        std::tie(card.lb, card.le) = get_card_layers(cut, k, m_.g.n_layers);
        if (card.stage != nullptr) {
            card.stage->lb = card.lb;
            card.stage->le = card.le;
        }
        session_set_layers(*card.own, m_.g, card.lb, card.le);
        for (const std::unique_ptr<Held>& h : held_) {
            Part& part = *h->parts[k];
            session_set_layers(part.ss, m_.g, card.lb, card.le);
            part.gdn_seen = nullptr;
        }
        set_stage(cards_, k);
        ok = card.ver->drop_batch_graphs(err) && ok;
        split_.end[k] = card.le;
    }
    // a moved layer's experts leave the card it left, even those it is to hold for the layer's new card: the hook put
    // them in the device's table, not in `lent`
    place_card_experts(false);
    if (!holder_.empty())
        place_card_experts(true);
    m_.split_at = cut;
    residency_changed_ = refilling_ = true;
    bring_back(false);
    trim_ram();   // the conversations left paused went to RAM beyond --tokens-max's budget
    if (!ok || !fill_cards(err))
        return false;
    std::string layers;
    for (const Card& card : cards_)
        layers += (layers.empty() ? "" : ", ") + card.id + " layers " + std::to_string(card.lb) + "-" +
                  std::to_string(card.le - 1);
    std::fprintf(stderr, "strata-q38: the layer cut moved: K=%s -> K=%s (%s; %zu conversations paused for it) in %.0f "
                         "ms\n",
                 format_cuts(before).c_str(), format_cuts(cut).c_str(), layers.c_str(), out.size(), get_ms_since(t0));
    return true;
}

bool PagedEngine::fill_cards(std::string& err) {
    for (Card& card : cards_)
        while (card.experts.is_refill_wanted()) {
            uint64_t wait_ms = 0;
            if (card.experts.refill(card.refill_batch, wait_ms, err) < 0)
                return false;
            if (wait_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
        }
    return publish_residency(err);
}

void PagedEngine::place_card_experts(bool lend) {
    using namespace strata::core;
    const int64_t n_layers = m_.g.n_layers, n_expert = m_.g.n_expert;
    ExpertPlacement placed;
    std::vector<ExpertKey> first;   // the profile's first --expert-cache: what one GPU would hold
    for (size_t i = 0; i < m_.profile_all.size() && (int64_t) i < m_.o.expert_cache; ++i)
        first.push_back({m_.profile_all[i].first, m_.profile_all[i].second});
    if (lend && !holder_.empty()) {
        const uint64_t page = cards_[0].experts.pool().page_bytes();
        std::vector<CardPool> pools;   // each card's pages less those kept for --tokens-min
        for (const Card& card : cards_) {
            const PagePool& pool = card.experts.pool();
            const uint64_t pages = pool.report().total.pages;
            pools.push_back({card.lb, card.le, (pages - std::min(pages, pool.pages_for(settings_.tokens_min))) * page});
        }
        std::vector<uint64_t> bytes;
        for (const uint64_t b : get_blob_bytes(m_))
            bytes.push_back((b + page - 1) / page * page);
        placed = place_experts(pools, first, n_layers, n_expert, bytes,
                               [&](int, int holder) { return is_hop_wanted(cards_[(size_t) holder]); });
    }
    if (!holder_.empty())   // in place: SplitDrive::holder points into it
        for (int64_t l = 0; l < n_layers; ++l)
            holder_[(size_t) l] = placed.remote.empty() ? -1 : placed.remote[(size_t) l];
    std::string lenders;
    for (Card& card : cards_) {
        int64_t max_experts = m_.o.expert_cache;
        std::vector<ExpertKey> ranking = get_card_ranking(m_, card.lb, card.le, max_experts);
        if (!placed.card.empty()) {   // those placed on it, hottest first, then the rest of its layers'
            const auto is_on_card = [&](const ExpertKey& k) {
                return placed.card[(size_t) (k.layer * n_expert + k.expert)] == (int32_t) card.index;
            };
            std::vector<ExpertKey> held;
            std::copy_if(first.begin(), first.end(), std::back_inserter(held), is_on_card);
            max_experts = (int64_t) held.size();
            const int64_t others = std::count_if(held.begin(), held.end(), [&](const ExpertKey& k) {
                return k.layer < card.lb || k.layer >= card.le;
            });
            std::remove_copy_if(ranking.begin(), ranking.end(), std::back_inserter(held), is_on_card);
            ranking = std::move(held);
            if (others > 0)
                lenders += (lenders.empty() ? "" : ", ") + card.id + " " + std::to_string(others);
        }
        const OnDevice on(card.dev);
        card.experts.set_ranking(ranking, max_experts);
    }
    if (!lenders.empty())
        std::fprintf(stderr, "strata-q38: experts of other cards' layers, computed where they are in one hop: %s\n",
                     lenders.c_str());
}

bool PagedEngine::is_hop_wanted(const Card& holder) const {
    const uint64_t row = (uint64_t) strata::kernels::cpu::H * sizeof(float);
    return settings_.remote_experts == "on" ||
           strata::core::is_hop_cheaper(strata::core::get_hop_ms(holder.link, settings_.step.rows, row, row), 1,
                                        kCpuExpertMs);
}

std::vector<int64_t> PagedEngine::get_cut() const {
    std::vector<int64_t> cut;
    for (size_t k = 1; k < cards_.size(); ++k)
        cut.push_back(cards_[k].lb);
    return cut;
}

strata::core::RegionSpan PagedEngine::get_gpu_span(const Card& card) const {
    return {1, SIZE_MAX, 0, 0, card.experts.pool().layout().regions[0].unit_bytes};
}

strata::core::RegionSpan PagedEngine::get_ram_span(int64_t lb, int64_t le) const {
    const strata::core::ModelGeometry& g = m_.g;
    const int64_t interval = std::max<int64_t>(g.qsa_interval, 1), q0 = lb / interval, q1 = le / interval;
    const uint64_t gdn_layer = ram_.pages().pool().layout().regions[0].unit_bytes / (uint64_t) g.n_gdn_layers();
    return {(size_t) (1 + q0), le == g.n_layers ? SIZE_MAX : (size_t) (q1 - q0), 0, (uint64_t) (lb - q0) * gdn_layer,
            (uint64_t) (le - q1 - (lb - q0)) * gdn_layer};
}

std::string PagedEngine::set_token_limits(int64_t min, int64_t max) {
    // the conversations that do not decode and hold more tokens than the new maximum
    while (Held* h = find_idle(nullptr, [&](const Held& x) { return (int64_t) x.live.size() > max; }))
        pause_or_forget(*h);
    for (Card& card : cards_) {
        strata::core::PagePool& pool = card.experts.pool();
        if (const std::string why = pool.set_token_limits(min, pool.tokens_max()); !why.empty()) {
            for (size_t k = 0; k < card.index; ++k) {   // the cards before it go back to the old minimum
                strata::core::PagePool& done = cards_[k].experts.pool();
                (void) done.set_token_limits(settings_.tokens_min, done.tokens_max());
            }
            return why;
        }
    }
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

uint64_t PagedEngine::get_smallest_gpu(const Card& card, int64_t tokens_min) const {
    const strata::core::PagePool& pool = card.experts.pool();
    uint64_t pages = pool.pages_for(tokens_min);
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding) {
            const int64_t end = h->dec.pos + h->dec.max_new - h->dec.produced + settings_.step.lookahead + 1;
            const ConversationPages& part = h->parts[card.index]->pages;
            pages = std::max(pages, pool.pages_for(part.get_pool_tokens(std::min(end, m_.o.max_context))));
        }
    const uint64_t chunk = pool.chunk_bytes();
    return std::max<uint64_t>((pages * pool.page_bytes() + chunk - 1) / chunk, 1) * chunk;
}

uint64_t PagedEngine::get_largest_gpu(const Card& card) const {
    const strata::core::PagePool& pool = card.experts.pool();
    const uint64_t chunk = pool.chunk_bytes();
    return (pool.report().total.bytes + get_gpu_room(m_, card, ram_.get_mapped_bytes())) / chunk * chunk;
}

uint64_t PagedEngine::get_smallest_ram() const {
    return ram_.pages().pool().chunk_bytes() + ram_.ring_bytes();
}

uint64_t PagedEngine::get_largest_ram() const {
    const strata::core::PagePool& pool = ram_.pages().pool();
    const uint64_t chunk = pool.chunk_bytes();
    return (pool.report().total.bytes + get_ram_room(m_.o.resident_headroom)) / chunk * chunk + ram_.ring_bytes();
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
        const int S = step_rows_ = (int) rows.size();
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
            Part& part = *h.parts.back();
            const int got = drafter->propose(session, part.pages, part.ss, conv_.ver.final_R_all(), n, out, err);
            if (got >= 0)
                return got;
            std::fprintf(stderr, "strata-q38: the drafter: %s\n", err.c_str());
            return 0;
        };
    return model;
}

}  // namespace
}  // namespace strata::qwen38

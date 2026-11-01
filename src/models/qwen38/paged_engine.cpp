// src/models/qwen38/paged_engine.cpp - strata-q38: conversations in a page pool, several decoding at once
// (paged_engine.hpp).
#include "strata/models/qwen38/paged_engine.hpp"

#include "strata/base/gpu/memory_info.hpp"
#include "strata/common/experts/expert_pages.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/common/serve/loop.hpp"
#include "strata/common/serve/request.hpp"
#include "strata/common/spec/step.hpp"
#include "strata/common/state/progress.hpp"
#include "strata/models/qwen38/conversation.hpp"
#include "strata/models/qwen38/conversation_pages.hpp"
#include "strata/models/qwen38/kernels/verify_kernels.hpp"
#include "strata/models/qwen38/model.hpp"
#include "strata/models/qwen38/options.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata::qwen38 {
namespace {

using Clock = std::chrono::steady_clock;

/// The conversation tokens whose pages experts leave free when --tokens-min is not given.
constexpr int64_t kDefaultTokensMin = 8192;
/// The most evicted experts put back in one step of the serve loop, so that a request waits little for it.
constexpr int kRefillBatch = 16;
/// Between two chunks of a prompt, the conversations that decode take steps for this share of the time the chunk
/// took, so that a long prompt does not stop them (the old engine's STRATA_BATCH_DECODE_SHARE default).
constexpr double kDecodeShare = 0.5;

/// Gets the milliseconds from `t0` to now.
double get_ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// strata-q38's own options (the rest of its command line is `strata`'s).
struct PagedSettings {
    int64_t tokens_min = kDefaultTokensMin;   ///< --tokens-min
    strata::spec::StepSettings step;          ///< --rows and --lookahead; `concurrent` comes from --batch
};

/// A conversation the engine holds: its state, in a session whose GDN state and K/V are page-pool pages, and the
/// tokens in it.  Its session is the Verifier's slot session of the same index.
struct Held {
    strata::core::SessionState ss;
    void* mem = nullptr;                ///< `ss`'s memory (without the state the pages hold)
    ConversationPages pages;            ///< given at the conversation's first request
    std::vector<int32_t> live;          ///< the tokens its state holds: the prompt, then the fed outputs
    strata::spec::Decoding dec;         ///< its decoding, while `decoding`
    bool decoding = false;
    int slot = -1;                      ///< while decoding: its batch slot (BT lines), or -1: it is the request on the
                                        ///< control lines (T lines)
    Clock::time_point t0;               ///< when it started decoding
    uint64_t used = 0;                  ///< the request count when it last served a request (the oldest goes first)

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

/// The engine once its model is up: its conversations, the pages they share with the experts, the step that decodes
/// them, and the requests' protocol.
class PagedEngine {
public:
    /// `slots`: the batch slots (--batch; 0: one conversation at a time).
    PagedEngine(Model& m, Conversation& conv, strata::core::ExpertPages& experts, const PagedSettings& settings,
                int slots);

    /// Carves each conversation's session and gives them to the verify windows; false (said on stderr) when one
    /// does not fit.
    bool init();
    /// Answers the server's commands until QUIT or the end of stdin; returns the exit status.
    int serve();

private:
    /// Runs one GEN or BGEN line; false on a failure that ends the engine (its ERR has been printed).
    bool serve_request(const std::string& line);
    /// Reads `ids[from, to)` into `h` (the request began at `t0`); `read_to` gets how far it got.  Conversations that
    /// decode meanwhile take steps between the prompt's chunks (kDecodeShare).  False with `err` "cancelled" after a
    /// STOP, else on a failure that ends the engine (`fatal`: a decode step failed, its ERR printed).
    bool read_prompt(Held& h, const std::vector<int64_t>& ids, int64_t from, int64_t to, Clock::time_point t0,
                     int64_t& read_to, bool& fatal, std::string& err);
    /// Runs one step of every decoding conversation and prints what it emitted; false (ERR printed) ends the engine.
    bool decode_step();
    /// Moves the idle conversation that holds the longest start of `ids` to place `target` of `at_` (swapping it with
    /// the one there); gets how many of `ids` it holds.
    int64_t take_conversation(size_t target, const std::vector<int64_t>& ids);
    /// Gives `h` the pages for `tokens` tokens, dropping idle conversations (the least recently used first) while the
    /// pool has no room, and tells the kernels about the experts that left those pages.
    bool make_room(Held& h, int64_t tokens, std::string& err);
    /// Puts back a batch of the experts the conversations evicted, or waits until pages open to them: the serve
    /// loop's work between requests.  False when a copy failed (said on stderr).
    bool refill_experts();
    /// Uploads the device's residency table when experts came or left since the last upload.
    bool publish_residency(std::string& err);
    /// Ends batch slot `slot`'s decoding at its next step (BSTOP).
    void stop_slot(int slot);
    /// Takes the BSTOP lines that came while a request runs (BYIELD lines are not taken: a read here never gives way).
    void take_slot_controls();
    /// Tells whether any conversation decodes.
    bool is_decoding() const;
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
    const PagedSettings settings_;
    strata::serve::Input input_;
    strata::spec::Step step_;
    std::vector<std::unique_ptr<Held>> held_;   ///< by session index
    /// held_'s index of the conversation in each batch slot, then of the one a GEN uses ("main")
    std::vector<size_t> at_;
    uint64_t sessions_bytes_ = 0;      ///< the device memory of every session (the engine's own included)
    uint64_t requests_ = 0;            ///< the requests served so far (Held::used)
    const Held* reading_ = nullptr;    ///< the conversation whose prompt read_prompt reads, which make_room keeps
    bool residency_changed_ = false;   ///< `m_.host_res` changed since the device's table was uploaded
};

/// Removes strata-q38's own options from `argv` (the rest is `strata`'s command line).  -1: go on, else exit status.
int take_own_options(int& argc, char** argv, PagedSettings& s) {
    struct Own {
        const char* name;
        int64_t* value;
    };
    int64_t rows = s.step.rows, lookahead = s.step.lookahead;
    const Own own[] = {{"--tokens-min", &s.tokens_min}, {"--rows", &rows}, {"--lookahead", &lookahead}};
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
        *o->value = std::atoll(argv[++i]);
    }
    argc = out;
    s.step.rows = (int) std::clamp<int64_t>(rows, 0, INT32_MAX);
    s.step.lookahead = (int) std::clamp<int64_t>(lookahead, -1, INT32_MAX);
    return -1;
}

/// Opens `m`'s expert cache in the pages of the GPU's page pool, for at most `max_experts` experts
/// (Model::open_expert_pages).  The pool takes the free memory less --vram-reserve-mib and the prompt path's own
/// buffers (on Windows, for a GPU whose memory is the system RAM, what is free of the share the BIOS gives it), but no
/// more than those experts (the profile's first) and a conversation at --max-context take; experts leave the pages of
/// `tokens_min` tokens free.
bool open_pages(Model& m, strata::core::ExpertPages& experts, int64_t tokens_min, int64_t max_experts,
                std::string& err) {
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint64_t> blobs;
    for (int64_t l = 0; l < m.g.n_layers; ++l)
        blobs.push_back(lay.blob_bytes(l));
    const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
    strata::core::StateLayout layout;
    if (!ConversationPages::get_layout(m.g, m.ss, layout, err))
        return false;
    uint64_t most = layout.bytes_for(m.o.max_context) + layout.regions.size() * page;   // + each region's last page
    for (size_t i = 0; i < m.profile.size() && (int64_t) i < max_experts; ++i)
        most += (blobs[(size_t) m.profile[i].first] + page - 1) / page * page;
    const uint64_t keep = ((uint64_t) m.o.vram_reserve_mib << 20) +
        strata::prefill::Prefill::bytes_needed_owned(m.g, m.ss, m.o.prefill_chunk, m.srcp != nullptr);
    uint64_t free_b = strata::core::device_free_bytes();
    if (uint64_t room = 0; strata::core::get_share_room(m.primary_device, room))
        free_b = std::min(free_b, room);
    const uint64_t pool_b = std::min(free_b > keep ? free_b - keep : 0, most);
    m.xcache.set_expert_bytes((int64_t) lay.max_blob);
    strata::core::PagePoolConfig config;
    config.tokens_min = std::clamp<int64_t>(tokens_min, 0, m.o.max_context);
    if (!m.xcache.open((int64_t) (pool_b / page), m.g.n_layers, m.g.n_expert, (int64_t) page, err) ||
        !experts.open(m.xcache, layout, config, blobs, m.g.n_expert, max_experts, err))
        return false;
    std::fprintf(stderr, "strata-q38: the page pool: %.2f GiB (%.2f GiB free) in pages of %llu bytes for the "
                         "experts and the conversations, %lld tokens of it kept for the conversations\n",
                 (double) experts.pool().report().total.bytes / 1073741824.0, (double) free_b / 1073741824.0,
                 (unsigned long long) page, (long long) config.tokens_min);
    return true;
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
        return "--batch (" + std::to_string(s.concurrent) + " conversations at once) must not exceed --rows";
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
    settings.step.concurrent = std::max(slots, 1);
    if (const std::string why = get_step_settings_problem(settings.step); !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    if (settings.step.lookahead > 0)
        std::fprintf(stderr, "strata-q38: --lookahead %d: there is no drafter yet, so no guesses\n",
                     settings.step.lookahead);
    // the scope of this step: no drafting, no K/V streaming or elastic K/V (the pool replaces them); the batch slots
    // are the engine's own conversations, not the old engine's sessions of the whole context
    o.serve = false;
    o.batch = 0;
    o.mtp.clear();
    o.draft.clear();
    o.spec = std::max(settings.step.rows, 2);   // a native pack's verify windows need room for two rows
    o.kv_grow = false;
    o.kv_resident = 0;
    if (o.prefill_chunk <= 0) o.prefill_chunk = 2048;   // open_pages leaves room for the prompt path's own buffers
    if (!o.layer_split.empty() || o.expert_cache_per_layer) {
        std::fprintf(stderr, "strata-q38: %s\n", o.expert_cache_per_layer
                                                      ? "--expert-cache-per-layer: its experts share the page pool"
                                                      : "--layer-split: it runs on one GPU");
        return 2;
    }
    strata::core::session_set_pool_state(true);
    Model m(o);
    strata::core::ExpertPages experts;
    m.open_expert_pages = [&](int64_t max_experts, std::string& err) {
        return open_pages(m, experts, settings.tokens_min, max_experts, err);
    };
    if (const int status = m.load(); status != 0) return status;
    if (const int status = m.build_hit_tables(false); status != 0) return status;
    if (const std::string why = get_unsupported_reason(m); !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    m.set_prompt_pinned_share();
    if (o.resident_cpu_experts)
        if (const int status = m.pin_resident_experts({}, -1); status != 0) return status;
    m.finish_expert_tiers();
    std::string err;
    m.arm_prompt_cpu_share();
    Conversation conv;
    if (!conv.init_prompt(m, m.ss, nullptr, 0, err) || !conv.init_decode(m, m.ss, o.spec, err)) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    PagedEngine engine(m, conv, experts, settings, slots);
    return engine.init() ? engine.serve() : 1;
}

namespace {

PagedEngine::PagedEngine(Model& m, Conversation& conv, strata::core::ExpertPages& experts,
                         const PagedSettings& settings, int slots)
    : m_(m), conv_(conv), experts_(experts), settings_(settings), step_(get_step_model(), m.o.eos_ids) {
    step_.settings = settings.step;
    for (int i = 0; i <= slots; ++i) {
        held_.push_back(std::make_unique<Held>());
        at_.push_back((size_t) i);
    }
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
    experts_.set_hooks(std::move(hooks));
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
    std::printf("INFO context=%lld kv=%s expert_slots=%lld expert_cache_mib=%lld spec=0 prefill_cap=%lld%s engine="
                STRATA_VERSION "\n",
                (long long) o.max_context, o.kv.c_str(), (long long) m_.xcache.resident(),
                (long long) (experts_.pool().report().experts.bytes >> 20), (long long) o.prefill_chunk,
                slots > 0 ? (" batch_slots=" + std::to_string(slots) + " slot_cache=0").c_str() : "");
    std::printf("READY %lld stop conversations memory\n", (long long) o.max_context);
    strata::serve::EngineHooks hooks;
    hooks.is_busy = [this] { return is_decoding() || experts_.is_refill_wanted(); };
    hooks.step = [this] { return is_decoding() ? decode_step() : refill_experts(); };
    hooks.stop_slot = [this](int slot) { stop_slot(slot); };
    hooks.list_conversations = [this] { return list_conversations(); };
    hooks.describe_memory = [this] { return describe_memory(); };
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
    if (req.batch && (req.slot < 0 || (size_t) req.slot >= slots || get_held((size_t) req.slot).decoding)) {
        std::printf("ERR BGEN: no such free slot (--batch %zu)\n", slots);
        return true;
    }
    if (!req.batch && slots > 0 &&
        (size_t) std::count_if(held_.begin(), held_.end(), [](const auto& h) { return h->decoding; }) >= slots) {
        std::printf("ERR strata-q38 decodes %zu conversations at once, and they all are\n", slots);
        return true;
    }
    for (const auto& [key, value] : req.keys)
        (void) strata::serve::read_sampling_key(key, value, smp);   // the engine-specific keys are not this engine's
    const int64_t n = (int64_t) ids.size();
    if (n >= o.max_context) {
        std::printf("ERR the prompt (%lld tokens) does not fit the context (%lld)\n", (long long) n,
                    (long long) o.max_context);
        return true;
    }
    input_.clear_stop();   // a STOP between requests is stale
    const Clock::time_point t0 = Clock::now();
    const size_t target = req.batch ? (size_t) req.slot : slots;
    const int64_t resume = take_conversation(target, ids);
    const size_t session = at_[target];
    Held& h = get_held(target);
    h.used = ++requests_;
    h.dec = strata::spec::Decoding{};
    if (!h.pages.is_ready() &&
        !h.pages.init(m_.g, h.ss, experts_.pool(), (strata::core::ConversationId) session + 1, err)) {
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
    conv_.ver.set_slot_sampling((int) session, sp);
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
        h.dec.session = (int) session;
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
    // the fields of the old engine's DONE that this one has (no drafts, no expert counts)
    std::printf("DONE %lld %lld %.1f %.1f %s 0 0 %lld 0 0 0 0 0.0 %lld 0 chunk=%lld experts_streamed=0\n",
                (long long) h.dec.produced, (long long) n, prompt_ms, get_ms_since(d0), finish, (long long) resume,
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
    conv_.sp.on_chunk = [&](const float*, int64_t T, int64_t pos0, std::string& e) -> bool {
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
    std::vector<strata::spec::Decoding*> active;
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding) {
            going.push_back(h.get());
            active.push_back(&h->dec);
        }
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
        if (std::strcmp(d.finish, "nonfinite") == 0) {   // #879: its state may carry the poison: start over next time
            std::fprintf(stderr, "strata-q38: non-finite logits (#879) at position %lld: the conversation's next "
                                 "request reads its prompt from token 0\n", (long long) d.pos);
            h->forget();
        }
        if (h->slot >= 0)
            std::printf("BDONE %d %lld %s %.1f\n", h->slot, (long long) d.produced, d.finish, get_ms_since(h->t0));
    }
    strata::core::progress_beat();
    return true;
}

int64_t PagedEngine::take_conversation(size_t target, const std::vector<int64_t>& ids) {
    const auto get_prefix = [&](const Held& h) -> int64_t {
        const size_t n = h.live.size();
        return n > 0 && n < ids.size() && std::equal(h.live.begin(), h.live.end(), ids.begin()) ? (int64_t) n : 0;
    };
    size_t best = target;
    int64_t most = get_prefix(get_held(target));
    for (size_t i = 0; i < at_.size(); ++i) {
        const Held& h = get_held(i);
        if (const int64_t prefix = get_prefix(h); !h.decoding && prefix > most) {
            best = i;
            most = prefix;
        }
    }
    std::swap(at_[target], at_[best]);
    return most;
}

bool PagedEngine::make_room(Held& h, int64_t tokens, std::string& err) {
    while (!h.pages.reserve(tokens, err)) {
        Held* idle = nullptr;
        for (const std::unique_ptr<Held>& x : held_)
            if (x.get() != &h && x.get() != reading_ && !x->decoding && !x->live.empty() &&
                (idle == nullptr || x->used < idle->used))
                idle = x.get();
        if (idle == nullptr)
            return false;
        std::fprintf(stderr, "strata-q38: an idle conversation of %zu tokens gives its pages to a growing one\n",
                     idle->live.size());
        idle->forget();
    }
    return publish_residency(err);
}

bool PagedEngine::refill_experts() {
    std::string err;
    uint64_t wait_ms = 0;
    if (experts_.refill(kRefillBatch, wait_ms, err) < 0 || !publish_residency(err)) {
        std::fprintf(stderr, "strata-q38: putting experts back on the GPU: %s\n", err.c_str());
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
    return list;
}

strata::serve::MemoryReport PagedEngine::describe_memory() const {
    const strata::core::PagePoolReport r = experts_.pool().report();
    strata::serve::MemoryReport report;
    strata::serve::Memory& gpu = report.gpu(m_.primary_device);
    strata::core::read_gpu_memory(m_.primary_device, gpu.name, gpu.total, gpu.free, gpu.unified);
    gpu.add("weights", m_.pool_bytes + m_.native_dense.allocated_bytes() + m_.native_head.weight_bytes());
    gpu.add("experts", r.experts.bytes);
    gpu.add("conversations", r.active.bytes + r.fixed.bytes);
    gpu.add("pool_free", r.free.bytes);
    gpu.add("scratch", sessions_bytes_);
    report.at("ram").add("experts", m_.o.mmap_experts ? m_.src.resident_bytes()
                                                      : strata::kernels::cpu::expert_layout().total);
    report.settings = {{"context", m_.o.max_context}, {"concurrent", settings_.step.concurrent},
                       {"rows", settings_.step.rows}, {"lookahead", settings_.step.lookahead},
                       {"tokens_min", settings_.tokens_min}};
    report.per_conversation = true;
    return report;
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
        return true;
    };
    model.keep = [this](const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                        std::string& err) {
        if (rows.size() == kept.size())   // one row each, kept
            return conv_.ver.commit_slots(err);
        std::vector<int> keep(held_.size(), 0);
        for (size_t r = 0, c = 0; r < rows.size(); ++r)
            if (r == 0 || rows[r].session != rows[r - 1].session)
                keep[(size_t) rows[r].session] = kept[c++];
        return conv_.ver.commit_slot_prefixes(keep.data(), err);
    };
    return model;
}

}  // namespace
}  // namespace strata::qwen38

// src/models/qwen38/paged_engine.cpp - strata-q38: one conversation at a time in a page pool (paged_engine.hpp).
#include "strata/models/qwen38/paged_engine.hpp"

#include "strata/base/gpu/memory_info.hpp"
#include "strata/common/experts/expert_pages.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/common/serve/loop.hpp"
#include "strata/common/serve/request.hpp"
#include "strata/common/state/progress.hpp"
#include "strata/models/qwen38/conversation.hpp"
#include "strata/models/qwen38/conversation_pages.hpp"
#include "strata/models/qwen38/model.hpp"
#include "strata/models/qwen38/options.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::qwen38 {
namespace {

using Clock = std::chrono::steady_clock;

/// The conversation tokens whose pages experts leave free when --tokens-min is not given.
constexpr int64_t kDefaultTokensMin = 8192;
/// The most evicted experts put back in one step of the serve loop, so that a request waits little for it.
constexpr int kRefillBatch = 16;

/// Gets the milliseconds from `t0` to now.
double get_ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// The engine once its model is up: the conversation, the pages it shares with the experts, and the requests'
/// protocol.
class PagedEngine {
public:
    PagedEngine(Model& m, Conversation& conv, strata::core::ExpertPages& experts, ConversationPages& pages,
                int64_t tokens_min);

    /// Answers the server's commands until QUIT or the end of stdin; returns the exit status.
    int serve();

private:
    /// Runs one GEN line; false on a failure that ends the engine (its ERR has been printed).
    bool generate(const std::string& line);
    /// Gives the conversation the pages for `tokens` tokens (ConversationPages::reserve) and tells the kernels about
    /// the experts that left them.
    bool reserve(int64_t tokens, std::string& err);
    /// Puts back a batch of the experts the conversation evicted, or waits until pages open to them: the serve loop's
    /// work between requests.  False when a copy failed (said on stderr).
    bool refill_experts();
    /// Uploads the device's residency table when experts came or left since the last upload.
    bool publish_residency(std::string& err);
    /// Describes the conversation for LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS.
    std::vector<strata::core::HeldConversation> list_conversations() const;
    /// Describes the memories for GET_MEMORY_USE.
    strata::serve::MemoryReport describe_memory() const;

    Model& m_;
    Conversation& conv_;
    strata::core::ExpertPages& experts_;
    ConversationPages& pages_;
    const int64_t tokens_min_;
    strata::serve::Input input_;
    std::vector<int32_t> live_;   ///< the tokens whose state the session holds (prompt and fed outputs)
    bool residency_changed_ = false;   ///< `m_.host_res` changed since the device's table was uploaded
};

/// Removes strata-q38's own options from `argv` (the rest is `strata`'s command line).  -1: go on, else exit status.
int take_own_options(int& argc, char** argv, int64_t& tokens_min) {
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tokens-min") != 0) {
            argv[out++] = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "strata-q38: --tokens-min needs a value\n");
            return 2;
        }
        tokens_min = std::atoll(argv[++i]);
    }
    argc = out;
    return -1;
}

/// Opens `m`'s expert cache in the pages of the GPU's page pool, for at most `max_experts` experts, and gives the
/// conversation its fixed region there (Model::open_expert_pages).  The pool takes the free memory less
/// --vram-reserve-mib and the prompt path's own buffers (on Windows, for a GPU whose memory is the system RAM, what is
/// free of the share the BIOS gives it), but no more than those experts (the profile's first) and a conversation at
/// --max-context take; experts leave the pages of `tokens_min` tokens free.
bool open_pages(Model& m, strata::core::ExpertPages& experts, ConversationPages& pages, int64_t tokens_min,
                int64_t max_experts, std::string& err) {
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
        !experts.open(m.xcache, layout, config, blobs, m.g.n_expert, max_experts, err) ||
        !pages.init(m.g, m.ss, experts.pool(), err))
        return false;
    std::fprintf(stderr, "strata-q38: the page pool: %.2f GiB (%.2f GiB free) in pages of %llu bytes for the "
                         "experts and the conversation, %lld tokens of it kept for the conversation\n",
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

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // each protocol line reaches the server as it is printed
    int64_t tokens_min = kDefaultTokensMin;
    if (const int status = take_own_options(argc, argv, tokens_min); status >= 0) return status;
    prepare_process(argc, argv);
    prime_gpu_queues();
    Options o;
    bool have_tokens = false;
    if (const int status = parse_options(argc, argv, o, have_tokens); status >= 0) return status;
    if (const int status = apply_device_defaults(o); status >= 0) return status;
    // the scope of this step: no drafting, no batch slots, no K/V streaming or elastic K/V (the pool replaces them)
    o.serve = false;
    o.batch = 0;
    o.mtp.clear();
    o.draft.clear();
    o.spec = 2;   // a native pack's verify windows need room for two rows; the decode runs them one row at a time
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
    ConversationPages pages;
    m.open_expert_pages = [&](int64_t max_experts, std::string& err) {
        return open_pages(m, experts, pages, tokens_min, max_experts, err);
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
    PagedEngine engine(m, conv, experts, pages, tokens_min);
    return engine.serve();
}

namespace {

PagedEngine::PagedEngine(Model& m, Conversation& conv, strata::core::ExpertPages& experts, ConversationPages& pages,
                         int64_t tokens_min)
    : m_(m), conv_(conv), experts_(experts), pages_(pages), tokens_min_(tokens_min) {
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

int PagedEngine::serve() {
    const Options& o = m_.o;
    const strata::core::PagePoolReport r = pages_.report();
    std::printf("INFO context=%lld kv=%s expert_slots=%lld expert_cache_mib=%lld spec=0 prefill_cap=%lld engine="
                STRATA_VERSION "\n",
                (long long) o.max_context, o.kv.c_str(), (long long) m_.xcache.resident(),
                (long long) (r.experts.bytes >> 20), (long long) o.prefill_chunk);
    std::printf("READY %lld stop conversations memory\n", (long long) o.max_context);
    strata::serve::EngineHooks hooks;
    hooks.is_busy = [this] { return experts_.is_refill_wanted(); };
    hooks.step = [this] { return refill_experts(); };
    hooks.list_conversations = [this] { return list_conversations(); };
    hooks.describe_memory = [this] { return describe_memory(); };
    strata::serve::Loop loop(input_, hooks);
    std::string line;
    while (loop.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) == 0) {
            if (!generate(line)) return 1;
            continue;
        }
        std::printf("ERR strata-q38 does not take %s (one conversation, no batch slots, no session files)\n",
                    line.substr(0, line.find(' ')).c_str());
    }
    return loop.is_failed() ? 1 : 0;
}

bool PagedEngine::generate(const std::string& line) {
    const Options& o = m_.o;
    strata::serve::RequestLine req;
    strata::serve::Sampling smp;
    std::vector<int64_t> ids;
    std::string err;
    if (!strata::serve::parse_request_line(line, req) || req.max_new < 1 ||
        !parse_i64_list(req.ids.c_str(), ids, err)) {
        std::printf("ERR %s\n", err.empty() ? "malformed request (or max_new < 1)" : err.c_str());
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
    // the conversation goes on when the prompt starts with every token the session holds; else a new one
    int64_t resume = (int64_t) live_.size();
    if (resume == 0 || resume >= n || !std::equal(live_.begin(), live_.end(), ids.begin())) {
        resume = 0;
        live_.clear();
        pages_.truncate(0);
        strata::core::session_zero(m_.ss, m_.g, nullptr, m_.main_cs);
        cudaStreamSynchronize(m_.main_stream);
    }
    std::printf("RESUME %lld\n", (long long) resume);
    strata::kernels::SamplerParams sp = get_sampler_params(smp);
    sp.penalty_last_n = 0;   // the penalties need a history per window (the old engine's set_history): not here
    conv_.ver.set_sampling(sp);
    const char* finish = "length";
    int64_t read_to = resume;   // the prompt tokens the session holds
    if (!reserve(n - 1, err)) {   // refused: the engine stays up
        std::printf("ERR %s\n", err.c_str());
        return true;
    }
    if (n - 1 > resume) {
        conv_.sp.on_chunk = [&](const float*, int64_t T, int64_t pos0, std::string& e) -> bool {
            read_to = pos0 + T;
            const double ms = get_ms_since(t0);
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) read_to, (long long) n, ms,
                        ms > 0 ? 1000.0 * (double) (read_to - resume) / ms : 0.0);
            strata::core::progress_beat();
            if (!input_.is_stop_requested()) return true;
            e = "cancelled";
            return false;
        };
        const bool ok = conv_.read_prompt(ids.data() + resume, n - 1 - resume, resume, err);
        conv_.sp.on_chunk = nullptr;
        if (!ok && err != "cancelled") {
            std::printf("ERR %s\n", err.c_str());
            return false;
        }
        if (!ok) finish = "cancel";
    }
    live_.assign(ids.begin(), ids.begin() + read_to);
    const double prompt_ms = get_ms_since(t0);
    std::printf("REUSED %lld\n", (long long) resume);
    const Clock::time_point d0 = Clock::now();
    const int64_t max_new = std::min<int64_t>(req.max_new, o.max_context - n);
    int64_t produced = 0;
    if (read_to == n - 1) {
        int32_t x = (int32_t) ids[(size_t) (n - 1)];
        for (int64_t p = n - 1; produced < max_new; ++p) {
            int32_t next = 0;
            if (!reserve(p + 1, err)) {
                std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
                break;
            }
            if (!conv_.decode(x, p, next, err)) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            live_.push_back(x);
            if (conv_.ver.nonfinite_row() >= 0) {   // #879: the session may carry the poison: start over next time
                live_.clear();
                finish = "nonfinite";
                break;
            }
            std::printf("T %d\n", (int) next);
            strata::core::progress_beat();
            ++produced;
            if (std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end()) {
                finish = "stop";
                break;
            }
            if (input_.is_stop_requested()) {
                finish = "cancel";
                break;
            }
            x = next;
        }
    }
    // the fields of the old engine's DONE that this one has (no drafts, no expert counts)
    std::printf("DONE %lld %lld %.1f %.1f %s 0 0 %lld 0 0 0 0 0.0 %lld 0 chunk=%lld experts_streamed=0\n",
                (long long) produced, (long long) n, prompt_ms, get_ms_since(d0), finish, (long long) resume,
                (long long) (read_to - resume), (long long) o.prefill_chunk);
    return true;
}

bool PagedEngine::reserve(int64_t tokens, std::string& err) {
    return pages_.reserve(tokens, err) && publish_residency(err);
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

std::vector<strata::core::HeldConversation> PagedEngine::list_conversations() const {
    const strata::core::PagePoolReport r = pages_.report();
    strata::core::HeldConversation c;
    c.id = "main";
    c.ids = live_.data();
    c.n = live_.size();
    c.capacity = (uint64_t) m_.o.max_context;
    c.bytes = c.used = r.active.bytes + r.fixed.bytes;
    return {c};
}

strata::serve::MemoryReport PagedEngine::describe_memory() const {
    const strata::core::PagePoolReport r = pages_.report();
    strata::serve::MemoryReport report;
    strata::serve::Memory& gpu = report.gpu(m_.primary_device);
    strata::core::read_gpu_memory(m_.primary_device, gpu.name, gpu.total, gpu.free, gpu.unified);
    gpu.add("weights", m_.pool_bytes + m_.native_dense.allocated_bytes() + m_.native_head.weight_bytes());
    gpu.add("experts", r.experts.bytes);
    gpu.add("conversations", r.active.bytes + r.fixed.bytes);
    gpu.add("pool_free", r.free.bytes);
    gpu.add("scratch", strata::core::session_bytes(m_.g, m_.o.max_context, m_.K));
    report.at("ram").add("experts", m_.o.mmap_experts ? m_.src.resident_bytes()
                                                      : strata::kernels::cpu::expert_layout().total);
    report.settings = {{"context", m_.o.max_context}, {"concurrent", 1}, {"lookahead", 0},
                       {"tokens_min", tokens_min_}};
    return report;
}

}  // namespace
}  // namespace strata::qwen38

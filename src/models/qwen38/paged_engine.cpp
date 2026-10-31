// src/models/qwen38/paged_engine.cpp - strata-q38: one conversation at a time in a page pool (paged_engine.hpp).
#include "strata/models/qwen38/paged_engine.hpp"

#include "strata/base/gpu/memory_info.hpp"
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

/// VRAM left free beside the page pool when its size is not given: hipBLAS and the runtime's own allocations.
constexpr uint64_t kHeadroomBytes = 512ull << 20;

/// Gets the milliseconds from `t0` to now.
double get_ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// The engine once its model is up: the conversation, its pages, and the requests' protocol.
class PagedEngine {
public:
    PagedEngine(Model& m, Conversation& conv, ConversationPages& pages) : m_(m), conv_(conv), pages_(pages) {}

    /// Answers the server's commands until QUIT or the end of stdin; returns the exit status.
    int serve();

private:
    /// Runs one GEN line; false on a failure that ends the engine (its ERR has been printed).
    bool generate(const std::string& line);
    /// Describes the conversation for LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS.
    std::vector<strata::core::HeldConversation> list_conversations() const;
    /// Describes the memories for GET_MEMORY_USE.
    strata::serve::MemoryReport describe_memory() const;

    Model& m_;
    Conversation& conv_;
    ConversationPages& pages_;
    strata::serve::Input input_;
    std::vector<int32_t> live_;   ///< the tokens whose state the session holds (prompt and fed outputs)
};

/// Removes strata-q38's own options from `argv` (the rest is `strata`'s command line).  -1: go on, else exit status.
int take_own_options(int& argc, char** argv, int64_t& conversation_mib) {
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--conversation-mib") != 0) {
            argv[out++] = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "strata-q38: --conversation-mib needs a value\n");
            return 2;
        }
        conversation_mib = std::atoll(argv[++i]);
    }
    argc = out;
    return -1;
}

/// Gets why strata-q38 cannot run the loaded model `m` (empty: it can).
std::string get_unsupported(const Model& m) {
    if (!m.native_pack) return "it needs a native pack (--native), which decodes through verify windows";
    if (m.multi_gpu || !m.stages.empty()) return "it runs on one GPU (no layer split)";
    if (m.thits.d_res == nullptr)
        return "it needs the device residency table (--expert-profile, --expert-cache and the token graph)";
    return {};
}

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int64_t conversation_mib = 0;
    if (const int status = take_own_options(argc, argv, conversation_mib); status >= 0) return status;
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
    if (!o.layer_split.empty()) {
        std::fprintf(stderr, "strata-q38: --layer-split: it runs on one GPU\n");
        return 2;
    }
    if (o.expert_cache < 0) o.vram_reserve_mib += (int) conversation_mib;   // the auto cache leaves the pool its room
    strata::core::session_set_pool_state(true);
    Model m(o);
    if (const int status = m.load(); status != 0) return status;
    if (const int status = m.build_hit_tables(false); status != 0) return status;
    if (const std::string why = get_unsupported(m); !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    if (o.prefill_chunk <= 0) o.prefill_chunk = 2048;
    m.set_prompt_pinned_share();
    if (o.resident_cpu_experts)
        if (const int status = m.pin_resident_experts({}, -1); status != 0) return status;
    m.finish_expert_tiers();
    std::string err;
    ConversationPages pages;
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const uint64_t prompt_b = strata::prefill::Prefill::bytes_needed_owned(m.g, m.ss, o.prefill_chunk,
                                                                                   m.srcp != nullptr);
        const uint64_t room = free_b > prompt_b + kHeadroomBytes ? free_b - prompt_b - kHeadroomBytes : 0;
        const uint64_t pool_b = conversation_mib > 0 ? (uint64_t) conversation_mib << 20 : room;
        if (!pages.init(m.g, m.ss, pool_b, err)) {
            std::fprintf(stderr, "strata-q38: %s (%.2f GiB asked, %.2f GiB free beside a %.2f GiB prompt path)\n",
                         err.c_str(), (double) pool_b / 1073741824.0, (double) room / 1073741824.0,
                         (double) prompt_b / 1073741824.0);
            return 1;
        }
        std::fprintf(stderr, "strata-q38: the conversation's page pool: %.2f GiB\n",
                     (double) pages.report().total.bytes / 1073741824.0);
    }
    m.arm_prompt_cpu_share();
    Conversation conv;
    if (!conv.init_prompt(m, m.ss, nullptr, 0, err) || !conv.init_decode(m, m.ss, o.spec, err)) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    PagedEngine engine(m, conv, pages);
    return engine.serve();
}

namespace {

int PagedEngine::serve() {
    const Options& o = m_.o;
    std::printf("INFO context=%lld kv=%s expert_slots=%lld expert_cache_mib=%lld spec=0 prefill_cap=%lld engine="
                STRATA_VERSION "\n",
                (long long) o.max_context, o.kv.c_str(), (long long) m_.xcache.slots(),
                (long long) (m_.xcache.bytes() >> 20), (long long) o.prefill_chunk);
    std::printf("READY %lld stop conversations memory\n", (long long) o.max_context);
    std::fflush(stdout);
    strata::serve::EngineHooks hooks;
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
        std::fflush(stdout);
    }
    return loop.is_failed() ? 1 : 0;
}

bool PagedEngine::generate(const std::string& line) {
    const Options& o = m_.o;
    strata::serve::RequestLine req;
    strata::serve::Sampling smp;
    std::vector<int64_t> ids;
    std::string err;
    if (!strata::serve::parse_request_line(line, req) || !parse_i64_list(req.ids.c_str(), ids, err)) {
        std::printf("ERR %s\n", err.empty() ? "malformed request" : err.c_str());
        std::fflush(stdout);
        return true;
    }
    for (const auto& [key, value] : req.keys)
        (void) strata::serve::read_sampling_key(key, value, smp);   // the engine-specific keys are not this engine's
    const int64_t n = (int64_t) ids.size();
    if (n >= o.max_context) {
        std::printf("ERR the prompt (%lld tokens) does not fit the context (%lld)\n", (long long) n,
                    (long long) o.max_context);
        std::fflush(stdout);
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
    std::fflush(stdout);
    strata::kernels::SamplerParams sp = get_sampler_params(smp);
    sp.penalty_last_n = 0;   // the penalties need a history per window (the old engine's set_history): not here
    conv_.ver.set_sampling(sp);
    const char* finish = "length";
    int64_t read_to = resume;   // the prompt tokens the session holds
    if (!pages_.reserve(n - 1, err)) {   // refused: the engine stays up
        std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        return true;
    }
    if (n - 1 > resume) {
        conv_.sp.on_chunk = [&](const float*, int64_t T, int64_t pos0, std::string& e) -> bool {
            read_to = pos0 + T;
            const double ms = get_ms_since(t0);
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) read_to, (long long) n, ms,
                        ms > 0 ? 1000.0 * (double) (read_to - resume) / ms : 0.0);
            std::fflush(stdout);
            strata::core::progress_beat();
            if (!input_.is_stop_requested()) return true;
            e = "cancelled";
            return false;
        };
        const bool ok = conv_.read_prompt(ids.data() + resume, n - 1 - resume, resume, err);
        conv_.sp.on_chunk = nullptr;
        if (!ok && err != "cancelled") {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            return false;
        }
        if (!ok) finish = "cancel";
    }
    live_.assign(ids.begin(), ids.begin() + read_to);
    const double prompt_ms = get_ms_since(t0);
    std::printf("REUSED %lld\n", (long long) resume);
    std::fflush(stdout);
    const Clock::time_point d0 = Clock::now();
    const int64_t max_new = std::min<int64_t>(req.max_new, o.max_context - n);
    int64_t produced = 0;
    if (read_to == n - 1) {
        int32_t x = (int32_t) ids[(size_t) (n - 1)];
        for (int64_t p = n - 1; produced < max_new; ++p) {
            int32_t next = 0;
            if (!pages_.reserve(p + 1, err)) {
                std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
                break;
            }
            if (!conv_.decode(x, p, next, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                return false;
            }
            live_.push_back(x);
            if (conv_.ver.nonfinite_row() >= 0) {   // #879: the session may carry the poison: start over next time
                live_.clear();
                finish = "nonfinite";
                break;
            }
            std::printf("T %d\n", (int) next);
            std::fflush(stdout);
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
    std::fflush(stdout);
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
    gpu.add("experts", (uint64_t) m_.xcache.bytes());
    gpu.add("conversations", r.active.bytes + r.fixed.bytes);
    gpu.add("pool_free", r.free.bytes);
    gpu.add("scratch", strata::core::session_bytes(m_.g, m_.o.max_context, m_.K));
    report.at("ram").add("experts", m_.o.mmap_experts ? m_.src.resident_bytes()
                                                      : strata::kernels::cpu::expert_layout().total);
    report.settings = {{"context", m_.o.max_context}, {"concurrent", 1}, {"lookahead", 0}};
    return report;
}

}  // namespace
}  // namespace strata::qwen38

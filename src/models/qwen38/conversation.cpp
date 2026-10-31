// src/models/qwen38/conversation.cpp - one sequence on a loaded Qwen3.8 model (conversation.hpp).
#include "strata/models/qwen38/conversation.hpp"

#include <algorithm>
#include <chrono>

namespace strata::qwen38 {

bool Conversation::init_prompt(Model& m, strata::core::SessionState& ss, void* borrow, uint64_t borrow_bytes,
                               std::string& err) {
    if (!m.multi_gpu && !m.o.no_pool) sp.set_cpu_pool(&*m.cpu_pool);
    return sp.init(m.wt, m.g, ss, m.srcp, m.o.expert_cache > 0 ? &m.xcache : nullptr,
                   m.host_res.empty() ? nullptr : m.host_res.data(), m.o.prefill_chunk, m.main_cs, err, borrow,
                   borrow_bytes);
}

bool Conversation::init_decode(Model& m, strata::core::SessionState& ss, int max_rows, std::string& err) {
    m_ = &m;
    if (!ver.init(m.wt, m.g, ss, m.get_verify_hits(), m.native_head.loaded() ? &m.native_head : nullptr, max_rows, err))
        return false;
    ver.set_split(m.o.spec_split);
    // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
    // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
    // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
    // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
    ver.set_pcie_mode(m.o.pcie_mode == "dma" ? 0 : m.o.pcie_mode == "direct" ? 1 : 2);
    m.drive.d.plan = ver.plan_sink();
    m.drive.d.pcie_num = std::clamp((int) (m.o.pcie_frac * 256.0 + 0.5), 0, 256);
    return true;
}

bool Conversation::run_window(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string& err) {
    strata::core::ExpertDispatch& d = m_->drive.d;
    d.layers = 0;
    d.experts = 0;
    d.failed = false;
    if (!ver.run(T, tokens, pos0, &drive_pool_multi, &m_->drive, out, err))
        return false;
    if (d.failed) {
        err = "the expert pool failed at layer " + std::to_string(d.fail_layer) + " expert " +
              std::to_string(d.fail_expert) + ": " + (d.fail ? d.fail : "(no message)");
        return false;
    }
    return true;
}

strata::kernels::SamplerParams get_sampler_params(const strata::serve::Sampling& s) {
    strata::kernels::SamplerParams sp;
    sp.greedy = s.temperature <= 0.0f;
    sp.temperature = s.temperature;
    sp.top_p = s.top_p;
    sp.top_k = s.top_k;
    sp.seed = s.seed ? s.seed : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
    sp.min_p = std::clamp(s.min_p, 0.0f, 1.0f);
    sp.penalty_last_n = std::max(s.penalty_last_n, 0);
    sp.penalty_repeat = s.penalty_repeat;
    sp.penalty_freq = s.penalty_freq;
    sp.penalty_present = s.penalty_present;
    sp.counter = 0;
    return sp;
}

}  // namespace strata::qwen38

// src/models/qwen38/conversation_drafter.cpp - Qwen3.8's MTP drafter for strata-q38's conversations
// (conversation_drafter.hpp).
#include "strata/models/qwen38/conversation_drafter.hpp"

#include "strata/common/spec/window_planner.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace strata::qwen38 {

ConversationDrafter::~ConversationDrafter() { (void) cudaFree(rows_); }

bool ConversationDrafter::init(Model& m, int sessions, std::string& err) {
    mtp_ = &m.mtp;
    min_p_ = m.o.spec_min_p;
    row_floats_ = (uint64_t) m.g.hc * (uint64_t) m.g.n_embd;
    const int max_t = mtp_->max_t();
    if (cudaMalloc((void**) &rows_, (size_t) max_t * row_floats_ * sizeof(float)) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "the drafter's residual rows do not fit";
        return false;
    }
    drafts_.resize((size_t) max_t);
    probs_.resize((size_t) max_t);
    last_.assign((size_t) sessions, Last{});
    return mtp_->bind(m.wt, &m.native_head, rows_, err);
}

bool ConversationDrafter::read_prompt(strata::prefill::Prefill& sp, const ConversationPages& pages,
                                      strata::core::SessionState& ss, const float* R_rows, const int32_t* next,
                                      int64_t n, int64_t cell0, int64_t prompt_len, std::string& err) {
    if (!select(pages, ss, prompt_len, err))
        return false;
    // the prompt path's batched pass where it can (E-9), else the drafter's own
    return sp.draft_kv(*mtp_, R_rows, next, n, cell0, err) ||
           (err.empty() && mtp_->prefill(R_rows, next, n, cell0, err));
}

void ConversationDrafter::note_window(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                                      const int32_t* picks) {
    for (Last& l : last_)
        l.rows = 0;
    for (size_t r = 0, c = 0; r < rows.size(); ++c) {
        Last& l = last_.at((size_t) rows[r].session);
        l.row0 = (int) r;
        l.pos = rows[r].pos;
        l.kept = kept[c];
        while (r < rows.size() && rows[r].session == rows[(size_t) l.row0].session)
            ++r;
        l.rows = (int) r - l.row0;
        l.picks.assign(picks + l.row0, picks + r);
    }
}

int ConversationDrafter::propose(int session, const ConversationPages& pages, strata::core::SessionState& ss,
                                 const float* window_R, int n, int32_t* out, std::string& err) {
    Last& l = last_.at((size_t) session);
    if (l.rows == 0)
        return 0;
    const int rows = l.rows;
    l.rows = 0;
    if (!select(pages, ss, 0, err))
        return -1;
    const float* R = window_R + (size_t) l.row0 * row_floats_;
    if (n == 0)   // its kept rows' cells only: row t pairs with the token picked after it
        return mtp_->prefill(R, l.picks.data(), l.kept, l.pos, err) ? 0 : -1;
    mtp_->set_source_R(R);
    mtp_->set_max_drafts(n);
    int got = 0;
    if (!mtp_->draft(rows, l.picks.data(), l.pos, l.kept - 1, drafts_.data(), err, probs_.data(), (float) min_p_,
                     &got))
        return -1;
    const int k = strata::spec::count_confident_drafts(probs_.data(), std::min(got, n), min_p_);
    std::copy_n(drafts_.begin(), k, out);
    return k;
}

bool ConversationDrafter::select(const ConversationPages& pages, strata::core::SessionState& ss, int64_t prompt_len,
                                 std::string& err) {
    mtp_->set_ple_session(&ss);
    mtp_->set_prompt_len(prompt_len);
    return mtp_->use_kv_units(pages.get_drafter_units(), err);
}

}  // namespace strata::qwen38

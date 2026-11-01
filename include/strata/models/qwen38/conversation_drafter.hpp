// include/strata/models/qwen38/conversation_drafter.hpp - Qwen3.8's MTP drafter for strata-q38's conversations
// (docs/MEMORY_MANAGER_PLAN.md, step M4, "Guesses in the page pool").
//
// One drafter (Model::mtp: its weights, buffers and graphs) serves every conversation in turn.  Its K/V, the only
// state it keeps per conversation, is one more region of each conversation in the page pool (ConversationPages), so
// a conversation's guesses write into its own pages and a rejected guess's cells are simply written again later.
// The drafter reads a step's rows one step late: what a conversation's last step kept is read into its K/V when the
// next step asks it for guesses (the same pass drafts them), so that the guesses' count is the next step's share.
#pragma once

#include "strata/common/spec/step.hpp"
#include "strata/models/qwen38/conversation_pages.hpp"
#include "strata/models/qwen38/model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen38 {

/// Proposes the guesses of strata-q38's conversations with the MTP drafter, whose K/V each conversation keeps in its
/// own pages.  Conversations are the step's sessions (strata::spec::StepRow::session).
class ConversationDrafter {
public:
    ConversationDrafter() = default;
    ~ConversationDrafter();
    ConversationDrafter(const ConversationDrafter&) = delete;
    ConversationDrafter& operator=(const ConversationDrafter&) = delete;

    /// Binds `m`'s drafter (Model::mtp, loaded with set_kv_in_pages) to the model's head and readies it for
    /// `sessions` conversations; the guesses keep to --spec-min-p.  Called again for a drafter loaded again.
    bool init(Model& m, int sessions, std::string& err);
    /// Writes the drafter's K/V of the prompt cells [cell0, cell0 + n) of a conversation (its `pages` and session
    /// `ss`) from the prompt path `sp`'s final residual rows `R_rows`: cell i pairs row i with `next[i]`, the token
    /// after it.  Cells that a prompt of `prompt_len` tokens leaves out of the drafter's attention window are skipped.
    /// For the prompt path's on_chunk.
    bool read_prompt(strata::prefill::Prefill& sp, const ConversationPages& pages, strata::core::SessionState& ss,
                     const float* R_rows, const int32_t* next, int64_t n, int64_t cell0, int64_t prompt_len,
                     std::string& err);
    /// Remembers what the verify window that just ran kept of each conversation in it: `rows` (a conversation's rows
    /// consecutive), the rows each kept (`kept`, in the order of `rows`), and `picks`, the window's token after each
    /// row.  Forgets the windows before it.
    void note_window(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                     const int32_t* picks);
    /// Forgets what the last window kept of conversation `session` (its state is about to change otherwise).
    void forget(int session) { last_.at((size_t) session).rows = 0; }
    /// Reads what the last window kept of conversation `session` (its `pages` and session `ss`) into its K/V, from
    /// `window_R`, that window's final residual rows; then writes up to `n` guesses of the tokens after its last kept
    /// one to `out` and gets how many (0 also when the window did not hold it).  `n` 0 only reads.  -1 (with `err`)
    /// when the drafter failed.
    int propose(int session, const ConversationPages& pages, strata::core::SessionState& ss, const float* window_R,
                int n, int32_t* out, std::string& err);

private:
    /// What the last window kept of one conversation.
    struct Last {
        int row0 = 0;       ///< its first row in the window
        int rows = 0;       ///< its rows; 0: the window did not hold it, or it was read already
        int kept = 0;
        int64_t pos = 0;    ///< its first row's position
        std::vector<int32_t> picks;   ///< the window's token after each of its rows
    };

    /// Points the drafter at a conversation's K/V, PLE history and prompt length (0 when decoding).
    bool select(const ConversationPages& pages, strata::core::SessionState& ss, int64_t prompt_len, std::string& err);

    strata::core::MtpDrafter* mtp_ = nullptr;
    double min_p_ = 0.0;
    uint64_t row_floats_ = 0;   ///< the floats of one residual row
    float* rows_ = nullptr;     ///< the bound residual rows: a conversation's rows of the window, copied in by draft()
    std::vector<Last> last_;    ///< per session
    std::vector<int32_t> drafts_;
    std::vector<float> probs_;
};

}  // namespace strata::qwen38

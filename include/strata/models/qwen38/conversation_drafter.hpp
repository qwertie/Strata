// include/strata/models/qwen38/conversation_drafter.hpp - Qwen3.8's MTP drafter for strata-q38's conversations
// (docs/MEMORY_MANAGER_PLAN.md, step M4, "Guesses in the page pool").
//
// One drafter (Model::mtp: its weights, buffers and graphs) serves every conversation in turn.  Its K/V, the only
// persistent model state it keeps per conversation, is one more region of each conversation in the page pool
// (ConversationPages), so a conversation's guesses write into its own pages and a rejected guess's cells are simply
// written again later.
// The drafter reads a step's rows one step late: what a conversation's last step kept is read into its K/V when the
// next step asks it for guesses (the same pass drafts them), so that the guesses' count is the next step's share.
// Pipelined groups retain transient residual snapshots until that read, since later groups reuse verifier scratch.
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

    /// Binds `m`'s drafter (Model::mtp, loaded with set_kv_in_pages) to the head of its card and readies it for
    /// `sessions` conversations. Called again for a drafter loaded again.
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
    /// row. With residuals, preserves other conversations and copies these rows for the next proposal; otherwise
    /// forgets earlier windows.
    bool note_window(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                     const int32_t* picks, std::string& err, const float* residuals = nullptr);
    /// Gets device bytes retained for per-conversation residual snapshots.
    uint64_t get_snapshot_bytes() const {
        return saved_rows_ ? last_.size() * (size_t) max_rows_ * row_floats_ * sizeof(float) : 0;
    }
    /// Reserves per-conversation copies before a wave, preserving pending rows from the last synchronous
    /// verifier output in `residuals`. Ownership remains with each conversation through live concurrency changes.
    bool reserve_snapshots(const float* residuals, std::string& err);
    /// Forgets what the last window kept of conversation `session` (its state is about to change otherwise).
    void forget(int session) { last_.at((size_t) session).rows = 0; }
    /// Reads what the last window kept of conversation `session` (its `pages` and session `ss`) into its K/V, from
    /// `window_R`, that window's final residual rows; then writes up to `n` guesses of the tokens after its last kept
    /// one to `out` and gets how many (0 also when the window did not hold it).  `n` 0 only reads.  -1 (with `err`)
    /// when the drafter failed. `draft` supplies the shared stopping callback and receives raw confidences.
    int propose(int session, const ConversationPages& pages, strata::core::SessionState& ss, const float* window_R,
                int n, int32_t* out, std::string& err, strata::spec::DraftContext* draft);

private:
    /// What the last window kept of one conversation.
    struct Last {
        bool saved = false; ///< Uses its owned residual rows rather than the verifier's current scratch.
        int row0 = 0;       ///< its first row in the window
        int rows = 0;       ///< its rows; 0: the window did not hold it, or it was read already
        int kept = 0;
        int64_t pos = 0;    ///< its first row's position
        std::vector<int32_t> picks;   ///< the window's token after each of its rows
    };

    /// Preserves one conversation's pending verifier output before scratch is reused.
    bool save_residuals(int session, const float* residuals, std::string& err);
    /// Points the drafter at a conversation's K/V, PLE history and prompt length (0 when decoding).
    bool select(const ConversationPages& pages, strata::core::SessionState& ss, int64_t prompt_len, std::string& err);

    strata::core::MtpDrafter* mtp_ = nullptr;
    uint64_t row_floats_ = 0;   ///< the floats of one residual row
    float* saved_rows_ = nullptr; ///< Per-session residual snapshots for pipelined windows.
    int max_rows_ = 0;
    float* rows_ = nullptr;     ///< the bound residual rows: a conversation's rows of the window, copied in by draft()
    int device_ = 0;            ///< the drafter's GPU, which `rows_` is on (a layer split's last card's)
    std::vector<Last> last_;    ///< per session
    std::vector<int32_t> drafts_;
    std::vector<float> probs_;
};

}  // namespace strata::qwen38

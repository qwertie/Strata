// include/strata/common/spec/step.hpp - the decode step for several conversations at once: one pass through the
// model computes a row for each active conversation's next token and for its guesses (docs/MEMORY_MANAGER_PLAN.md,
// "The step" and "How rows are shared").  It is model-neutral: the model runs the rows (StepModel).  strata-q38
// (src/models/qwen38/paged_engine.cpp) decodes through it.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::spec {

/// The settings that shape a step.
struct StepSettings {
    int concurrent = 1;   ///< the most conversations that decode in the same step
    int rows = 8;         ///< the most rows one step computes; at least `concurrent`
    int lookahead = 0;    ///< the most guesses one conversation may add to a step; 0 turns drafting off
};

/// Gets each of `active` conversations' share of the rows a step has left once each has its one row: as even as
/// possible and at most `s.lookahead` each.  The conversations are in priority order and the first `ties` of them
/// have the same priority.  The extra rows of an uneven split go to those `ties` first, from the one at `turn %
/// ties` on (wrapping among them), then to the others in order; so a caller that moves `turn` along rotates the
/// extra rows between equals.
std::vector<int> share_guess_rows(const StepSettings& s, int active, int ties, int turn);

/// One row of a step: a token of one conversation at its position.
struct StepRow {
    int session = 0;   ///< the model's state that the row reads and extends (Decoding::session)
    int32_t token = 0;
    int64_t pos = 0;
    int parent = -1;   ///< the row whose token this one follows; -1: the conversation's last kept token
};

/// What the model does for a step.
struct StepModel {
    /// Gives `session` room for its first `tokens` tokens; false when there is none (the conversation then ends).
    std::function<bool(int session, int64_t tokens)> make_room;
    /// Runs `rows` in one pass through the model: picks[r] gets the token that follows row r, or -1 when row r's
    /// logits were not finite.  A conversation's rows are consecutive.  False (with `err`) ends the engine.
    std::function<bool(const std::vector<StepRow>& rows, std::vector<int32_t>& picks, std::string& err)> run;
    /// Makes the first `kept[c]` rows of the c-th conversation in `rows` (the last run's) part of its state and
    /// forgets the rest.  False (with `err`) ends the engine.
    std::function<bool(const std::vector<StepRow>& rows, const std::vector<int>& kept, std::string& err)> keep;
    /// Writes up to `n` guesses of the tokens that follow `token` at `pos` in `session` to `out`; gets how many.  It
    /// is called before each conversation's rows of a step, also with `n` 0, and with `n` 0 (and `out` null) when
    /// a conversation ends, so that a drafter can read what the conversation's last step kept.  Empty: the model has
    /// no drafter.
    std::function<int(int session, int32_t token, int64_t pos, int n, int32_t* out)> propose;
};

/// A conversation that decodes in steps, and what its last step did.
struct Decoding {
    int session = 0;        ///< the model's state of it, which the model numbers
    int32_t next = 0;       ///< the token the next step feeds, which is not in the state yet
    int64_t pos = 0;        ///< `next`'s position
    int64_t produced = 0;   ///< the tokens it has emitted
    int64_t max_new = 0;    ///< the most tokens it may emit; also keeps its rows inside the model's context
    bool stop = false;      ///< end it at the next step, without a row
    /// Steps take conversations of higher priority first; among equal ones they take turns at the extra guess rows.
    int64_t priority = 0;
    int64_t guessed = 0;         ///< the guesses its steps ran
    int64_t guessed_right = 0;   ///< the guesses it kept: the model picked the same tokens

    std::vector<int32_t> emitted;   ///< the last step's tokens for the client, in order
    std::vector<int32_t> fed;       ///< the tokens the last step added to its state
    /// Why the last step ended it ("stop": an end token, "length", "cancel", "nonfinite": its logits were not
    /// finite and its state may be poisoned); null while it goes on.
    const char* finish = nullptr;
};

/// Runs the steps of the conversations that decode together.
class Step {
public:
    /// `end_tokens`: the tokens that end a conversation (emitted, then "stop").
    Step(StepModel model, std::vector<int64_t> end_tokens) : model_(std::move(model)), end_(std::move(end_tokens)) {}

    StepSettings settings;

    /// Runs one step over `active` (in Decoding::priority order, highest first; at most settings.concurrent): shares
    /// the rows, asks the drafter for guesses, runs the rows, keeps each conversation's accepted ones and updates it.
    /// A conversation that has `stop` set, or no room, ends without a row.  False (with `err`) when the model failed.
    bool run(const std::vector<Decoding*>& active, std::string& err);

private:
    /// Gets how many of the `n` rows of `d` (its token, then its guesses) from `picks` it keeps, and sets its
    /// emitted tokens and finish.
    int accept(Decoding& d, const StepRow* rows, const int32_t* picks, int n) const;
    /// Asks the drafter for no guesses for `d`, which ends at this step, so that it reads what d's last step kept.
    void end_drafting(const Decoding& d) const;
    /// Tells whether `token` is one of the end tokens.
    bool is_end(int32_t token) const;

    StepModel model_;
    std::vector<int64_t> end_;
    int turn_ = 0;   ///< share_guess_rows' turn: which of the highest-priority equals gets the next extra row
};

}  // namespace strata::spec

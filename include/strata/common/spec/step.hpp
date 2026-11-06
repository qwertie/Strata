// include/strata/common/spec/step.hpp - the decode step for several conversations at once: each group makes one
// pass through the model for its conversations' next tokens and guesses (docs/MEMORY_MANAGER_PLAN.md,
// "The step" and "How rows are shared").  It is model-neutral: the model runs the rows (StepModel).  The paged engine
// (src/common/serve/paged_engine.cpp) decodes through it.
#pragma once

#include "strata/common/spec/window_planner.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::spec {

/// The settings that shape a step.
struct StepSettings {
    int concurrent = 1;   ///< the most conversations that decode in the same step
    int rows = 8;         ///< the most rows each group computes; at least its active conversation count
    int lookahead = 0;    ///< the most guesses one conversation may add to a step; 0 turns drafting off
    int model_lookahead = -1; ///< --mtp-max-t minus one; negative leaves the model draft width uncapped.
    int max_draft_conversations = 0; ///< Above this count guesses are suppressed; 0 imposes no limit.
    int groups = 1;       ///< The pipeline groups selected from live concurrency; 1 keeps the serial path.
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

/// Selects confidence floors and the model's capacity restrictions for the shared step.
struct DraftLimits {
    float confidence = 0.0f, cumulative = 0.0f;
    int max_conversations = 0; ///< Zero imposes no limit on simultaneous drafting.
    bool cool_after_rejection = false; ///< Allows a plain step after the whole model proposal failed its cut.
};

/// Supplies the shared stopping rule and receives the model drafter's confidences for one proposal.
struct DraftContext {
    WindowPlanner* planner = nullptr;
    float* probs = nullptr; ///< Receives one confidence per returned token; storage for n floats.
    DraftLimits limits;
    ConfidenceCut cut;
    /// Leaves the network's full-chain fast path available when no stopping rule is configured.
    std::function<bool(int, float)> get_stopping_rule() {
        if (limits.confidence <= 0 && limits.cumulative <= 0 && (!planner || !planner->cost_aware))
            return {};
        return [this](int idx, float p) { return keep_draft(idx, p); };
    }
    /// Applies the shared confidence and cost rules while an iterative network produces its proposal.
    bool keep_draft(int idx, float p) {
        if (idx == 0)
            cut = ConfidenceCut{limits.confidence, limits.cumulative};
        const bool confident = cut.keep(p);
        if (!planner || !planner->cost_aware)
            return confident;
        cost_planned = true;
        return planner->keep_draft(idx, p) && confident;
    }
    bool cost_planned = false; ///< Prevents replaying an already evaluated cost chain.
};

/// Carries one conversation's row allowance through an optional shared drafter forward pass.
struct DraftRequest {
    int session = 0;
    int32_t token = 0;
    int64_t pos = 0;
    int limit = 0, count = 0;
    std::vector<int32_t> tokens;
    std::vector<float> confidence;
    DraftContext context;
};

/// Owns one pipeline group's rows and the tokens picked after them.
struct StepGroup {
    std::vector<StepRow> rows;
    std::vector<int32_t> picks;
};

/// Supplies nonblocking stage operations to the shared pipeline scheduler.
struct StageModel {
    int stages = 0;
    int* peak_in_flight = nullptr; ///< Receives the most stages occupied together in this wave.
    std::function<bool(int, StepGroup&, std::string&)> prepare;
    std::function<bool(int stage, int group, std::string&)> launch;
    /// Returns 1 when complete, 0 while running, or -1 on failure.
    std::function<int(int stage, int group, StepGroup&, std::string&)> poll;
    /// Finalizes picks once every stage is idle; optional for models that already returned them from poll.
    std::function<bool(int group, StepGroup&, std::string&)> finish;
};

/// Prepares every group before launching, then drains all stages before returning, including on failure.
bool run_stage_groups(StageModel model, std::vector<StepGroup>& groups, std::string& err);

/// What the model does for a step.
struct StepModel {
    /// Prepares the model for the number of conversations sharing this pass.
    std::function<void(int)> begin;
    /// Gives `session` room for its first `tokens` tokens; false retries without guesses, then ends if none fit.
    std::function<bool(int session, int64_t tokens)> make_room;
    /// Runs `rows` in one pass through the model: picks[r] gets the token that follows row r, or -1 when row r's
    /// logits were not finite.  A conversation's rows are consecutive.  False (with `err`) ends the engine.
    std::function<bool(const std::vector<StepRow>& rows, std::vector<int32_t>& picks, std::string& err)> run;
    /// Runs disjoint conversation groups through the model's stages, returning only after they drain; needs keep_group.
    std::function<bool(std::vector<StepGroup>&, std::string&)> run_groups;
    /// Commits one drained group's accepted prefixes.
    std::function<bool(int, const std::vector<StepRow>&, const std::vector<int>&, std::string&)> keep_group;
    /// Makes the first `kept[c]` rows of the c-th conversation in `rows` (the last run's) part of its state and
    /// forgets the rest.  False (with `err`) ends the engine.
    std::function<bool(const std::vector<StepRow>& rows, const std::vector<int>& kept, std::string& err)> keep;
    /// Writes up to `n` guesses of the tokens that follow `token` at `pos` in `session` to `out`; gets how many.  It
    /// is called before each conversation's rows of a step, also with `n` 0, and with `n` 0 (and `out` null) when
    /// a conversation ends, so that a drafter can read what the conversation's last step kept.  Empty: the model has
    /// no drafter. `draft` supplies the window policy and receives probabilities; null only when decoding ends.
    std::function<int(int session, int32_t token, int64_t pos, int n, int32_t* out, DraftContext* draft)> propose;
    /// Runs the network once for all eligible requests; unset preserves the individual proposal path.
    std::function<void(std::vector<DraftRequest>&)> propose_batch;
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

    DraftLimits draft_limits; ///< This request's confidence cuts and drafting restrictions.
    bool draft_enabled = true; ///< The request's spec key; false disables every source.
    bool draft_cooling = false; ///< The model skips one eligible step after an empty confidence cut.
    std::unique_ptr<WindowPlanner> planner; ///< Keeps learned costs across requests; its token index is rebuilt.
    int64_t suffix_offered = 0, suffix_kept = 0; ///< Lookup guesses verified and accepted in this request.

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
    /// Pipeline groups prepare separately and all drain before acceptance. A conversation that has `stop` set,
    /// or no room, ends without a row. False (with `err`) when the model failed.
    bool run(const std::vector<Decoding*>& active, std::string& err);

private:
    /// Keeps proposals alive until every group's stages have drained.
    struct Work {
        StepGroup group;
        std::vector<Window> windows;
        std::vector<Decoding*> stepping;
        std::vector<size_t> start;
    };
    /// Builds a group's windows before any pipeline work starts; -1 uses the serial rotation.
    /// wave_conversations includes peers whose windows belong to other groups.
    void prepare(const std::vector<Decoding*>& active, Work& work, int group = -1, int wave_conversations = 0);
    /// Applies the model's picks after its stages have drained.
    bool complete(Work& work, int group, double ms, std::string& err);
    /// Gets how many of the `n` rows of `d` (its token, then its guesses) from `picks` it keeps, and sets its
    /// emitted tokens and finish.
    int accept(Decoding& d, const StepRow* rows, const int32_t* picks, int n) const;
    /// Asks the drafter for no guesses for `d`, which ends at this step, so that it reads what d's last step kept.
    void end_drafting(const Decoding& d) const;
    /// Tells whether `token` is one of the end tokens.
    bool is_end(int32_t token) const;

    StepModel model_;
    std::vector<int64_t> end_;
    /// Keeps each group's extra-row rotation independent of the number of groups.
    std::vector<int> group_turns_;
    int turn_ = 0;   ///< share_guess_rows' turn: which of the highest-priority equals gets the next extra row
};

}  // namespace strata::spec

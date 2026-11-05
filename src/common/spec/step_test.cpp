// src/common/spec/step_test.cpp - the decode step's row sharing and acceptance (step.hpp), with a model that answers
// each token with the token after it (no GPU).
#include "strata/common/spec/step.hpp"
#include "strata/common/serve/penalty_history.hpp"

#include <cstdio>
#include <map>

using namespace strata::spec;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

/// A model whose pick after token x is x + 1 (-1 in the sessions in `poisoned`), and whose drafter guesses right
/// `right` times and then guesses 0.
struct FakeModel {
    std::vector<StepRow> last;
    std::vector<int> kept;
    std::map<int, int64_t> room;   ///< per session, the tokens make_room gave it
    int64_t room_limit = 1 << 20;
    int poisoned = -1;
    int right = 0;
    int active = 0;   ///< Conversations reported before drafting begins.

    StepModel hooks() {
        StepModel m;
        m.begin = [this](int count) { active = count; };
        m.make_room = [this](int session, int64_t tokens) {
            room[session] = tokens;
            return tokens <= room_limit;
        };
        m.run = [this](const std::vector<StepRow>& rows, std::vector<int32_t>& picks, std::string&) {
            last = rows;
            for (size_t r = 0; r < rows.size(); ++r)
                picks[r] = rows[r].session == poisoned ? -1 : rows[r].token + 1;
            return true;
        };
        m.keep = [this](const std::vector<StepRow>&, const std::vector<int>& k, std::string&) {
            kept = k;
            return true;
        };
        return m;
    }
    std::vector<int> asked;   ///< the `n` of each propose call
    void add_drafter(StepModel& m) {
        m.propose = [this](int, int32_t token, int64_t, int n, int32_t* out, DraftContext* draft) {
            check(active > 0 || n == 0, "preparation precedes requests for guesses");
            asked.push_back(n);
            if (draft && draft->probs)
                std::fill_n(draft->probs, n, 0.95f);
            for (int i = 0; i < n; ++i)
                out[i] = i < right ? token + 1 + i : 0;
            return n;
        };
    }
};

Decoding conversation(int session, int32_t next, int64_t pos, int64_t max_new) {
    Decoding d;
    d.session = session;
    d.next = next;
    d.pos = pos;
    d.max_new = max_new;
    return d;
}
}  // namespace

int main() {
    {
        std::vector<StepGroup> groups(1);
        std::string err;
        check(!run_stage_groups({}, groups, err), "missing stages are rejected without polling");
    }
    for (int failure : {-1, 0, 1, 2}) {
        StageModel model;
        model.stages = 2;
        std::vector<StepGroup> groups(2);
        for (auto& group : groups) {
            group.rows.push_back({});
            group.picks.resize(1);
        }
        int prepared = 0, busy[2] = {-1, -1}, polls[2] = {};
        bool overlap = false;
        model.prepare = [&](int, StepGroup&, std::string&) {
            ++prepared;
            return true;
        };
        model.launch = [&](int stage, int group, std::string& err) {
            check(prepared == 2, "all captures and page reservations precede launches");
            check(busy[stage] < 0, "each stage owns at most one group");
            if (stage == 0 && group == 1 && failure == 0) {
                err = "launch failed";
                return false;
            }
            busy[stage] = group;
            polls[stage] = 0;
            overlap |= busy[0] >= 0 && busy[1] >= 0;
            return true;
        };
        model.poll = [&](int stage, int group, StepGroup&, std::string& err) {
            check(busy[stage] == group, "polls refer to the stage's owner");
            if (++polls[stage] < 3)
                return 0;
            busy[stage] = -1;
            if (failure == 1 && stage == 1 && group == 0) {
                err = "poll failed";
                return -1;
            }
            return 1;
        };
        model.finish = [&](int group, StepGroup& g, std::string& err) {
            check(busy[0] < 0 && busy[1] < 0, "sampling waits for every stage to drain");
            if (failure == 2) {
                err = "sampling failed";
                return false;
            }
            g.picks[0] = group + 10;
            return true;
        };
        std::string err;
        check(run_stage_groups(model, groups, err) == (failure < 0), "pipeline propagates launch and poll failures");
        check(busy[0] < 0 && busy[1] < 0, "every stage drains on success and failure");
        if (failure < 0) {
            check(overlap, "different groups run on both stages at once");
            check(groups[0].picks[0] == 10 && groups[1].picks[0] == 11, "each group keeps its own picks");
        }
    }
    {
        FakeModel fm;
        auto hooks = fm.hooks();
        fm.add_drafter(hooks);
        fm.right = 2;
        hooks.run_groups = [&](std::vector<StepGroup>& groups, std::string&) {
            check(groups.size() == 2 && groups[0].rows.size() == 4 && groups[1].rows.size() == 4,
                  "each group shares its own row budget with drafts");
            for (auto& g : groups)
                for (size_t r = 0; r < g.rows.size(); ++r)
                    g.picks[r] = g.rows[r].token + 1;
            return true;
        };
        int commits = 0;
        hooks.keep_group = [&](int, const std::vector<StepRow>&, const std::vector<int>& kept, std::string&) {
            check(kept == std::vector<int>{3}, "each group commits its accepted draft prefix");
            ++commits;
            return true;
        };
        Step step(hooks, {});
        step.settings = {2, 8, 3};
        step.settings.groups = 2;
        auto a = conversation(0, 10, 0, 32), b = conversation(1, 30, 0, 32);
        std::string err;
        check(step.run({&a, &b}, err) && commits == 2, "grouped drafts run through shared acceptance");
        check(a.emitted == std::vector<int32_t>{11, 12, 13} && b.emitted == std::vector<int32_t>{31, 32, 33},
              "grouped conversations emit their standalone accepted tokens");
        step.settings.groups = 1;
        check(step.run({&a, &b}, err), "live scheduling returns to one group");
    }
    {
        const std::vector<int32_t> a{1, 2}, b{7};
        const std::vector<StepRow> rows{{0, 3, 2, -1}, {0, 4, 3, 0}, {1, 8, 1, -1}, {1, 9, 2, 2}};
        const std::vector<const std::vector<int32_t>*> tails{&a, &a, &b, &b};
        const auto histories = strata::serve::get_penalty_histories(rows, tails, {3, 4});
        check(histories == std::vector<std::vector<int32_t>>{{1, 2, 3}, {2, 3, 4}, {-1, -1, 7, 8},
                                                              {-1, 7, 8, 9}},
              "each conversation and draft row has its own penalty history");
    }
    {
        FakeModel fm;
        auto hooks = fm.hooks();
        fm.add_drafter(hooks);
        fm.right = 7;
        hooks.run_groups = [](std::vector<StepGroup>& groups, std::string&) {
            for (auto& group : groups)
                for (size_t r = 0; r < group.rows.size(); ++r)
                    group.picks[r] = group.rows[r].token + 1;
            return true;
        };
        hooks.keep_group = [](int, const std::vector<StepRow>&, const std::vector<int>&, std::string&) {
            return true;
        };
        Step step(hooks, {});
        step.settings = {4, 7, 7};
        step.settings.groups = 2;
        auto a = conversation(0, 10, 0, 32), b = conversation(1, 20, 0, 32);
        auto c = conversation(2, 30, 0, 32), d = conversation(3, 40, 0, 32);
        std::string err;
        check(step.run({&a, &b, &c, &d}, err) && step.run({&a, &b, &c, &d}, err), "uneven grouped shares run");
        check(a.guessed == 5 && b.guessed == 5 && c.guessed == 5 && d.guessed == 5,
              "equal-priority conversations rotate the extra row within every group");
    }
    {
        FakeModel fm;
        fm.right = 3;
        auto hooks = fm.hooks();
        fm.add_drafter(hooks);
        hooks.run_groups = [](std::vector<StepGroup>& groups, std::string&) {
            for (auto& group : groups) {
                check(group.rows.size() == 1, "single-conversation drafter stays paused across groups");
                group.picks = {group.rows[0].token + 1};
            }
            return true;
        };
        hooks.keep_group = [](int, const std::vector<StepRow>&, const std::vector<int>&, std::string&) {
            return true;
        };
        Step step(hooks, {});
        step.settings = {2, 8, 3};
        step.settings.groups = 2;
        step.settings.max_draft_conversations = 1;
        auto a = conversation(0, 10, 0, 32), b = conversation(1, 20, 0, 32);
        std::string err;
        check(step.run({&a, &b}, err) && fm.asked == std::vector<int>{0, 0},
              "model draft limits count conversations across the whole wave");
        fm.asked.clear();
        auto c = conversation(2, 30, 0, 32);
        check(step.run({&c}, err) && fm.asked == std::vector<int>{3} && c.emitted.size() == 4,
              "single-conversation drafting resumes after a grouped wave");
    }
    {   // the plan's table: 3 conversations at once, 8 rows
        StepSettings s{3, 8, 8};
        check(share_guess_rows(s, 3, 3, 0) == std::vector<int>{2, 2, 1}, "3 active: 2, 2, 1");
        check(share_guess_rows(s, 2, 2, 0) == std::vector<int>{3, 3}, "2 active: 3, 3");
        check(share_guess_rows(s, 1, 1, 0) == std::vector<int>{7}, "1 active: 7");
        s.lookahead = 4;
        check(share_guess_rows(s, 1, 1, 0) == std::vector<int>{4}, "1 active: the lookahead if lower");
        s.lookahead = 8;
        check(share_guess_rows(s, 3, 3, 1) == std::vector<int>{1, 2, 2}, "the extra rows rotate");
        check(share_guess_rows(s, 3, 3, 2) == std::vector<int>{2, 1, 2}, "the extra rows wrap");
        s.lookahead = 0;
        check(share_guess_rows(s, 3, 3, 0) == std::vector<int>{0, 0, 0}, "lookahead 0: no guesses");
        check(share_guess_rows(StepSettings{3, 3, 4}, 3, 3, 0) == std::vector<int>{0, 0, 0}, "no rows left");
    }
    {   // priorities: the extra rows go to the highest priority, and rotate only among equals
        const StepSettings s{4, 10, 8};   // 6 rows left for 4: 1 each and 2 extra
        check(share_guess_rows(s, 4, 1, 3) == std::vector<int>{2, 2, 1, 1}, "one first: it and the next");
        check(share_guess_rows(s, 4, 2, 1) == std::vector<int>{2, 2, 1, 1}, "two equals first: both, any turn");
        check(share_guess_rows(s, 4, 3, 1) == std::vector<int>{1, 2, 2, 1}, "three equals first: two by turn");
        check(share_guess_rows(s, 4, 3, 2) == std::vector<int>{2, 1, 2, 1}, "three equals first: turns wrap");
    }
    {   // no drafting: one row per conversation, each emits one token per step
        FakeModel fm;
        Step step(fm.hooks(), {500});
        step.settings = {3, 8, 0};
        Decoding a = conversation(0, 10, 100, 2), b = conversation(1, 498, 7, 50), c = conversation(2, 30, 0, 50);
        c.stop = true;
        std::string err;
        check(step.run({&a, &b, &c}, err), "a step runs");
        check(fm.active == 2, "preparation excludes stopped conversations");
        check(fm.last.size() == 2 && fm.last[0].session == 0 && fm.last[1].session == 1, "one row each, none for c");
        check(fm.last[0].parent == -1 && fm.last[1].pos == 7, "rows start at the conversations' positions");
        check(fm.kept == std::vector<int>{1, 1}, "each keeps its row");
        check(a.emitted == std::vector<int32_t>{11} && a.fed == std::vector<int32_t>{10} && a.pos == 101 &&
                  a.next == 11 && a.finish == nullptr, "a emits the pick after its token");
        check(c.finish != nullptr && std::string(c.finish) == "cancel" && c.emitted.empty(), "a stopped one ends");
        check(fm.room[0] == 101, "room for the row");
        check(step.run({&a, &b}, err), "a second step runs");
        check(a.finish != nullptr && std::string(a.finish) == "length" && a.produced == 2, "max_new ends a");
        check(b.finish != nullptr && std::string(b.finish) == "stop" && b.emitted == std::vector<int32_t>{500},
              "an end token ends b");
    }
    {   // guesses: a chain after the conversation's token, kept up to the first wrong one
        FakeModel fm;
        StepModel m = fm.hooks();
        fm.add_drafter(m);
        fm.right = 2;
        Step step(m, {});
        step.settings = {1, 8, 3};
        Decoding a = conversation(0, 10, 100, 50);
        std::string err;
        check(step.run({&a}, err), "a step with guesses runs");
        check(fm.last.size() == 4 && fm.last[1].parent == 0 && fm.last[3].parent == 2 && fm.last[3].pos == 103,
              "a chain of 3 guesses");
        check(fm.room[0] == 104, "room for the guesses");
        check(fm.kept == std::vector<int>{3}, "2 right guesses kept");
        check(a.guessed == 3 && a.guessed_right == 2, "the guesses are counted");
        check(a.emitted == std::vector<int32_t>({11, 12, 13}) && a.fed == std::vector<int32_t>({10, 11, 12}) &&
                  a.pos == 103 && a.next == 13, "3 tokens from one step");
        Decoding b = conversation(0, 10, 100, 2);
        check(step.run({&b}, err) && fm.last.size() == 2, "no guess past max_new");
        Decoding one = conversation(0, 10, 100, 1);
        const size_t asked = fm.asked.size();
        check(step.run({&one}, err) && fm.last.size() == 1 &&
                  std::vector<int>(fm.asked.begin() + (std::ptrdiff_t) asked, fm.asked.end()) ==
                      std::vector<int>({0, 0}),
              "the drafter is asked for no guess at the last token, and again as the conversation ends");
        Decoding stopped = conversation(0, 10, 100, 50);
        stopped.stop = true;
        check(step.run({&stopped}, err) && fm.asked.size() == asked + 3 && fm.asked.back() == 0,
              "a stopped conversation's drafter is asked for no guess");
        Step ends(m, {12});
        ends.settings = step.settings;
        Decoding c = conversation(0, 10, 100, 50);
        check(ends.run({&c}, err) && fm.kept == std::vector<int>{2} && c.emitted == std::vector<int32_t>({11, 12}) &&
                  std::string(c.finish) == "stop", "an end token among the guesses ends it there");
    }
    {   // non-finite logits and a full memory end the conversation they concern
        FakeModel fm;
        fm.poisoned = 1;
        Step step(fm.hooks(), {});
        Decoding a = conversation(0, 10, 100, 50), b = conversation(1, 20, 5, 50);
        std::string err;
        check(step.run({&a, &b}, err) && std::string(b.finish) == "nonfinite" && b.emitted.empty() &&
                  fm.kept == std::vector<int>{1, 1} && a.finish == nullptr, "non-finite logits end b only");
        fm.room_limit = 50;
        fm.last.clear();
        Decoding c = conversation(0, 10, 100, 50);
        check(step.run({&c}, err) && std::string(c.finish) == "length" && fm.last.empty(), "no room: no row");
    }
    {
        FakeModel fm;
        Step step(fm.hooks(), {});
        step.settings = {2, 8, 3};
        Decoding a = conversation(0, 12, 8, 20), b = conversation(1, 32, 8, 20);
        const std::vector<int32_t> ah{10, 11, 12, 13, 14, 15, 10, 11, 12};
        const std::vector<int32_t> bh{30, 31, 32, 33, 34, 35, 30, 31, 32};
        a.produced = b.produced = 1;
        a.planner = std::make_unique<WindowPlanner>(8, 8, 3, 0, 3, 64);
        b.planner = std::make_unique<WindowPlanner>(8, 8, 3, 0, 3, 64);
        a.planner->sfx.append(ah.data(), ah.size());
        b.planner->sfx.append(bh.data(), bh.size());
        std::string err;
        check(step.run({&a, &b}, err) && a.emitted == std::vector<int32_t>({13, 14, 15}) &&
                  b.emitted == std::vector<int32_t>({33, 34, 35}), "independent lookup windows");
        check(a.suffix_offered == 2 && a.suffix_kept == 2 && a.planner->sfx.history().back() == 15,
              "lookup counts and history include only accepted tokens");
        a = conversation(0, 12, 8, 20);
        a.produced = 1;
        a.planner = std::make_unique<WindowPlanner>(8, 8, 3, 0, 3, 64);
        a.planner->sfx.append(ah.data(), ah.size());
        step.settings.max_draft_conversations = 1;
        check(step.run({&a, &b}, err) && a.emitted.size() == 1 && a.suffix_offered == 0,
              "multiple conversations suppress lookup where configured");
        Decoding first = conversation(0, 12, 8, 20);
        first.planner = std::make_unique<WindowPlanner>(8, 8, 3, 0, 3, 64);
        first.planner->sfx.append(ah.data(), ah.size());
        check(step.run({&first}, err) && first.emitted.size() == 1 && first.suffix_offered == 0,
              "the first row closes the prompt before lookup begins");
        Decoding mismatch = conversation(0, 12, 8, 20);
        mismatch.produced = 1;
        mismatch.planner = std::make_unique<WindowPlanner>(8, 8, 3, 0, 3, 64);
        const std::vector<int32_t> wrong{10, 11, 12, 99, 10, 11, 12};
        mismatch.planner->sfx.append(wrong.data(), wrong.size());
        check(step.run({&mismatch}, err) && mismatch.emitted == std::vector<int32_t>{13} &&
                  mismatch.suffix_kept == 0 && mismatch.planner->sfx.history().back() == 13,
              "rejected lookup guesses never enter history");
    }
    {
        FakeModel fm;
        fm.right = 3;
        StepModel hooks = fm.hooks();
        fm.add_drafter(hooks);
        Step step(hooks, {});
        step.settings = {1, 8, 3};
        Decoding d = conversation(0, 12, 8, 20);
        d.produced = 1;
        d.planner = std::make_unique<WindowPlanner>(8, 8, 3, 0, 3, 64);
        const std::vector<int32_t> history{10, 11, 12, 13, 14, 15, 10, 11, 12};
        d.planner->sfx.append(history.data(), history.size());
        std::string err;
        check(step.run({&d}, err) && fm.asked == std::vector<int>{3} && d.guessed_right == 3,
              "lookup and model proposals share the planner");
        check(step.run({&d}, err) && fm.asked == std::vector<int>({3, 3}) &&
                  d.emitted == std::vector<int32_t>({17, 18, 19, 20}),
              "model drafting resumes when lookup has no continuation");
    }
    {
        FakeModel fm;
        fm.right = 1;
        StepModel hooks = fm.hooks();
        fm.add_drafter(hooks);
        Step step(hooks, {});
        step.settings = {1, 8, 5};
        step.settings.model_lookahead = 1;
        Decoding d = conversation(0, 12, 12, 20);
        d.produced = 1;
        d.planner = std::make_unique<WindowPlanner>(8, 8, 0, 3, 3, 64);
        const std::vector<int32_t> history{10, 11, 12, 13, 14, 15, 16, 17, 10, 11, 12};
        d.planner->sfx.append(history.data(), history.size());
        const double costs[]{10, 11, 12, 13, 14, 15, 16, 17};
        d.planner->policy.seed_costs(costs, 8, 4);
        std::string err;
        check(step.run({&d}, err) && fm.asked.front() == 1 && d.emitted.size() > 2 &&
                  d.planner->chained.drafts > 0 && d.suffix_kept == d.suffix_offered,
              "lookup chain extends the model proposal within its row share");
    }
    {
        FakeModel fm;
        Step step(fm.hooks(), {});
        step.settings = {1, 8, 5};
        Decoding d = conversation(0, 12, 8, 20);
        d.produced = 1;
        d.planner = std::make_unique<WindowPlanner>(8, 8, 3, 0, 3, 64);
        const std::vector<int32_t> history{10, 11, 12, 13, 14, 15, 10, 11, 12};
        d.planner->sfx.append(history.data(), history.size());
        const double costly[]{1, 10, 20, 30, 40, 50, 60, 70};
        d.planner->policy.seed_costs(costly, 8, 4);
        std::string err;
        check(step.run({&d}, err) && d.emitted == std::vector<int32_t>{13} && d.suffix_offered == 0,
              "the shared cost rule rejects an unprofitable lookup without adding rows");
    }
    {
        const float confidence[]{0.99f, 0.9f, 0.8f, 0.6f};
        check(count_confident_drafts(confidence, 4, 0.65, 0.0f) == 3 &&
                  count_confident_drafts(confidence, 4, 0.0, 0.75f) == 2,
              "both model families share individual and cumulative confidence cuts");
        FakeModel fm;
        StepModel hooks = fm.hooks();
        fm.add_drafter(hooks);
        Step step(hooks, {});
        step.settings = {1, 8, 3};
        Decoding d = conversation(0, 12, 8, 20);
        d.draft_limits.confidence = 0.99f;
        d.draft_limits.cool_after_rejection = true;
        std::string err;
        check(step.run({&d}, err) && d.emitted.size() == 1 && d.draft_cooling,
              "a rejected model proposal schedules the shared cooldown");
        check(step.run({&d}, err) && !d.draft_cooling && fm.asked.back() == 0 &&
                  step.run({&d}, err) && fm.asked.back() == 3,
              "drafting resumes after one plain cooldown step");
    }
    {
        DraftContext context;
        check(!context.get_stopping_rule(), "zero floors preserve the native full-chain fast path");
        context.limits.cumulative = 0.75f;
        const auto stop = context.get_stopping_rule();
        check(stop && stop(0, 0.99f) && stop(1, 0.9f) && !stop(2, 0.8f),
              "incremental and batched networks use the same cumulative cut");
    }
    {
        const auto plain = get_window_rows(4, 0, 0, 0);
        const auto suffix = get_window_rows(4, 0, 3, 0);
        const auto chain = get_window_rows(4, 0, 3, 1);
        const auto capped = get_window_rows(4, 3, 3, 0);
        check(plain.total == 4 && plain.model == 0 && suffix.total == 6 && suffix.model == 4 &&
                  chain.total == 7 && chain.model == 4 && capped.total == 4 && capped.model == 3,
              "lookup capacity and explicit model caps agree across the original and paged engines");
    }
    for (int count : {1, 2, 3}) {
        for (int rows : {3, 4, 5, 8}) {
            FakeModel individual, batched;
            individual.right = 2;
            auto ordinary = individual.hooks();
            individual.add_drafter(ordinary);
            auto shared = batched.hooks();
            int calls = 0;
            shared.propose = [&](int, int32_t, int64_t, int n, int32_t*, DraftContext*) {
                check(n == 0, "batched networks bypass individual nonempty proposals");
                return 0;
            };
            shared.propose_batch = [&](std::vector<DraftRequest>& requests) {
                ++calls;
                check((int) requests.size() == count, "one batch contains all active conversations");
                for (auto& request : requests) {
                    request.count = request.limit;
                    for (int i = 0; i < request.limit; ++i) {
                        request.tokens[(size_t) i] = i < 2 ? request.token + i + 1 : 0;
                        request.confidence[(size_t) i] = 0.95f;
                    }
                }
            };
            Step before(ordinary, {}), after(shared, {});
            before.settings = after.settings = {count, rows, 5};
            std::vector<Decoding> a, b;
            for (int i = 0; i < count; ++i) {
                a.push_back(conversation(i, 100 + i * 20, 10 + i, 100));
                b.push_back(conversation(i, 100 + i * 20, 10 + i, 100));
            }
            a[0].draft_limits = b[0].draft_limits = {0.99f, 0, 0, true};
            std::vector<Decoding*> pa, pb;
            for (int i = 0; i < count; ++i) {
                pa.push_back(&a[(size_t) i]);
                pb.push_back(&b[(size_t) i]);
            }
            std::string err;
            for (int turn = 0; turn < 3; ++turn) {
                check(before.run(pa, err) && after.run(pb, err), "both proposal paths run");
                check(calls == turn + 1, "the drafter batch runs once per step");
                check(individual.kept == batched.kept, "batching preserves accepted prefixes");
                for (int i = 0; i < count; ++i) {
                    const auto& x = a[(size_t) i];
                    const auto& y = b[(size_t) i];
                    check(x.emitted == y.emitted && x.pos == y.pos && x.next == y.next &&
                              x.guessed == y.guessed && x.guessed_right == y.guessed_right &&
                              x.draft_cooling == y.draft_cooling,
                          "row sharing, confidence cuts and cooldown agree with individual proposals");
                }
            }
        }
    }
    std::printf(g_fail ? "step_test: %d FAILED\n" : "step_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}

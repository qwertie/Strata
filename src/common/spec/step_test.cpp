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
        m.propose = [this](int, int32_t token, int64_t, int n, int32_t* out) {
            check(active > 0 || n == 0, "preparation precedes requests for guesses");
            asked.push_back(n);
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
        const std::vector<int32_t> a{1, 2}, b{7};
        const std::vector<StepRow> rows{{0, 3, 2, -1}, {0, 4, 3, 0}, {1, 8, 1, -1}, {1, 9, 2, 2}};
        const std::vector<const std::vector<int32_t>*> tails{&a, &a, &b, &b};
        const auto histories = strata::serve::get_penalty_histories(rows, tails, {3, 4});
        check(histories == std::vector<std::vector<int32_t>>{{1, 2, 3}, {2, 3, 4}, {-1, -1, 7, 8},
                                                              {-1, 7, 8, 9}},
              "each conversation and draft row has its own penalty history");
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
    std::printf(g_fail ? "step_test: %d FAILED\n" : "step_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}

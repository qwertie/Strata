// src/common/spec/step.cpp - the decode step for several conversations at once (step.hpp).
#include "strata/common/spec/step.hpp"

#include <algorithm>
#include <chrono>

namespace strata::spec {

std::vector<int> share_guess_rows(const StepSettings& s, int active, int ties, int turn) {
    if (active <= 0)
        return {};
    ties = std::clamp(ties, 1, active);
    const int left = std::max(s.rows - active, 0);
    std::vector<int> share((size_t) active);
    for (int i = 0; i < active; ++i) {
        const int rank = i < ties ? ((i - turn) % ties + ties) % ties : i;   // in the order the extra rows go
        share[(size_t) i] = std::min(left / active + (rank < left % active ? 1 : 0), s.lookahead);
    }
    return share;
}

bool Step::run(const std::vector<Decoding*>& active, std::string& err) {
    std::vector<Decoding*> going;
    for (Decoding* d : active) {
        d->emitted.clear();
        d->fed.clear();
        d->finish = d->stop ? "cancel" : nullptr;
        if (!d->stop)
            going.push_back(d);
        else
            end_drafting(*d);
    }
    if (going.empty())
        return true;
    if (model_.begin)
        model_.begin((int) going.size());
    const int ties = (int) (std::find_if(going.begin(), going.end(),
                                         [&](const Decoding* d) { return d->priority != going[0]->priority; }) -
                            going.begin());
    const int turn = turn_ % ties;
    turn_ = turn + 1;
    const std::vector<int> share = share_guess_rows(settings, (int) going.size(), ties, turn);
    std::vector<StepRow> rows;
    std::vector<Window> windows;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Decoding*> stepping;
    std::vector<size_t> start;   ///< each stepping conversation's first row
    std::vector<int32_t> guesses((size_t) std::max(settings.rows, 1));
    const bool draft = settings.max_draft_conversations == 0 ||
                       (int) going.size() <= settings.max_draft_conversations;
    for (size_t i = 0; i < going.size(); ++i) {
        Decoding& d = *going[i];
        // no guess past what it may still emit, which also keeps its rows inside the context
        const int want = !draft || !d.draft_enabled ? 0 :
            (int) std::max<int64_t>(std::min<int64_t>(share[i], d.max_new - d.produced - 1), 0);
        std::vector<float> probs((size_t) DraftPolicy::kMaxT);
        DraftContext context{d.planner.get(), probs.data(), d.draft_limits};
        const bool cooling = model_.propose && want > 0 && d.draft_cooling;
        if (cooling)
            d.draft_cooling = false;
        const int model_want = cooling ? 0 : settings.model_lookahead < 0 ? want :
            std::min(want, settings.model_lookahead);
        int n = model_.propose ?
            model_.propose(d.session, d.next, d.pos, model_want, guesses.data(), &context) : 0;
        const int offered = n;
        n = count_confident_drafts(probs.data(), n, d.draft_limits.confidence, d.draft_limits.cumulative);
        if (offered > 0 && n == 0 && d.draft_limits.cool_after_rejection)
            d.draft_cooling = true;
        Window window{n + 1};
        if (d.planner) {
            auto& planner = *d.planner;
            if (planner.cost_aware && !context.cost_planned)
                for (int j = 0; j < n; ++j)
                    if (!context.keep_draft(j, probs[(size_t) j]))
                        break;
            window = planner.begin(d.pos, n + 1, n > 0, probs.data(), 0.0, d.produced == 0 || want == 0);
            window.T = std::min(window.T, n + 1);
            planner.pick_lookup(window, guesses.data(), want + 1);
            planner.pick_chain(window, guesses.data(), want + 1);
            if (window.lookup)
                std::copy_n(planner.lookup_drafts(), window.T - 1, guesses.data());
            if (window.chain > 0)
                std::copy_n(planner.chain_drafts(), window.chain, guesses.data() + window.T - 1);
            n = window.rows() - 1;
        }
        if (!model_.make_room(d.session, d.pos + 1 + n)) {
            d.finish = "length";
            continue;
        }
        windows.push_back(window);
        stepping.push_back(&d);
        start.push_back(rows.size());
        rows.push_back({d.session, d.next, d.pos, -1});
        for (int j = 0; j < n; ++j)
            rows.push_back({d.session, guesses[(size_t) j], d.pos + 1 + j, (int) rows.size() - 1});
    }
    if (rows.empty())
        return true;
    std::vector<int32_t> picks(rows.size());
    if (!model_.run(rows, picks, err))
        return false;
    std::vector<int> kept;
    for (size_t c = 0; c < stepping.size(); ++c) {
        const size_t end = c + 1 < stepping.size() ? start[c + 1] : rows.size();
        Decoding& d = *stepping[c];
        kept.push_back(accept(d, rows.data() + start[c], picks.data() + start[c], (int) (end - start[c])));
    }
    if (!model_.keep(rows, kept, err))
        return false;
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    for (size_t c = 0; c < stepping.size(); ++c) {
        Decoding& d = *stepping[c];
        const size_t end = c + 1 < stepping.size() ? start[c + 1] : rows.size();
        if (!d.planner)
            continue;
        std::vector<int32_t> tokens;
        for (size_t r = start[c]; r < end; ++r)
            tokens.push_back(rows[r].token);
        const int accepted = d.planner->accept(windows[c], tokens.data(), picks.data() + start[c]);
        // This step's timing excludes later cache effects of speculative expert traffic.
        // Joint passes do not measure the cost of one conversation's window width.
        if (stepping.size() == 1)
            d.planner->observe(windows[c], accepted, ms, d.finish != nullptr);
        for (int32_t token : d.emitted)
            d.planner->append(token);
        d.suffix_offered = d.planner->lookup.drafts + d.planner->chained.drafts;
        d.suffix_kept = d.planner->lookup.ok + d.planner->chained.ok;
    }
    for (const Decoding* d : stepping)
        if (d->finish != nullptr)
            end_drafting(*d);
    return true;
}

int Step::accept(Decoding& d, const StepRow* rows, const int32_t* picks, int n) const {
    int k = 1;   // the rows whose tokens are the conversation's: its token, then the guesses the model picked too
    while (k < n && picks[k - 1] == rows[k].token)
        ++k;
    if (std::any_of(picks, picks + k, [](int32_t p) { return p < 0; })) {
        d.finish = "nonfinite";
        return 1;
    }
    int e = 0;
    while (e < k && d.finish == nullptr) {
        d.emitted.push_back(picks[e]);
        d.fed.push_back(rows[e].token);
        ++d.produced;
        if (is_end(picks[e]))
            d.finish = "stop";
        else if (d.produced >= d.max_new)
            d.finish = "length";
        ++e;
    }
    d.pos += e;
    d.next = picks[e - 1];
    d.guessed += n - 1;
    d.guessed_right += e - 1;
    return e;
}

void Step::end_drafting(const Decoding& d) const {
    if (model_.propose)
        (void) model_.propose(d.session, d.next, d.pos, 0, nullptr, nullptr);
}

bool Step::is_end(int32_t token) const { return std::find(end_.begin(), end_.end(), (int64_t) token) != end_.end(); }

}  // namespace strata::spec

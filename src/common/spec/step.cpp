// src/common/spec/step.cpp - the decode step for several conversations at once (step.hpp).
#include "strata/common/spec/step.hpp"

#include <algorithm>

namespace strata::spec {

std::vector<int> share_guess_rows(const StepSettings& s, int active, int first) {
    if (active <= 0)
        return {};
    const int left = std::max(s.rows - active, 0);
    std::vector<int> share((size_t) active);
    for (int i = 0; i < active; ++i) {
        const bool extra = ((i - first) % active + active) % active < left % active;
        share[(size_t) i] = std::min(left / active + (extra ? 1 : 0), s.lookahead);
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
    }
    if (going.empty())
        return true;
    const int first = first_ % (int) going.size();
    first_ = first + 1;
    const std::vector<int> share = share_guess_rows(settings, (int) going.size(), first);
    std::vector<StepRow> rows;
    std::vector<Decoding*> stepping;
    std::vector<size_t> start;   ///< each stepping conversation's first row
    std::vector<int32_t> guesses((size_t) std::max(settings.rows, 1));
    for (size_t i = 0; i < going.size(); ++i) {
        Decoding& d = *going[i];
        // no guess past what it may still emit, which also keeps its rows inside the context
        const int want = (int) std::min<int64_t>(share[i], d.max_new - d.produced - 1);
        const int n = want > 0 && model_.propose ? model_.propose(d.session, d.next, d.pos, want, guesses.data()) : 0;
        if (!model_.make_room(d.session, d.pos + 1 + n)) {
            d.finish = "length";
            continue;
        }
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
        kept.push_back(accept(*stepping[c], rows.data() + start[c], picks.data() + start[c], (int) (end - start[c])));
    }
    return model_.keep(rows, kept, err);
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
    return e;
}

bool Step::is_end(int32_t token) const { return std::find(end_.begin(), end_.end(), (int64_t) token) != end_.end(); }

}  // namespace strata::spec

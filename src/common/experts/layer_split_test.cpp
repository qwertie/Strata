// The layer split's planner (layer_split.cpp) on synthetic cards: no GPU, no model.  The cards are the NVIDIA test PC's
// (an RTX 3060 12 GB on a PCIe 3.0 x4 link and an RTX 3080 20 GB on x16) and a model shaped like Qwen3.8 IQ3_S (48
// layers of 512 experts of 1.5 to 2.7 MB).
#include "strata/common/experts/layer_split.hpp"

#include "strata/common/experts/split_rules.hpp"

#include <cstdio>
#include <numeric>

using namespace strata::core;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {

constexpr int64_t kLayers = 48, kExperts = 512;
constexpr uint64_t kGiB = 1ull << 30;

/// Gets one decode layer's predicted ms on a card of `sms` multiprocessors at `ghz` (the old engine's rule).
double get_layer_ms(int sms, double ghz) { return strata::program::split_rules::get_layer_ms(sms * ghz); }

/// Gets per-layer sizes shaped like Qwen3.8 IQ3_S: experts of 1,510,400 to 2,662,400 bytes, 56 MB of dense weights,
/// a GDN layer's 3 MiB of state on three of four layers and K/V on the fourth.
LayerSizes get_sizes() {
    LayerSizes s;
    for (int64_t l = 0; l < kLayers; ++l) {
        s.expert.push_back(1510400 + (uint64_t) (l % 7) * 192000);
        s.dense.push_back(56ull << 20);
        s.fixed_state.push_back(l % 4 == 3 ? 0 : 3ull << 20);
        s.token_bytes.push_back(l % 4 == 3 ? 1184 : 0);
    }
    return s;
}

/// Gets the profile: every expert, the low numbers of each layer first (a stand-in for routing counts).
std::vector<int32_t> get_ranked_layers() {
    std::vector<int32_t> ranked;
    for (int64_t e = 0; e < kExperts; ++e)
        for (int64_t l = 0; l < kLayers; ++l)
            ranked.push_back((int32_t) l);
    return ranked;
}

/// Gets the problem of a 3060 (12 GB, x4 link) then a 3080 (20 GB, x16) whose engine takes `mem_3060` and `mem_3080`.
SplitProblem get_3060_3080(uint64_t mem_3060, uint64_t mem_3080) {
    SplitProblem p = make_paged_split_problem(get_sizes(), {mem_3060, mem_3080}, 1, 8192);
    p.cards[0] = {get_layer_ms(28, 1.78), 1.0};
    p.cards[1] = {get_layer_ms(68, 1.71), 1.0};
    const std::vector<double> scale = strata::program::split_rules::link_miss_scale({3.2, 24.0});
    p.cards[0].miss_scale = scale[0];
    p.cards[1].miss_scale = scale[1];
    p.ranked_layers = get_ranked_layers();
    return p;
}

/// Checks that each card of `plan` holds no more than its room and runs the layers of its cut.
void check_fills(const SplitPlan& plan, int64_t n_layers) {
    CHECK(!plan.fills.empty());
    int64_t lb = 0, held = 0;
    for (size_t i = 0; i < plan.fills.size(); ++i) {
        const SplitFill& f = plan.fills[i];
        CHECK(f.lb == lb);
        CHECK(f.le == (i < plan.cuts.size() ? plan.cuts[i] : n_layers));
        CHECK(f.used <= f.room);
        lb = f.le;
        held += f.held;
    }
    CHECK(held == plan.held);
}

}  // namespace

int main() {
    const char* log = "layer_split_test: ";
    {   // two equal cards with room for everything: every cut holds the whole profile, so the balanced one wins
        SplitProblem p = make_paged_split_problem(get_sizes(), {80 * kGiB, 80 * kGiB}, 1, 8192);
        p.cards = {{1.0, 1.0}, {1.0, 1.0}};
        p.ranked_layers = get_ranked_layers();
        const SplitPlan plan = plan_layer_split(p, log);
        CHECK(plan.cuts == std::vector<int64_t>{24});
        CHECK(plan.held == kLayers * kExperts);
        CHECK(!plan.start_unchecked);
        check_fills(plan, kLayers);
    }
    {   // the 3060 + 3080 with everything they have: the faster, larger 3080 runs most layers, and neither card runs
        // fewer than the three-layer floor
        const SplitPlan plan = plan_layer_split(get_3060_3080(11 * kGiB, 19 * kGiB), log);
        CHECK(plan.cuts.size() == 1);
        CHECK(plan.cuts[0] >= strata::program::split_rules::kMinStageLayers && plan.cuts[0] < 24);
        check_fills(plan, kLayers);
        std::printf("3060 x4 (11 GiB) + 3080 x16 (19 GiB): K=%lld, %lld of %lld experts on the cards, %.1f ms\n",
                    (long long) plan.cuts[0], (long long) plan.held, (long long) (kLayers * kExperts), plan.ms);
        // the 3080's memory setting goes down (CHANGE_SETTINGS, step M9): the cut moves so that the 3080 runs
        // fewer layers, or stays
        const SplitPlan less = plan_layer_split(get_3060_3080(11 * kGiB, 9 * kGiB), log);
        CHECK(less.cuts.size() == 1 && less.cuts[0] >= plan.cuts[0]);
        check_fills(less, kLayers);
        std::printf("... with the 3080 at 9 GiB: K=%lld, %lld experts on the cards\n", (long long) less.cuts[0],
                    (long long) less.held);
        // and the 3060's down instead: it runs fewer layers
        const SplitPlan small = plan_layer_split(get_3060_3080(3 * kGiB, 19 * kGiB), log);
        CHECK(small.cuts.size() == 1 && small.cuts[0] <= plan.cuts[0]);
        check_fills(small, kLayers);
    }
    {   // a placement the engine cannot start is not chosen; with none startable, the fastest is (without the
        // three-layer retry, which looks only among startable placements), and it says so
        SplitProblem p = get_3060_3080(11 * kGiB, 19 * kGiB);
        const SplitPlan free_pick = plan_layer_split(p, log);
        p.can_start = [&](const std::vector<SplitFill>& f) { return f[1].lb != free_pick.cuts[0]; };
        const SplitPlan gated = plan_layer_split(p, log);
        CHECK(!gated.cuts.empty() && gated.cuts != free_pick.cuts && !gated.start_unchecked);
        p.can_start = [](const std::vector<SplitFill>&) { return false; };
        const SplitPlan none = plan_layer_split(p, log);
        CHECK(none.start_unchecked && none.cuts == std::vector<int64_t>{2});
    }
    {   // a pick with a tiny card is searched again with at least three layers per card (#1616): a card so fast that
        // one layer on the other is best
        SplitProblem p = make_paged_split_problem(get_sizes(), {80 * kGiB, 80 * kGiB}, 1, 8192);
        p.cards = {{5.0, 1.0}, {1.0, 1.0}};
        p.ranked_layers = get_ranked_layers();
        const SplitPlan plan = plan_layer_split(p, log);
        CHECK(plan.cuts.size() == 1 && plan.cuts[0] >= strata::program::split_rules::kMinStageLayers);
    }
    {   // four equal cards (the memoised search): the balanced placement
        SplitProblem p = make_paged_split_problem(get_sizes(), {40 * kGiB, 40 * kGiB, 40 * kGiB, 40 * kGiB}, 1, 8192);
        p.cards.assign(4, {1.0, 1.0});
        p.ranked_layers = get_ranked_layers();
        p.coverage_curve = true;
        const SplitPlan plan = plan_layer_split(p, log);
        CHECK((plan.cuts == std::vector<int64_t>{12, 24, 36}));
        check_fills(plan, kLayers);
    }
    {   // a sizes table: the dense weights and both kinds of state count
        const LayerSizes s = get_sizes();
        CHECK(s.get_range_bytes(0, 4, 0, 0) == 4 * (56ull << 20));
        CHECK(s.get_range_bytes(0, 4, 2, 10) == 4 * (56ull << 20) + 2 * (3 * (3ull << 20) + 10 * 1184));
    }
    {   // one card: nothing to plan
        SplitProblem p = make_paged_split_problem(get_sizes(), {80 * kGiB}, 1, 8192);
        CHECK(plan_layer_split(p, log).cuts.empty());
    }
    if (failed == 0) std::printf("layer_split_test: all passed\n");
    return failed == 0 ? 0 : 1;
}

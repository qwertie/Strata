// Shared confidence cuts for batched proposals and incremental model-drafter callbacks.
#pragma once

namespace strata::spec {

/// Tracks prefix confidence for both batched proposals and incremental drafter callbacks.
struct ConfidenceCut {
    float minimum = 0.0f, cumulative_minimum = 0.0f, cumulative = 1.0f;
    /// Returns whether the next candidate meets both confidence floors.
    bool keep(float p) {
        return p >= minimum && (cumulative *= p) >= cumulative_minimum;
    }
};

/// Gets how many of `n` drafts the --spec-min-p rule offers: those before the first whose probability under the
/// drafter (`probs`) is below `min_p` or their product is below `min_cumulative`.
inline int count_confident_drafts(const float* probs, int n, double min_p, float min_cumulative = 0.0f) {
    int k = 0;
    ConfidenceCut cut{(float) min_p, min_cumulative};
    while (k < n && cut.keep(probs[k]))
        ++k;
    return k;
}

}  // namespace strata::spec

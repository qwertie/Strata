#include "strata/common/kernels/expert_scratch.hpp"

namespace strata::kernels {

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

}  // namespace strata::kernels

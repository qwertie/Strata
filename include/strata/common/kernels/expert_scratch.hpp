#pragma once

#include <cstdint>

namespace strata::kernels {

/// Returns the arena size for grouped experts' projections, quantized intermediates and activation scales.
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff);

}  // namespace strata::kernels

// include/strata/common/experts/expert_profile.hpp - the expert profile file (`--expert-profile`,
// `--expert-profile-save`, tools/make_profile.py): which (layer, expert) pairs the GPU expert cache holds first.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// **THE STATIC RESIDENCY PLAN, READ BACK FROM THE FILE `tools/make_profile.py` WROTE.**
///
/// `profile.bin` is `STRP`, then `version, n_layers, n_expert, slots, n_ranked`, then `n_ranked` `(layer,
/// expert)` pairs ranked by descending routing frequency, then a `n_layers x n_expert` frequency table.
///
/// **WHY A PROFILE AND NOT THE COMPULSORY-MISS POLICY THIS ENGINE ALREADY HAD.**  Compulsory-miss fills with
/// the first `S` DISTINCT experts the run happens to route, which on this workload is whatever the prompt
/// touched first - it measured `h = 0.4864`.  A profile ranked by frequency over a whole trace is what the
/// plan's `0.6447` refers to, and the CPU's half of the drain is linear in `h`, so the gap is worth about
/// 4 ms of drain.  The point of reading it from a file is that a profile built on ONE prompt can be scored on
/// ANOTHER, which is the only way to know whether it transfers.
///
/// The frequency table is not returned: it is the profile's own working, and the engine measures `h` by
/// running rather than by re-deriving what the file claims.
bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

/// #477 (--expert-profile-save): what the adaptive tier learned, as a profile ranking EVERY (layer, expert) pair -
/// the experts resident in VRAM now first (where the swaps left the cache), then the rest; within each, by `heat`
/// (the routing the adaptive tier counted since the start, descending), then by `prior` (the profile the engine
/// started from: an expert this run never routed keeps its old place), then by index.  A start from it begins
/// where this one ended; one with more slots adds the hottest of the rest, one with fewer keeps the hottest.
/// `resident` and `heat`: n_layers x n_expert entries (`heat` may be empty: no counts, the prior decides).
std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior);

/// #477: writes `ranked` in tools/make_profile.py's format, byte for byte: `STRP`, `1, n_layers, n_expert,
/// n_ranked, n_ranked` (uint32), the pairs (uint16 layer, uint16 expert), then the n_layers x n_expert int32 table
/// of each pair's rank (-1: not ranked).  Atomically: a `<path>.tmp` beside it, renamed over `path` once complete,
/// so a reader (the next start) never sees half a file.  False with `err` when it could not be written.
bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);

}  // namespace strata::core

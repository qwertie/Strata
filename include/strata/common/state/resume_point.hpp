// include/strata/common/state/resume_point.hpp - how much of a new prompt a kept conversation can continue from: its
// whole state, or a position short of its end that it can go back to (a re-rendered history, earlier reasoning
// dropped, splits from the conversation before its end).  The paged engine (src/common/serve/paged_engine.cpp) and
// strata-dsv41's slots (src/models/deepseekv41/ds4_slotmeta.hpp) choose by it.  No GPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

/// Gets how many of `prompt`'s first tokens a conversation whose state holds `tokens` can continue from: all of them
/// when they start the prompt and leave it at least its last token to read, else the latest of `points` (positions
/// short of its end it can go back to) at or before the first token where the two differ; 0 when none.
template <class Token>
size_t get_resume_tokens(const std::vector<int32_t>& tokens, const std::vector<int64_t>& points,
                         const std::vector<Token>& prompt) {
    if (prompt.empty())
        return 0;
    const size_t limit = std::min(tokens.size(), prompt.size() - 1);
    size_t same = 0;
    while (same < limit && tokens[same] == prompt[same])
        ++same;
    if (same == tokens.size())
        return same;
    size_t back = 0;
    for (const int64_t p : points)
        if (p > 0 && (size_t) p <= same)
            back = std::max(back, (size_t) p);
    return back;
}

}  // namespace strata::core

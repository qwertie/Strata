#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

/// Finds where history ends before the next answer, excluding a temporary effort turn when present.
inline int64_t find_prompt_turn_boundary(const std::vector<int64_t>& ids, int64_t from, int64_t turn_token,
                                         int64_t tail_role_token) {
    if (turn_token < 0)
        return -1;
    int64_t turn = -1;
    for (int64_t i = (int64_t) ids.size() - 1; i > from; --i)
        if (ids[(std::size_t) i] == turn_token) {
            turn = i;
            break;
        }
    if (turn > 0 && tail_role_token >= 0)
        for (int64_t i = turn - 1; i > from; --i)
            if (ids[(std::size_t) i] == turn_token) {
                if (ids[(std::size_t) i + 1] == tail_role_token)
                    turn = i;
                break;
            }
    return turn;
}

/// Finds the end of a sufficiently long opening system turn for root-pinned checkpoint retention.
inline int64_t find_prompt_root_boundary(const std::vector<int64_t>& ids, int64_t turn, int64_t turn_token,
                                         int64_t minimum) {
    if (turn_token >= 0 && minimum > 0)
        for (int64_t i = 1; i < turn; ++i)
            if (ids[(std::size_t) i] == turn_token)
                return i >= minimum ? i : -1;
    return -1;
}

}  // namespace strata::core

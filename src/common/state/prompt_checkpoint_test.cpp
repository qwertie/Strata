#include "strata/common/state/prompt_checkpoint.hpp"

#include <cstdio>
#include <cstdlib>

using strata::core::find_prompt_root_boundary;
using strata::core::find_prompt_turn_boundary;

namespace {
void check(bool condition, const char* description);
}

int main() {
    constexpr int64_t turn = 10, system = 11, user = 12, assistant = 13, end = 14;
    const std::vector<int64_t> ordinary = {turn, system, 1, end, turn, user, 2, end, turn, assistant};
    const std::vector<int64_t> effort = {
        turn, system, 1, end, turn, user, 2, end, turn, system, 3, end, turn, assistant};
    check(find_prompt_turn_boundary(ordinary, 0, turn, -1) == 8, "ordinary assistant header stays outside history");
    check(find_prompt_turn_boundary(ordinary, 0, turn, system) == 8, "a user turn is retained in history");
    check(find_prompt_turn_boundary(effort, 0, turn, system) == 8,
          "temporary effort turn stays outside the reusable history");
    check(find_prompt_turn_boundary(effort, 0, turn, -1) == 12, "disabled effort handling keeps its system turn");
    check(find_prompt_turn_boundary(effort, 8, turn, system) == 12,
          "an existing resume point is not crossed backwards");
    check(find_prompt_turn_boundary(effort, 12, turn, system) == -1, "a reused assistant header adds no split");
    check(find_prompt_turn_boundary(effort, 0, -1, system) == -1, "disabled turn token adds no split");
    check(find_prompt_turn_boundary({}, 0, turn, system) == -1, "empty prompt adds no split");
    check(find_prompt_turn_boundary({turn, system}, 0, turn, system) == -1,
          "the first header cannot become a history checkpoint");
    check(find_prompt_root_boundary(ordinary, 8, turn, 4) == 4, "long opening system prompt becomes the root");
    check(find_prompt_root_boundary(ordinary, 8, turn, 5) == -1, "a short system prompt has no pinned root");
    check(find_prompt_root_boundary(ordinary, 8, turn, 0) == -1, "root checkpoints can be disabled");
    check(find_prompt_root_boundary(ordinary, 8, -1, 1) == -1, "disabled turn token adds no root");
    check(find_prompt_root_boundary(ordinary, 4, turn, 1) == -1,
          "a root is not added at the answer history boundary");
    std::puts("prompt_checkpoint_test: 14 checks passed");
}

namespace {
void check(bool condition, const char* description) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}
}  // namespace

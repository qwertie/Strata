#include "strata/common/serve/token_budget.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
void check(bool value, const char* description) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}
}  // namespace

int main() {
    using strata::serve::TokenBudget;
    using strata::serve::get_token_limits_error;
    check(get_token_limits_error(204800, 204800, 8192, 401000).empty(), "separate limits accept the user's settings");
    check(!get_token_limits_error(204800, 204800, 8192, 200000).empty(),
          "lowering the combined budget below the per-conversation limit is rejected");
    check(!get_token_limits_error(204801, 204800, 8192, 401000).empty(),
          "raising the per-conversation limit above the loaded capacity requires a reload");
    check(get_token_limits_error(200000, 204800, 8192, 200000).empty(), "joint edits may meet at the same limit");
    check(!get_token_limits_error(0, 204800, 8192, 401000).empty(), "a conversation needs a positive limit");
    check(!get_token_limits_error(204800, 204800, 401001, 401000).empty(), "the minimum cannot exceed the budget");
    TokenBudget budget(401000);
    {
        TokenBudget::Reservation first(budget), second(budget), waiting(budget);
        check(first.try_reserve(180000) && second.try_reserve(180000), "two conversations fit together");
        check(!waiting.try_reserve(50000), "the aggregate limit applies even when every conversation fits alone");
        check(budget.get_used() == 360000 && waiting.get_used() == 0, "a failed reservation changes nothing");
        check(first.try_reserve(200000), "decoding can grow inside the remaining shared allowance");
        check(!second.try_reserve(202000), "decode growth cannot overcommit the shared budget");
        first.shrink_to(190000);
        check(budget.get_used() == 370000, "unused speculative tokens return to the allowance");
        check(!budget.try_set_limit(300000), "a lower allowance cannot silently exceed its limit");
        second.shrink_to(0);
        check(budget.try_set_limit(300000) && waiting.try_reserve(100000),
              "pausing a conversation makes room for a waiting request");
        check(budget.get_used() == 290000 && budget.get_available() == 10000,
              "all reservations share one counter");
    }
    check(budget.get_used() == 0, "destroyed conversations release their claims");
    check(budget.try_set_limit(0), "an empty allowance can be disabled");
    TokenBudget::Reservation empty(budget);
    check(empty.try_reserve(0) && !empty.try_reserve(1), "a zero budget permits only empty state");
    TokenBudget unlimited;
    TokenBudget::Reservation full(unlimited), overflow(unlimited);
    check(full.try_reserve(UINT64_MAX) && !overflow.try_reserve(1), "the accounting cannot wrap at UINT64_MAX");
    std::puts("token_budget_test: aggregate admission, decode growth, shrink and lifetime checks passed");
}

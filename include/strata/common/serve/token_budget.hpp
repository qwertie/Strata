#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace strata::serve {

/// Gets why a per-conversation limit and a shared GPU allowance cannot be applied together.
inline std::string get_token_limits_error(int64_t context, int64_t capacity, int64_t minimum, int64_t maximum) {
    if (context < 1 || context > capacity)
        return "the per-conversation limit must be from 1 to the loaded capacity of " + std::to_string(capacity) +
               " tokens; increasing the loaded capacity requires a model reload";
    if (context > maximum)
        return "the per-conversation limit (" + std::to_string(context) +
               ") cannot exceed the combined GPU token budget (" + std::to_string(maximum) + ")";
    if (minimum < 0 || minimum > maximum)
        return "the conversation token minimum (" + std::to_string(minimum) + ") is above the maximum (" +
               std::to_string(maximum) + ")";
    return {};
}

/// Limits logical conversation tokens independently of model-specific padding and spare KV cells.
class TokenBudget {
public:
    /// Owns one paged conversation's share; releasing its GPU state returns that share.
    class Reservation {
    public:
        explicit Reservation(TokenBudget& budget) : budget_(budget) {}
        ~Reservation() { shrink_to(0); }
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;

        /// Claims enough room for a proposed prompt or decode window, leaving the budget unchanged on failure.
        bool try_reserve(uint64_t tokens) {
            if (tokens <= used_)
                return true;
            const uint64_t growth = tokens - used_;
            if (growth > budget_.get_available())
                return false;
            budget_.used_ += growth;
            used_ = tokens;
            return true;
        }
        /// Returns claims beyond the computed prefix, or every claim when the conversation leaves the GPU.
        void shrink_to(uint64_t tokens) {
            const uint64_t keep = std::min(tokens, used_);
            budget_.used_ -= used_ - keep;
            used_ = keep;
        }
        uint64_t get_used() const { return used_; }

    private:
        TokenBudget& budget_;
        uint64_t used_ = 0;
    };

    explicit TokenBudget(uint64_t limit = UINT64_MAX) : limit_(limit) {}
    TokenBudget(const TokenBudget&) = delete;
    TokenBudget& operator=(const TokenBudget&) = delete;
    /// Changes the allowance only when existing conversations fit.
    bool try_set_limit(uint64_t limit) {
        if (limit < used_)
            return false;
        limit_ = limit;
        return true;
    }
    uint64_t get_used() const { return used_; }
    uint64_t get_available() const { return limit_ - used_; }

private:
    uint64_t limit_, used_ = 0;
};

}  // namespace strata::serve

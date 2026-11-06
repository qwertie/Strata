#pragma once

#include "strata/common/state/state_bytes.hpp"

#include <utility>

namespace strata::qwen38 {

/// Retains a prompt boundary for the paged model, with heap storage allocated only when captured.
struct RewindCheckpoint {
    int64_t tokens = 0; ///< Consumed prefix length, excluding the next token to read.
    std::vector<uint8_t> recurrent; ///< Whole-model GDN rows in layer order, independent of the GPU cut.
    uint64_t used = 0; ///< Recency for root-pinned retention.
    bool turn = false; ///< A continuation may replace the answer and temporary header after this position.
    bool pinned = false; ///< Shared prefix retained until another pin is requested.
    std::vector<uint8_t> state; ///< Final two token IDs followed by PLE, indexer and rotary metadata.
};

/// Includes every retained recurrent snapshot in the paged engine's host-state payload for RAM and disk pauses.
inline void write_rewind_checkpoints(std::vector<uint8_t>& out, const std::vector<RewindCheckpoint>& checkpoints) {
    using namespace strata::core;
    size_t bytes = out.size() + sizeof(uint64_t);
    for (const auto& checkpoint : checkpoints)
        bytes += 5 * sizeof(uint64_t) + checkpoint.state.size() + checkpoint.recurrent.size();
    out.reserve(bytes);
    append_state_u64(out, checkpoints.size());
    for (const auto& checkpoint : checkpoints) {
        for (uint64_t value : {(uint64_t) checkpoint.tokens, checkpoint.used,
                               (uint64_t) (checkpoint.turn | (checkpoint.pinned << 1))})
            append_state_u64(out, value);
        append_state_vector(out, checkpoint.state);
        append_state_vector(out, checkpoint.recurrent);
    }
}

/// Rejects truncated or incompatible snapshots before replacing the caller's chain.
inline bool read_rewind_checkpoints(strata::core::StateReader& reader, uint64_t maximum, uint64_t tokens,
                                    uint64_t recurrent_bytes, std::vector<RewindCheckpoint>& out) {
    const uint64_t count = reader.take_u64();
    if (!reader.ok || count > maximum)
        return false;
    std::vector<RewindCheckpoint> checkpoints;
    for (uint64_t i = 0; i < count; ++i) {
        RewindCheckpoint checkpoint;
        const uint64_t position = reader.take_u64();
        checkpoint.tokens = (int64_t) position;
        checkpoint.used = reader.take_u64();
        const uint64_t flags = reader.take_u64();
        checkpoint.turn = (flags & 1) != 0;
        checkpoint.pinned = (flags & 2) != 0;
        reader.take_vector(checkpoint.state);
        reader.take_vector(checkpoint.recurrent);
        if (!reader.ok || flags > 3 || position == 0 || position > tokens ||
            (!checkpoints.empty() && checkpoints.back().tokens >= checkpoint.tokens) ||
            checkpoint.state.size() < 16 || checkpoint.recurrent.size() != recurrent_bytes)
            return false;
        checkpoints.push_back(std::move(checkpoint));
    }
    out = std::move(checkpoints);
    return true;
}

}  // namespace strata::qwen38

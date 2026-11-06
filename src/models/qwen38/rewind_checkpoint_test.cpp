#include "strata/models/qwen38/rewind_checkpoint.hpp"
#include "strata/common/serve/conversation_residency.hpp"

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
    using namespace strata::qwen38;
    using namespace strata::core;
    using strata::serve::get_default_resident_conversations;
    constexpr uint64_t gib = 1ull << 30;
    check(get_default_resident_conversations(1, 8, 8 * gib, 10 * gib) == 2 &&
              get_default_resident_conversations(1, 8, 10 * gib, 10 * gib) == 3 &&
              get_default_resident_conversations(1, 8, 0, 10 * gib) == 2 &&
              get_default_resident_conversations(3, 8, 8 * gib, 10 * gib) == 4 &&
              get_default_resident_conversations(1, 8, 64 * gib, UINT64_MAX) == 2,
          "model-specific residency defaults distinguish 8 GiB, 10 GiB, unknown VRAM and explicit concurrency");
    using strata::serve::is_conversation_stale;
    check(is_conversation_stale(false, false, 34, 100, 66) &&
              !is_conversation_stale(false, false, 35, 100, 66) &&
              !is_conversation_stale(true, false, 1, 100, 66) &&
              !is_conversation_stale(false, true, 1, 100, 66) &&
              !is_conversation_stale(false, false, 101, 100, 66),
          "idle age uses completed-token activity and protects decoding, prompt reads and backward clock changes");
    RewindCheckpoint empty;
    check(empty.recurrent.capacity() == 0, "unused checkpoints reserve no recurrent host payload");
    std::vector<RewindCheckpoint> chain;
    for (int i = 1; i <= 6; ++i) {
        RewindCheckpoint checkpoint;
        checkpoint.tokens = i * 100;
        checkpoint.used = i + 10;
        checkpoint.turn = i == 6;
        checkpoint.pinned = i == 1;
        checkpoint.state.assign(32, (uint8_t) i);
        checkpoint.recurrent.assign(1024, (uint8_t) (i + 1));
        chain.push_back(std::move(checkpoint));
    }
    std::vector<uint8_t> bytes;
    write_rewind_checkpoints(bytes, chain);
    std::vector<RewindCheckpoint> restored;
    StateReader reader{bytes};
    check(read_rewind_checkpoints(reader, 6, 600, 1024, restored) && reader.at == bytes.size(),
          "every retained checkpoint survives host serialization");
    for (size_t i = 0; i < chain.size(); ++i)
        check(restored[i].tokens == chain[i].tokens && restored[i].used == chain[i].used &&
                  restored[i].turn == chain[i].turn && restored[i].pinned == chain[i].pinned &&
                  restored[i].state == chain[i].state && restored[i].recurrent == chain[i].recurrent,
              "rewind restores recurrent bytes and retention metadata independently for each position");
    for (size_t end : {size_t{0}, size_t{7}, bytes.size() - 1}) {
        std::vector<uint8_t> truncated(bytes.begin(), bytes.begin() + end);
        StateReader invalid{truncated};
        check(!read_rewind_checkpoints(invalid, 6, 600, 1024, restored) && restored.size() == 6,
              "truncated snapshots cannot replace an existing chain");
    }
    StateReader wrong_size{bytes}, too_many{bytes}, too_long{bytes};
    check(!read_rewind_checkpoints(wrong_size, 6, 600, 512, restored) &&
              !read_rewind_checkpoints(too_many, 5, 600, 1024, restored) &&
              !read_rewind_checkpoints(too_long, 6, 599, 1024, restored),
          "geometry, checkpoint-count and context bounds apply before restoring");
    std::vector<std::vector<float>> nested{{1.0f, 2.0f}, {3.0f}};
    check(get_owned_vector_bytes(nested) == nested.capacity() * sizeof(nested[0]) +
              (nested[0].capacity() + nested[1].capacity()) * sizeof(float),
          "shared host accounting includes nested checkpoint buffers");
    std::puts("rewind_checkpoint_test: lazy RAM state, round trips, bounds and residency defaults passed");
}

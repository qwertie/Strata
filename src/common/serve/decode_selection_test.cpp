#include "strata/common/serve/decode_selection.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
/// Supplies the request fields PagedEngine's selector reads, plus a count of simulated decode steps.
struct Conversation {
    struct {
        bool stop = false;
        int64_t priority = 0;
    } dec;
    uint64_t used = 0;
    int steps = 0;
};

void check(bool value, const char* description) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}
}  // namespace

int main() {
    Conversation first{{false, 0}, 1}, second{{false, 0}, 2}, waiting{{false, 0}, 3};
    std::vector<Conversation*> ready{&waiting, &second, &first};
    for (int step = 0; step < 20; ++step) {
        auto selected = ready;
        strata::serve::select_decode_conversations(selected, 2);
        for (Conversation* conversation : selected)
            ++conversation->steps;
    }
    check(first.steps == 20 && second.steps == 20 && waiting.steps == 0,
          "equal-priority requests keep their places instead of rotating through the waiting request");

    ready = {&waiting, &second};
    strata::serve::select_decode_conversations(ready, 2);
    check(ready == std::vector<Conversation*>({&second, &waiting}),
          "a waiting request advances when an earlier request completes");

    waiting.dec.priority = 10;
    ready = {&first, &second, &waiting};
    strata::serve::select_decode_conversations(ready, 2);
    check(ready == std::vector<Conversation*>({&waiting, &first}),
          "a higher-priority arrival displaces the newest lower-priority request");

    second.dec.stop = true;
    ready = {&first, &waiting, &second};
    strata::serve::select_decode_conversations(ready, 1);
    check(ready == std::vector<Conversation*>({&second}),
          "a cancelled waiting request drains even while higher-priority requests remain active");

    second.dec.stop = false;
    ready = {&second, &waiting, &first};
    strata::serve::select_decode_conversations(ready, 1);
    check(ready == std::vector<Conversation*>({&waiting}), "serial decoding still selects the highest priority");
    ready = {&second, &first};
    strata::serve::select_decode_conversations(ready, 1);
    check(ready == std::vector<Conversation*>({&first}), "a reduced limit preserves the oldest equal-priority request");
    strata::serve::select_decode_conversations(ready, 0);
    check(ready.empty(), "a zero limit selects nothing");
    strata::serve::select_decode_conversations(ready, 2);
    check(ready.empty(), "an empty ready list stays empty");
    std::puts("decode_selection_test: stable priority selection, completion and cancellation passed");
}

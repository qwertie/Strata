// The conversation-state descriptor (state_layout.hpp): sizes at N tokens, packed host images, and the copies of a
// prefix between two places, run here on host memory with memcpy.
#include "strata/common/state/state_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

/// Copies like copy_state, with memcpy in host memory.
bool copy_on_host(const StateLayout& layout, const StateBuffers& from, const StateBuffers& to, int64_t tokens) {
    return for_each_region_copy(layout, from, to, tokens, [](void* d, size_t dp, const void* s, size_t sp, size_t w,
                                                             size_t h) {
        for (size_t row = 0; row < h; ++row)
            std::memcpy((uint8_t*) d + row * dp, (const uint8_t*) s + row * sp, w);
        return true;
    });
}
}  // namespace

int main() {
    // a recurrent state of 40 bytes, K/V of 2 heads x 8 bytes a token, index keys of 12 bytes per block of 4 tokens
    StateLayout layout;
    check(layout.add(RegionKind::fixed, 40) == 0, "add returns the region's index");
    layout.add(RegionKind::per_token, 8, 2);
    layout.add(RegionKind::per_block, 12, 1, 4);

    check(layout.regions[0].bytes_for(0) == 40 && layout.regions[0].bytes_for(1000) == 40, "a fixed region");
    check(layout.regions[1].bytes_for(0) == 0 && layout.regions[1].bytes_for(5) == 80, "a per-token region");
    check(layout.regions[2].bytes_for(3) == 0 && layout.regions[2].bytes_for(9) == 24, "completed blocks only");
    check(layout.bytes_for(10) == 40 + 160 + 24, "the layout's size");

    // a session of 16 cells holding a pattern, packed into a host image of 10 tokens and back into another session
    const int64_t cells = 16, tokens = 10;
    std::vector<uint8_t> a(layout.bytes_for(cells)), b(a.size(), 0), image(layout.bytes_for(tokens), 0);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = (uint8_t) (i * 7 + 1);
    const StateBuffers sa = get_host_image(layout, cells, a.data()), sb = get_host_image(layout, cells, b.data());
    const StateBuffers packed = get_host_image(layout, tokens, image.data());
    check(packed.at[1] == image.data() + 40 && packed.at[2] == image.data() + 200, "the packed image's regions");
    check(copy_on_host(layout, sa, packed, tokens) && copy_on_host(layout, packed, sb, tokens), "the copies run");

    check(std::memcmp(b.data(), a.data(), 40) == 0, "the fixed region came back whole");
    for (int head = 0; head < 2; ++head) {
        const size_t row = 40 + (size_t) head * cells * 8;
        check(std::memcmp(b.data() + row, a.data() + row, tokens * 8) == 0, "each head's first tokens came back");
        check(b[row + tokens * 8] == 0, "a head's cells past the prefix stay untouched");
    }
    const size_t keys = 40 + 2 * cells * 8;
    check(std::memcmp(b.data() + keys, a.data() + keys, 2 * 12) == 0 && b[keys + 2 * 12] == 0,
          "the completed blocks' keys came back, no more");
    check(std::memcmp(image.data() + 40 + tokens * 8, a.data() + 40 + cells * 8, tokens * 8) == 0,
          "the image packs the second head right after the first one's prefix");

    int calls = 0;
    for_each_region_copy(layout, sa, sb, 0, [&](void*, size_t, const void*, size_t, size_t, size_t) {
        ++calls;
        return false;
    });
    check(calls == 1, "a copy of no tokens moves the fixed regions only, and a refused copy stops it");

    std::printf("state_layout_test: %d checks passed\n", checks);
    return 0;
}

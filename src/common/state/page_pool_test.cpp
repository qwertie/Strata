// The page pool (page_pool.hpp) on host memory with a hand-moved clock: conversations growing across blocks, experts
// filling and giving back pages, the token limits, the hysteresis, growing and shrinking, and the report.
#include "strata/common/state/page_pool.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

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

constexpr uint64_t page = 64, chunk = 4 * page;

/// Gets the test's conversation: a 100-byte recurrent state (2 pages) and K/V of 2 heads x 8 bytes a token (4 tokens
/// a page), so a conversation of N tokens takes 2 + ceil(N / 4) pages.
StateLayout get_layout() {
    StateLayout layout;
    layout.add(RegionKind::fixed, 100);
    layout.add(RegionKind::per_token, 8, 2);
    return layout;
}

PagePoolConfig get_config(int64_t tokens_min = 0, int64_t tokens_max = INT64_MAX) {
    PagePoolConfig config;
    config.page_bytes = page;
    config.chunk_bytes = chunk;
    config.tokens_min = tokens_min;
    config.tokens_max = tokens_max;
    config.hysteresis = 100;
    return config;
}

/// A pool of 3 chunks (12 pages) on host memory, and what it called.
struct Fixture {
    uint64_t clock = 0;
    int chunks = 0, max_chunks = 100;
    std::vector<ExpertKey> evicted;
    PagePool pool;

    explicit Fixture(PagePoolConfig config = get_config()) : pool(get_layout(), config, get_hooks()) {
        check(pool.resize(3 * chunk).refusal.empty() && chunks == 3, "the pool takes 3 chunks");
    }
    PagePoolHooks get_hooks() {
        PagePoolHooks hooks;
        hooks.alloc = [this](size_t bytes) -> void* {
            if (chunks == max_chunks)
                return nullptr;
            ++chunks;
            return std::calloc(1, bytes);
        };
        hooks.free = [this](void* p) {
            --chunks;
            std::free(p);
        };
        hooks.now = [this] { return clock; };
        hooks.evicted = [this](ExpertKey key, PageId) { evicted.push_back(key); };
        hooks.copy = [this](PageId to, PageId from) {
            std::memcpy(pool.page_address(to), pool.page_address(from), page);
            return true;
        };
        return hooks;
    }
    /// Adds experts 0 to `count` - 1 of layer 0, expert n at clock time n.
    void add_experts(int count) {
        for (int n = 0; n < count; clock = ++n)
            pool.add_expert({0, n});
    }
};

bool is_key(ExpertKey key, int expert) {
    return key.layer == 0 && key.expert == expert;
}

void test_allocation() {
    Fixture f;
    check(f.pool.report().total.pages == 12 && f.pool.report().free.pages == 12, "a new pool's pages are free");
    check(f.pool.pages_for(0) == 2 && f.pool.pages_for(5) == 4, "pages_for: the fixed pages and the blocks");
    check(f.pool.grow_conversation(1, 5).empty(), "a conversation of 5 tokens");
    check(f.pool.conversation_pages(1, 0) == std::vector<PageId>{0, 1}, "the fixed region takes whole pages once");
    check(f.pool.conversation_pages(1, 1) == std::vector<PageId>{2, 3}, "5 tokens take 2 blocks");
    check(f.pool.grow_conversation(1, 8).empty() && f.pool.conversation_pages(1, 1).size() == 2,
          "tokens 6 to 8 fit the second block");
    check(f.pool.grow_conversation(1, 9).empty() && f.pool.conversation_pages(1, 1).back() == 4,
          "token 9 takes a third block, the lowest free page");
    check(f.pool.grow_conversation(1, 3).empty() && f.pool.report().tokens == 9, "fewer tokens keep the pages");
    check(f.pool.page_address(1) == f.pool.page_address(0) + page && f.pool.page_address(4) != nullptr,
          "pages of a chunk are adjacent");
    bool threw = false;
    try {
        PagePoolConfig tiny = get_config();
        tiny.page_bytes = 8;
        PagePool pool(get_layout(), tiny, f.get_hooks());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a page smaller than one token's K/V is refused");
}

void test_experts_trade_pages() {
    Fixture f;
    check(f.pool.grow_conversation(1, 4).empty(), "a conversation of 3 pages");
    f.add_experts(9);
    check(f.pool.report().free.pages == 0 && f.pool.use_expert({0, 0}) == 11 && f.evicted.empty(),
          "experts fill the free pages from the top");
    f.pool.use_expert({0, 1});
    check(f.pool.add_expert({0, 1}) == 10 && f.evicted.empty(), "adding a held expert counts a use");
    check(f.pool.add_expert({1, 0}) == 9 && f.evicted.size() == 1 && is_key(f.evicted[0], 2),
          "a new expert in a full pool takes the page of the one used longest ago");
    check(f.pool.grow_conversation(1, 5).empty() && f.evicted.size() == 2 && is_key(f.evicted[1], 3) &&
              f.pool.conversation_pages(1, 1).back() == 8,
          "a growing conversation takes the least-used expert's page");
    check(f.pool.use_expert({0, 3}) == no_page, "an evicted expert is gone");
    f.pool.remove_expert({0, 4});
    check(f.pool.report().free.pages == 1 && f.pool.add_expert({0, 4}) == 7 && f.evicted.size() == 2,
          "a removed expert's page is open to experts at once");
}

void test_minimum() {
    Fixture f(get_config(8));   // 8 tokens: 4 pages
    f.add_experts(8);
    check(f.pool.report().free.pages == 4 && f.evicted.empty(), "experts leave the minimum's pages free");
    check(f.pool.add_expert({1, 0}) != no_page && f.evicted.size() == 1, "past the minimum an expert replaces one");
    check(f.pool.grow_conversation(1, 8).empty() && f.evicted.size() == 1, "a conversation takes the kept pages");
    check(f.pool.grow_conversation(1, 9).empty() && f.evicted.size() == 2, "and experts' pages past the minimum");
}

void test_maximum() {
    Fixture f(get_config(0, 10));
    check(f.pool.grow_conversation(1, 8).empty(), "8 tokens fit a maximum of 10");
    check(!f.pool.grow_conversation(2, 4).empty() && f.pool.report().active_conversations == 1,
          "a conversation past the maximum is refused and not added");
    check(!f.pool.set_token_limits(0, 6).empty(), "a maximum below an active conversation is refused");
    check(!f.pool.set_token_limits(9, 4).empty(), "a minimum above the maximum is refused");
    check(!f.pool.set_token_limits(41, 100).empty(), "a minimum larger than the pool is refused");
    check(f.pool.set_token_limits(0, 100).empty() && f.pool.grow_conversation(2, 4).empty(), "a raised maximum");
    check(f.pool.grow_conversation(3, 40).find("the conversation needs") == 0, "a full pool refuses a conversation");
}

void test_hysteresis() {
    Fixture f;
    check(f.pool.grow_conversation(1, 40).empty() && f.pool.report().free.pages == 0, "a conversation fills the pool");
    f.pool.remove_conversation(1);
    f.clock = 50;
    check(f.pool.add_expert({0, 0}) == no_page, "experts wait for the hysteresis");
    check(f.pool.grow_conversation(2, 4).empty() && f.pool.conversation_pages(2, 0)[0] == 0,
          "conversations do not");
    f.pool.set_paused(2, true);
    check(f.pool.grow_conversation(2, 5) == "the conversation is paused", "a paused conversation does not grow");
    f.clock = 100;
    check(f.pool.add_expert({0, 0}) == 11, "after the hysteresis experts take the pages");
}

void test_resize() {
    Fixture f;
    check(f.pool.grow_conversation(1, 12).empty() && f.pool.grow_conversation(2, 4).empty(), "pages 0 to 7");
    f.add_experts(4);
    check(f.pool.use_expert({0, 3}) == 8, "experts on pages 11 to 8");
    const std::vector<PageId> kv = f.pool.conversation_pages(2, 1);
    std::memset(f.pool.page_address(kv[0]), 0x5a, page);
    f.pool.remove_conversation(1);
    check(f.pool.resize(chunk + 100).refusal.empty() && f.chunks == 1, "the pool shrinks to its first chunk");
    check(f.evicted.size() == 4, "the given-back chunks' experts are evicted");
    const std::vector<PageId> moved = f.pool.conversation_pages(2, 1);
    check(moved[0] < 4 && f.pool.page_address(moved[0])[page - 1] == 0x5a, "a moved page keeps its contents");
    check(f.pool.report().total.pages == 4 && f.pool.report().free.pages == 1, "the smaller pool's pages");
    check(f.pool.resize(3 * chunk).refusal.empty() && f.pool.report().free.pages == 9, "and grows again");
    f.max_chunks = 4;
    check(f.pool.resize(5 * chunk).refusal == "the memory had room for 4 of 5 chunks" && f.chunks == 4,
          "a growth the memory has no room for keeps the chunks it got");
}

void test_shrink_refusals() {
    Fixture f;
    check(f.pool.grow_conversation(1, 24).empty(), "a conversation of 8 pages");
    check(f.pool.resize(chunk).refusal.find("an active conversation needs") == 0 && f.chunks == 3,
          "a pool too small for the biggest active conversation is refused");
    check(f.pool.grow_conversation(2, 4).empty(), "a second conversation on pages 8 to 10");
    const PoolResize r = f.pool.resize(2 * chunk);
    check(!r.refusal.empty() && r.to_pause == std::vector<ConversationId>{2} && f.chunks == 3,
          "conversation pages with nowhere to go name the conversations to pause");
    f.pool.set_paused(2, true);
    check(f.pool.resize(2 * chunk).to_pause.size() == 1, "a paused conversation keeps its pages until removed");
    f.pool.remove_conversation(2);
    check(f.pool.resize(2 * chunk).refusal.empty() && f.chunks == 2, "once removed the pool shrinks");
}

void test_report() {
    Fixture f;
    f.pool.grow_conversation(1, 5);
    f.pool.grow_conversation(2, 4);
    f.pool.set_paused(2, true);
    f.pool.add_expert({0, 0});
    f.pool.add_expert({0, 1});
    const PagePoolReport r = f.pool.report();
    check(r.active.pages == 2 && r.fixed.pages == 2 && r.paused.pages == 3 && r.experts.pages == 2 &&
              r.free.pages == 3,
          "the pages by use");
    check(r.active.pages + r.fixed.pages + r.paused.pages + r.experts.pages + r.free.pages == r.total.pages,
          "the uses add up to the pool");
    check(r.total.bytes == 12 * page && r.paused.bytes == 3 * page, "bytes are pages times the page size");
    check(r.active_conversations == 1 && r.paused_conversations == 1 && r.tokens == 9, "the conversations");
}
}  // namespace

int main() {
    test_allocation();
    test_experts_trade_pages();
    test_minimum();
    test_maximum();
    test_hysteresis();
    test_resize();
    test_shrink_refusals();
    test_report();
    std::printf("page_pool_test: %d checks passed\n", checks);
    return 0;
}

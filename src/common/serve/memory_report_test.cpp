// --serve's GET_MEMORY_USE and CHANGE_SETTINGS (memory_report.hpp): the lines strata_common/engine_client.py
// reads and writes.
#include "strata/common/serve/memory_report.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::serve;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}
}  // namespace

int main() {
    MemoryReport r;
    Memory& gpu = r.gpu(0);
    gpu.name = "Radeon 8060S Graphics";
    gpu.total = 1000;
    gpu.free = 300;
    gpu.unified = true;
    gpu.add("weights", 100);
    gpu.add("experts", 400);
    gpu.add("drafter", 0);
    gpu.add("experts", 50);
    r.at("ram").paused_budget = 800;
    check(&r.at("gpu0") == &gpu && r.memories.size() == 2, "at() finds a memory by its id");
    r.settings = {{"context", 32768}, {"concurrent", 1}};
    const int32_t ids[] = {1, 2, 3};
    const std::vector<strata::core::HeldConversation> held = {
        {"main", "vram", "holding", ids, 3, 32768, 9999, 10},
        {"parked1", "ram", "parked", ids, 3, 3, 200, 200},
        {"parked2", "ram", "parked", ids, 2, 2, 100, 100},
        {"parked3", "disk", "parked", ids, 1, 1, 70, 70}};
    check(format_memory_report(r, held) ==
              "MEMORY gpu0 total=1000 free=300 unified=1 weights=100 experts=450 name=Radeon 8060S Graphics\n"
              "MEMORY ram paused=300 paused_count=2 paused_budget=800\n"
              "MEMORY ssd paused=70 paused_count=1\n"
              "SETTINGS context=32768 concurrent=1\n"
              "END_MEMORY\n",
          "the memories, the parked conversations by place, the settings");
    r.per_conversation = true;
    check(format_memory_report(r, {held[0]}) ==
              "MEMORY gpu0 total=1000 free=300 unified=1 weights=100 experts=450 name=Radeon 8060S Graphics\n"
              "MEMORY ram paused_count=0 paused_budget=800\n"
              "CONVERSATION main vram holding 3 32768 9999 10\n"
              "SETTINGS context=32768 concurrent=1\n"
              "END_MEMORY\n",
          "a line per conversation");
    check(format_memory_report(MemoryReport{}, {}) == "SETTINGS\nEND_MEMORY\n", "an engine that knows nothing");
    MemoryReport pool;
    Memory& card = pool.gpu(1);
    card.add("experts", 9);
    card.pool_pages = 1000;
    card.allocated_pages = 973;
    card.fragmented_pages = 2;
    card.name = "card";
    check(format_memory_report(pool, {}) ==
              "MEMORY gpu1 experts=9 pool_pages=1000 allocated_pages=973 fragmented_pages=2 name=card\n"
              "SETTINGS\nEND_MEMORY\n",
          "a page pool's pages, before the name");
    MemoryReport budget;
    budget.at("ram").paused_budget = 5;
    check(format_memory_report(budget, {}) == "MEMORY ram paused_count=0 paused_budget=5\nSETTINGS\nEND_MEMORY\n",
          "a budget with nothing parked");
    card.setting = 512;
    card.smallest = 256;
    card.largest = 2048;
    check(format_memory_report(pool, {}) ==
              "MEMORY gpu1 experts=9 pool_pages=1000 allocated_pages=973 fragmented_pages=2 setting=512 smallest=256 "
              "largest=2048 name=card\nSETTINGS\nEND_MEMORY\n",
          "a pool's setting and its range, before the name");
    budget.at("ram").mapped = 7;
    budget.shortfall = "the GPU's pool: out of memory";
    check(format_memory_report(budget, {}) == "MEMORY ram paused_count=0 paused_budget=5 mapped=7\nSETTINGS\n"
                                              "SHORTFALL the GPU's pool: out of memory\nEND_MEMORY\n",
          "the GPU-mapped bytes, and a change's shortfall before the end");
    MemoryReport labelled;
    labelled.gpu(0).add_blocks({{{"arbitrary category", "Pool / one", "Drafter & buffers"}, 123},
                                {{"arbitrary category", "Pool / one", "Drafter & buffers"}, 77}});
    labelled.gpu(0).setting = 200;
    labelled.gpu(0).pool_offset = 10;
    const std::string wire = format_memory_report(labelled, {});
    check(wire.find("pool_offset=10") != std::string::npos, "labelled pools declare their divider position");
    check(wire.find("MEMORY_BLOCK gpu0 200 arbitrary%20category Pool%20%2F%20one Drafter%20%26%20buffers") !=
              std::string::npos, "labels are aggregated and protocol-safe without a category list");
    strata::core::MemoryManager manager;
    int allocation = 0;
    manager.add(&allocation, 42, "ram", {"other", "Drafter", "Working buffers"}, true);
    check(manager.get_blocks("ram").size() == 1 && manager.get_mapped_bytes() == 42,
          "mapped RAM is counted once with its owning labels");
    manager.set_label(&allocation, {"", "Page pool", "Backing"});
    check(manager.get_blocks("ram").empty() && manager.get_mapped_bytes() == 42,
          "pool backing is excluded from totals but retains its mapping");
    manager.remove(&allocation);
    check(manager.get_mapped_bytes() == 0, "release removes the allocation's mapping and labels");
    manager.add(&allocation, 42, "ram", {"other", "Pool", "First lifetime"});
    const uint64_t first_id = manager.get_allocation_id(&allocation);
    manager.add(&allocation, 84, "ram", {"other", "Pool", "Reused address"});
    manager.remove_if_current(&allocation, first_id);
    check(manager.get_blocks("ram").at(0).bytes == 84,
          "a completed release cannot erase another thread's new allocation at the same address");
    manager.remove_if_current(&allocation, manager.get_allocation_id(&allocation));
    check(manager.get_blocks("ram").empty(), "a matching lifetime is removed");
    using Values = std::vector<std::pair<std::string, std::string>>;
    Values values;
    std::string err;
    check(parse_settings_line("CHANGE_SETTINGS memory_mib.gpu0=4096  drafter=off", values, err) &&
              values == Values{{"memory_mib.gpu0", "4096"}, {"drafter", "off"}},
          "CHANGE_SETTINGS' fields, in order");
    check(!parse_settings_line("CHANGE_SETTINGS rows", values, err) && !err.empty(), "a field without a value");
    check(!parse_settings_line("CHANGE_SETTINGS", values, err) && !err.empty(), "no field");
    std::printf("memory_report_test: %d checks passed\n", checks);
    return 0;
}

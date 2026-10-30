// --serve's MEM (memory_report.hpp): the lines strata_common/engine_client.py reads.
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
    check(format_memory_report(MemoryReport{}, {}) == "SETTINGS\nEND_MEMORY\n", "an engine that knows nothing");
    MemoryReport budget;
    budget.at("ram").paused_budget = 5;
    check(format_memory_report(budget, {}) == "MEMORY ram paused_count=0 paused_budget=5\nSETTINGS\nEND_MEMORY\n",
          "a budget with nothing parked");
    std::printf("memory_report_test: %d checks passed\n", checks);
    return 0;
}

// strata/common/serve/memory_report.hpp - --serve's MEM: what the engine holds in each memory (each GPU, the RAM, the
// SSD), for the server's GET /v1/memory and the web page's Memory panel (docs/MEMORY_MANAGER_PLAN.md, step M0).
//
//   MEM
//     MEMORY <id> [total=<B>] [free=<B>] [unified=1] [<use>=<B> ...] [paused_count=<n>] [paused_budget=<B>]
//            [name=<text>]                                     one line per memory, then
//     SETTINGS [context=<tokens>] [concurrent=<n>] [lookahead=<n>]
//     END_MEMORY
//
// id: `gpu<N>` (the runtime's device number), `ram` or `ssd`.  total, free: the memory's size and what is free in it
// now (the GPUs'; the server adds the machine's RAM).  unified=1: a GPU whose memory is the system RAM (an integrated
// GPU such as Strix Halo's).  <use>: the bytes the engine holds there for one use - `weights` (the dense weights),
// `experts`, `conversations` (the sessions of active conversations, each allocated for its whole context), `paused`
// (parked conversations), `drafter`, `scratch`, `engram` (DeepSeek's n-gram table).  paused_count: the parked
// conversations there; paused_budget: the most they may take.  name: the device's name, to the end of the line.
// settings: the context in tokens, the conversations decoded at once and the most guesses one adds in a step.  A
// value the engine does not know is left out, not guessed.  The command changes nothing.
#pragma once

#include "strata/common/state/token_listing.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace strata::serve {

/// One memory the engine uses: a GPU, the RAM or the SSD (a `MEMORY` line).
struct Memory {
    std::string id;                                       ///< gpu<N>, ram or ssd
    std::string name;                                     ///< the device's name; empty: not said
    uint64_t total = 0;                                   ///< its size in bytes; 0: unknown
    uint64_t free = 0;                                    ///< what is free in it now (said only with total)
    bool unified = false;                                 ///< a GPU whose memory is the system RAM
    std::vector<std::pair<std::string, uint64_t>> uses;   ///< bytes held per use, in the order first added
    uint64_t paused_budget = 0;                           ///< the most parked conversations may take; 0: no budget
    int paused_count = 0;                                 ///< parked conversations held here

    /// Adds `bytes` to use `use`; a zero adds nothing.
    void add(const std::string& use, uint64_t bytes) {
        if (bytes == 0) return;
        for (auto& [u, b] : uses)
            if (u == use) {
                b += bytes;
                return;
            }
        uses.emplace_back(use, bytes);
    }
};

/// What an engine says about its memories and settings (EngineHooks::describe_memory).
struct MemoryReport {
    std::deque<Memory> memories;   ///< (a deque: at() keeps the references it gave out valid)
    std::vector<std::pair<std::string, int64_t>> settings;   ///< SETTINGS' keys, in order

    /// Gets the memory named `id`, adding an empty one at the end when there is none.
    Memory& at(const std::string& id) {
        for (Memory& m : memories)
            if (m.id == id) return m;
        Memory& m = memories.emplace_back();
        m.id = id;
        return m;
    }
    /// Gets GPU `device`'s memory (at("gpu<device>")).
    Memory& gpu(int device) { return at("gpu" + std::to_string(device)); }
};

/// Writes the answer to MEM: `report` plus the parked conversations of `held` (LIST_CONVERSATIONS' entries; a parked
/// one in RAM counts on `ram`, one on disk on `ssd`).
inline std::string format_memory_report(MemoryReport report, const std::vector<core::HeldConversation>& held) {
    for (const core::HeldConversation& c : held) {
        if (std::string(c.state) != "parked") continue;
        Memory& m = report.at(std::string(c.place) == "disk" ? "ssd" : "ram");
        m.add("paused", c.bytes);
        ++m.paused_count;
    }
    std::string s;
    const auto put = [&s](const std::string& key, uint64_t v) { s += " " + key + "=" + std::to_string(v); };
    for (const Memory& m : report.memories) {
        s += "MEMORY " + m.id;
        if (m.total > 0) {
            put("total", m.total);
            put("free", m.free);
        }
        if (m.unified) put("unified", 1);
        for (const auto& [use, bytes] : m.uses)
            put(use, bytes);
        if (m.paused_count > 0 || m.paused_budget > 0) put("paused_count", (uint64_t) m.paused_count);
        if (m.paused_budget > 0) put("paused_budget", m.paused_budget);
        if (!m.name.empty()) s += " name=" + m.name;
        s += '\n';
    }
    s += "SETTINGS";
    for (const auto& [key, v] : report.settings)
        s += " " + key + "=" + std::to_string(v);
    s += "\nEND_MEMORY\n";
    return s;
}

}  // namespace strata::serve

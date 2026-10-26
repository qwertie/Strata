// --serve's LIST_SESSION_TOKENS and LIST_PARKED_CONVERSATIONS answers (serve/server.py's API monitor shows them):
// what a session in VRAM or a parked conversation holds, one stdout line each, the token ids last and comma-separated
// as GEN takes them.  Shared by the engine paths (generate.cpp, q36_engine.cu, ds4_main.cpp) so the lines are alike.
#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {

// `<head> <n> <id,id,...>` (no ids after the count when n is 0), one write
inline void print_token_line(const std::string& head, const int32_t* ids, size_t n) {
    std::string s;
    s.reserve(head.size() + 24 + n * 7);
    s += head;
    s += ' ';
    s += std::to_string(n);
    char num[16];
    for (size_t i = 0; i < n; ++i) {
        s += i == 0 ? ' ' : ',';
        const auto r = std::to_chars(num, num + sizeof num, ids[i]);
        s.append(num, r.ptr);
    }
    s += '\n';
    std::fwrite(s.data(), 1, s.size(), stdout);
}

// LIST_SESSION_TOKENS: `SESSION_TOKENS <name> <state> <n> <ids>`.  name: `main` (the engine's own session) or
// `slot<b>` (a --batch slot's); state: `holding` (idle, holding the tokens), `decoding`, `reading` (a prompt read
// that gave way and goes on later) or `empty`.  END_SESSION_TOKENS follows the last session.
inline void print_session_tokens(const std::string& name, const char* state, const int32_t* ids, size_t n) {
    print_token_line("SESSION_TOKENS " + name + " " + state, ids, n);
}

// LIST_PARKED_CONVERSATIONS: `PARKED_CONVERSATION <id> <ram|disk> <bytes> <n> <ids>`, the next to be evicted first.
// id tells the entries of one answer apart (it is not kept across answers: the DeepSeek path renumbers an entry it
// takes back).  END_PARKED_CONVERSATIONS follows the last one.
inline void print_parked_conversation(uint64_t id, bool on_disk, uint64_t bytes, const int32_t* ids, size_t n) {
    print_token_line("PARKED_CONVERSATION " + std::to_string(id) + (on_disk ? " disk " : " ram ") +
                     std::to_string(bytes), ids, n);
}

}  // namespace strata::core

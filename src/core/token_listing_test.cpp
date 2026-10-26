// The lines of --serve's LIST_SESSION_TOKENS and LIST_PARKED_CONVERSATIONS (token_listing.hpp), as serve/server.py
// reads them.
#include "strata/core/token_listing.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
}

int main() {
    const char* path = "token_listing_test.out";
    if (std::freopen(path, "wb", stdout) == nullptr) { std::fprintf(stderr, "FAIL: cannot write %s\n", path); return 1; }
    const int32_t ids[] = {248045, 0, 7, -1, 2147483647};
    print_session_tokens("main", "holding", ids, 5);
    print_session_tokens("slot1", "empty", nullptr, 0);
    print_parked_conversation(42, false, 1535867580, ids, 2);
    print_parked_conversation(43, true, 17, ids + 2, 1);
    std::fflush(stdout);
    std::fclose(stdout);
    std::ifstream in(path, std::ios::binary);
    std::stringstream got;
    got << in.rdbuf();
    in.close();
    std::remove(path);
    std::istringstream lines(got.str());
    std::string l;
    std::getline(lines, l);
    check(l == "SESSION_TOKENS main holding 5 248045,0,7,-1,2147483647", "a session: name, state, count, ids");
    std::getline(lines, l);
    check(l == "SESSION_TOKENS slot1 empty 0", "an empty session: no ids after the count");
    std::getline(lines, l);
    check(l == "PARKED_CONVERSATION 42 ram 1535867580 2 248045,0", "a parked conversation in RAM");
    std::getline(lines, l);
    check(l == "PARKED_CONVERSATION 43 disk 17 1 7", "one on disk");
    check(!std::getline(lines, l), "one line each, nothing else");
    std::fprintf(stderr, "token_listing_test: %d checks passed\n", checks);
}

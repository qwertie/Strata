// --serve's LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS (token_listing.hpp): the lines serve/server.py reads.
#include "strata/core/token_listing.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
// what answering `line` writes to stdout
std::string answer(const std::string& line, const std::vector<HeldConversation>& held) {
    const char* path = "token_listing_test.out";
    if (std::freopen(path, "wb", stdout) == nullptr) { std::fprintf(stderr, "FAIL: cannot write %s\n", path); std::exit(1); }
    answer_conversation_command(line, held);
    std::fclose(stdout);
    std::ifstream in(path, std::ios::binary);
    std::stringstream got;
    got << in.rdbuf();
    in.close();
    std::remove(path);
    return got.str();
}
}

int main() {
    // a system prompt, a user turn (248045 = <|im_start|>, 872 = user, 198 = \n), its text and the answer's start
    const int32_t chat[] = {248045, 9125, 198, 1, 2, 248046, 248045, 872, 198, 40, 41, 42, 43, 248046};
    const std::vector<HeldConversation> held = {
        {"main", "vram", "holding", chat, 14, 204800, 3092376453, 1048576},
        {"slot0", "vram", "empty", nullptr, 0, 204800, 3092376453, 0},
        {"parked7", "ram", "parked", chat, 5, 5, 1535867580, 1535867580},
    };
    check(is_conversation_command("LIST_CONVERSATIONS") && is_conversation_command("GET_CONTEXT_WINDOWS main") &&
          !is_conversation_command("GEN 1 2,3"), "the two commands, nothing else");
    check(answer("LIST_CONVERSATIONS", held) ==
              "CONVERSATION main vram holding 14 204800 3092376453 1048576\n"
              "CONVERSATION slot0 vram empty 0 204800 3092376453 0\n"
              "CONVERSATION parked7 ram parked 5 5 1535867580 1535867580\n"
              "END_CONVERSATIONS\n",
          "a list: id, place, state, tokens held and the most it can hold, bytes and the bytes used - no tokens");
    check(answer("LIST_CONVERSATIONS 248045,872,198 3", held) ==
              "CONVERSATION main vram holding 14 204800 3092376453 1048576 40,41,42\n"
              "CONVERSATION slot0 vram empty 0 204800 3092376453 0\n"
              "CONVERSATION parked7 ram parked 5 5 1535867580 1535867580\n"
              "END_CONVERSATIONS\n",
          "a preview: the tokens after the user-turn marker, none where the conversation does not hold it");
    check(answer("LIST_CONVERSATIONS 248045,872,198 99", held).rfind("CONVERSATION main vram holding 14 204800 3092376453 1048576 40,41,42,43,248046\n", 0) == 0,
          "a preview stops at the end of the conversation");
    check(answer("GET_CONTEXT_WINDOWS parked7 slot0 parked3", held) ==
              "CONTEXT_WINDOW parked7 5 248045,9125,198,1,2\n"
              "CONTEXT_WINDOW slot0 0\n"
              "CONTEXT_WINDOW parked3 gone\n"
              "END_CONTEXT_WINDOWS\n",
          "context windows in the order asked; an id the engine no longer holds is gone");
    check(answer("LIST_CONVERSATIONS 1,x 3", held).rfind("ERR ", 0) == 0 &&
          answer("LIST_CONVERSATIONS 248045", held).rfind("ERR ", 0) == 0,
          "a malformed preview is refused");
    std::fprintf(stderr, "token_listing_test: %d checks passed\n", checks);
}

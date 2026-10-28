// strata/common/serve/request.hpp - a request line of the server's protocol, split into its parts:
//
//     GEN    <max_new> [key=value ...] <id,id,...>
//     GENI   <max_new> [key=value ...] <file> <id,id,...>         (the prompt's pictures are in <file>)
//     BGEN   <slot> <max_new> [key=value ...] <id,id,...>         (--batch: continue the request in a batch slot)
//     BGENI  <slot> <max_new> [key=value ...] <file> <id,id,...>
//
// The keys every engine reads are the sampling ones (Sampling); the others belong to one engine, and an engine skips
// a key it does not know.  Absent keys keep greedy decoding without penalties.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::serve {

/// One request line's parts, as the line gives them: whether they make sense is the engine's to check.
struct RequestLine {
    bool batch = false;         ///< BGEN / BGENI
    bool pictures = false;      ///< GENI / BGENI
    long slot = 0;              ///< BGEN's batch slot
    long long max_new = 0;      ///< the most tokens to generate (0 where the line has no number)
    std::vector<std::pair<std::string, std::string>> keys;   ///< the key=value options, in the line's order
    std::string file;           ///< GENI's picture file; empty when no space and ids follow it
    std::string ids;            ///< the rest of the line: the prompt's token ids, comma-separated
};

/// Splits `line` into `out`; false when it is not a request line.
bool parse_request_line(const std::string& line, RequestLine& out);

/// The sampling options of a request: temperature=F top_p=F top_k=N min_p=F seed=N penalty_last_n=N
/// penalty_repeat=F penalty_freq=F penalty_present=F.
struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 20;                ///< Qwen's sampler default (its sampled path needs 1..64)
    unsigned long long seed = 0;   ///< 0: the engine picks one
    int penalty_last_n = 0;        ///< how many recent tokens the penalties look at
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
};

/// Stores `value` in `s` if `key` names a sampling option; false for any other key.
bool read_sampling_key(const std::string& key, const std::string& value, Sampling& s);

/// Reads comma-separated hexadecimal numbers (the img_keys option: one key per picture); empty if any is malformed.
std::vector<uint64_t> parse_hex_list(const std::string& text);

}  // namespace strata::serve

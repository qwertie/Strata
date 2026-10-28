// strata/common/serve/request.cpp - see request.hpp.
#include "strata/common/serve/request.hpp"

#include <cstdlib>
#include <cstring>

namespace strata::serve {

bool parse_request_line(const std::string& line, RequestLine& out) {
    out = RequestLine{};
    out.batch = line.rfind("BGEN", 0) == 0;
    const char* p = line.c_str() + (out.batch ? 1 : 0);
    if (std::strncmp(p, "GENI ", 5) == 0) out.pictures = true;
    else if (std::strncmp(p, "GEN ", 4) != 0) return false;
    p += out.pictures ? 5 : 4;
    char* e = nullptr;
    if (out.batch) {
        out.slot = std::strtol(p, &e, 10);
        p = e;
    }
    out.max_new = std::strtoll(p, &e, 10);
    p = e;
    for (;;) {   // the options, up to the first word without '='
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string word(start, (size_t) (p - start));
        const size_t eq = word.find('=');
        if (word.empty() || eq == std::string::npos) {
            p = start;
            break;
        }
        out.keys.emplace_back(word.substr(0, eq), word.substr(eq + 1));
    }
    if (out.pictures) {
        const char* gap = std::strchr(p, ' ');
        if (gap != nullptr) {
            out.file.assign(p, (size_t) (gap - p));
            p = gap;
        }
    }
    out.ids = p;
    return true;
}

bool read_sampling_key(const std::string& key, const std::string& value, Sampling& s) {
    const char* v = value.c_str();
    if (key == "temperature") s.temperature = std::strtof(v, nullptr);
    else if (key == "top_p") s.top_p = std::strtof(v, nullptr);
    else if (key == "top_k") s.top_k = std::atoi(v);
    else if (key == "min_p") s.min_p = std::strtof(v, nullptr);
    else if (key == "penalty_last_n") s.penalty_last_n = std::atoi(v);
    else if (key == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
    else if (key == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
    else if (key == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
    else if (key == "seed") s.seed = std::strtoull(v, nullptr, 10);
    else return false;
    return true;
}

std::vector<uint64_t> parse_hex_list(const std::string& text) {
    std::vector<uint64_t> out;
    const char* q = text.c_str();
    while (*q != '\0') {
        char* qe = nullptr;
        const unsigned long long v = std::strtoull(q, &qe, 16);
        if (qe == q || (*qe != ',' && *qe != '\0')) return {};
        out.push_back((uint64_t) v);
        q = *qe == ',' ? qe + 1 : qe;
    }
    return out;
}

}  // namespace strata::serve

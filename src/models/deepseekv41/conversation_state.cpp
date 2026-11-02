// src/models/deepseekv41/conversation_state.cpp - a DeepSeek conversation's host side and its bytes
// (conversation_state.hpp).
#include "strata/models/deepseekv41/conversation_state.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace strata::ds4 {
namespace {

constexpr uint64_t kStateMagic = 0x3154415453345344ull;   // "DS4STAT1" (little endian)
/// The least bytes a checkpoint's layer takes: the counts of its ring and its partial group's two parts.
constexpr uint64_t kLayerBytesMin = 3 * sizeof(uint64_t);

/// Appends `n` bytes at `p` to `out`.
void put(std::vector<uint8_t>& out, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    out.insert(out.end(), b, b + n);
}
void put_u64(std::vector<uint8_t>& out, uint64_t v) { put(out, &v, sizeof v); }
/// Appends a vector's element count, then its elements.
template <class T>
void put_vector(std::vector<uint8_t>& out, const std::vector<T>& v) {
    put_u64(out, v.size());
    put(out, v.data(), v.size() * sizeof(T));
}

/// Reads encode_state's bytes from the front; every read fails once one has run past the end.
struct Reader {
    const std::vector<uint8_t>& in;
    size_t at = 0;
    bool ok = true;

    bool take(void* p, size_t n) {
        ok = ok && n <= in.size() - at;
        if (ok) {
            std::memcpy(p, in.data() + at, n);
            at += n;
        }
        return ok;
    }
    uint64_t take_u64() {
        uint64_t v = 0;
        (void) take(&v, sizeof v);
        return v;
    }
    template <class T>
    void take_vector(std::vector<T>& v) {
        const uint64_t n = take_u64();
        ok = ok && n <= (in.size() - at) / sizeof(T);
        if (ok) {
            v.resize((size_t) n);
            (void) take(v.data(), (size_t) n * sizeof(T));
        }
    }
};

}  // namespace

std::vector<int64_t> get_check_positions(const ConversationState& s) {
    std::vector<int64_t> v;
    for (const Checkpoint& c : s.checks)
        if (c.pos > 0 && c.pos <= (int64_t) s.tokens.size() && !c.win.empty() &&
            std::find(v.begin(), v.end(), c.pos) == v.end())
            v.push_back(c.pos);
    std::sort(v.rbegin(), v.rend());
    return v;
}

void encode_state(const ConversationState& s, std::vector<uint8_t>& out) {
    out.clear();
    put_u64(out, kStateMagic);
    put_vector(out, s.tokens);
    put_u64(out, (uint64_t) std::count_if(s.checks.begin(), s.checks.end(), [](const Checkpoint& c) {
        return c.pos >= 0;
    }));
    for (size_t role = 0; role < s.checks.size(); ++role) {
        const Checkpoint& c = s.checks[role];
        if (c.pos < 0)
            continue;
        put_u64(out, role);
        put(out, &c.pos, sizeof c.pos);
        put_u64(out, c.win.size());
        for (size_t l = 0; l < c.win.size(); ++l) {
            put_vector(out, c.win[l]);
            for (size_t k = 2 * l; k < 2 * l + 2; ++k)
                put_vector(out, k < c.comp.size() ? c.comp[k] : std::vector<float>{});
        }
    }
    put_u64(out, kStateMagic);
}

bool decode_state(const std::vector<uint8_t>& in, ConversationState& s, std::string& err) {
    Reader r{in};
    ConversationState got;
    bool ok = r.take_u64() == kStateMagic;
    r.take_vector(got.tokens);
    const uint64_t n = r.take_u64();
    ok = ok && r.ok && n <= (uint64_t) kCheckpoints;
    for (uint64_t i = 0; ok && i < n; ++i) {
        const uint64_t role = r.take_u64();
        ok = r.ok && role < (uint64_t) kCheckpoints && got.checks[(size_t) role].pos < 0;
        if (!ok)
            break;
        Checkpoint& c = got.checks[(size_t) role];
        const uint64_t layers = r.take(&c.pos, sizeof c.pos) ? r.take_u64() : 0;
        ok = r.ok && c.pos >= 0 && c.pos <= (int64_t) got.tokens.size() &&
             layers <= (in.size() - r.at) / kLayerBytesMin;
        c.win.resize(ok ? (size_t) layers : 0);
        c.comp.resize(c.win.size() * 2);
        for (size_t l = 0; ok && l < c.win.size(); ++l) {
            r.take_vector(c.win[l]);
            r.take_vector(c.comp[2 * l]);
            r.take_vector(c.comp[2 * l + 1]);
            ok = r.ok;
        }
    }
    if (!ok || r.take_u64() != kStateMagic || !r.ok || r.at != in.size()) {
        err = "the paused conversation's checkpoints are cut short or damaged";
        return false;
    }
    s = std::move(got);
    return true;
}

}  // namespace strata::ds4

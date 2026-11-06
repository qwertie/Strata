#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace strata::core {

/// Appends one native little-endian field used by the engines' versioned host-state formats.
inline void append_state_bytes(std::vector<uint8_t>& out, const void* data, size_t bytes) {
    if (bytes) {
        const auto* first = static_cast<const uint8_t*>(data);
        out.insert(out.end(), first, first + bytes);
    }
}
inline void append_state_u64(std::vector<uint8_t>& out, uint64_t value) {
    append_state_bytes(out, &value, sizeof value);
}
/// Appends a vector's element count followed by its contents, preserving existing checkpoint file layouts.
template<class T>
void append_state_vector(std::vector<uint8_t>& out, const std::vector<T>& values) {
    append_state_u64(out, values.size());
    append_state_bytes(out, values.data(), values.size() * sizeof(T));
}

/// Reads bounded fields from checkpoint payloads. A failed read leaves all subsequent reads failed.
struct StateReader {
    const std::vector<uint8_t>& in;
    size_t at = 0;
    bool ok = true;

    bool take(void* out, size_t bytes) {
        ok = ok && bytes <= in.size() - at;
        if (ok && bytes) {
            std::memcpy(out, in.data() + at, bytes);
            at += bytes;
        }
        return ok;
    }
    uint64_t take_u64() {
        uint64_t value = 0;
        (void) take(&value, sizeof value);
        return value;
    }
    template<class T>
    bool take_vector(std::vector<T>& out) {
        const uint64_t count = take_u64();
        ok = ok && count <= (in.size() - at) / sizeof(T);
        if (ok) {
            out.resize((size_t) count);
            (void) take(out.data(), out.size() * sizeof(T));
        }
        return ok;
    }
};

/// Gets heap storage owned by nested host-state vectors, including retained capacity.
template<class T>
uint64_t get_owned_vector_bytes(const T&) { return 0; }
template<class T>
uint64_t get_owned_vector_bytes(const std::vector<T>& values) {
    uint64_t bytes = values.capacity() * sizeof(T);
    if constexpr (!std::is_trivially_copyable_v<T>)
        for (const auto& value : values)
            bytes += get_owned_vector_bytes(value);
    return bytes;
}

}  // namespace strata::core

// include/strata/models/deepseekv41/kv_rows.hpp - the formats of DeepSeek V4.1's cache rows, their sizes and where
// they lie: the kernels (ds4_kernels.cuh) write and read them, and host code sizes a conversation's state with them
// (conversation_layout.hpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_DS4_HD __host__ __device__
#else
#define STRATA_DS4_HD
#endif

namespace strata::ds4 {

/// The caches' rows: DeepSeek's own roundings kept as codes and scales - lossless, the values are on those grids
/// already (the numbers are Finish::quant's; a finish with that quant and a pdst writes the format), a row of d values
/// in kv_row_bytes(fmt, d) bytes, padded to 16 (the hand-offs copy 16-byte units):
///   kKvFp8   the window ring: d E4M3 codes, then d / 32 power-of-two scales (UE8M0: the F32 exponent byte)
///   kKvFp4   the compressed KV: d / 2 bytes of E2M1 codes (value 2i in the low nibble), then d / 16 E4M3 scales
///   kKvFp4p  the index keys: d / 2 bytes of E2M1 codes, then d / 32 UE8M0 scales
/// kKvBf16: plain BF16 rows (a DSpark block's own rows, and the reference the tests check the packed rows against).
constexpr int kKvBf16 = 0, kKvFp8 = 1, kKvFp4 = 2, kKvFp4p = 3;
STRATA_DS4_HD constexpr size_t kv_row_bytes(int fmt, int d) {
    return fmt == kKvBf16  ? (size_t) d * 2
           : fmt == kKvFp8 ? ((size_t) d + (size_t) d / 32 + 15) / 16 * 16
           : fmt == kKvFp4 ? ((size_t) d / 2 + (size_t) d / 16 + 15) / 16 * 16
                           : ((size_t) d / 2 + (size_t) d / 32 + 15) / 16 * 16;
}

/// Where one KV source's rows of one format lie on a device: back to back from `base` (the old engine's caches, which
/// hold the whole context), or in the page pool's blocks of `per_block` rows each, anywhere (a conversation's
/// per_block regions, docs/MEMORY_MANAGER_PLAN.md step D2).  A row never crosses a block.  Kernels take it by value.
struct KvRows {
    uint8_t* base = nullptr;                 ///< row 0 when `blocks` is null
    uint8_t* const* blocks = nullptr;        ///< device memory: each block's first row; null: the rows are at `base`
    uint8_t* const* host_blocks = nullptr;   ///< `blocks` copied to host memory, for the addresses host code computes
    uint32_t per_block = 1;                  ///< rows in each of the `blocks`
    uint32_t bytes = 0;                      ///< size of one row (kv_row_bytes)

    /// Tells whether the device keeps these rows.
    STRATA_DS4_HD explicit operator bool() const { return base != nullptr || blocks != nullptr; }

    /// Gets row r's device address.  A kernel reads the block's address from `blocks`, host code from `host_blocks`.
    STRATA_DS4_HD uint8_t* row(int64_t r) const {
        if (blocks == nullptr)
            return base + (size_t) r * bytes;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        uint8_t* const* table = blocks;
#else
        uint8_t* const* table = host_blocks;
#endif
        return table[(uint32_t) r / per_block] + (size_t) ((uint32_t) r % per_block) * bytes;
    }

    /// Calls f(offset, n_bytes, at) for each part of rows [r0, r0 + n) that lies in one block (all of them in one
    /// call when the rows are at `base`), `at` its device address and `offset` the bytes of the rows before it, until
    /// f returns false.  Returns whether every call returned true.
    template <class F>
    bool for_each_run(int64_t r0, int64_t n, F&& f) const {
        for (int64_t done = 0; done < n;) {
            const int64_t k = blocks == nullptr ? n - done
                                                : std::min<int64_t>(n - done, per_block - (r0 + done) % per_block);
            if (!f((size_t) done * bytes, (size_t) k * bytes, row(r0 + done)))
                return false;
            done += k;
        }
        return true;
    }
};

/// Describes rows of format `fmt` with `d` values each, lying back to back from `base`.
inline KvRows make_kv_rows(uint8_t* base, int fmt, int d) {
    KvRows r;
    r.base = base;
    r.bytes = (uint32_t) kv_row_bytes(fmt, d);
    return r;
}

}  // namespace strata::ds4

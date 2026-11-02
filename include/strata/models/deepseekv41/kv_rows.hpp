// include/strata/models/deepseekv41/kv_rows.hpp - the formats of DeepSeek V4.1's cache rows and their sizes: the
// kernels (ds4_kernels.cuh) write and read them, and host code sizes a conversation's state with them
// (conversation_layout.hpp).
#pragma once

#include <cstddef>

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

}  // namespace strata::ds4

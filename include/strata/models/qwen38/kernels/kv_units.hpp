// include/strata/models/qwen38/kernels/kv_units.hpp - a QSA layer's K/V and indexer keys in the page pool's pages
// (docs/MEMORY_MANAGER_PLAN.md M2b).
//
// Today's (packed) layout gives each array its own buffer: K codes `[page][kv_head][page_size][head_dim]`, V the
// same, their scales, and the pooled indexer keys `[block][idx_dim]`, every array indexed by one row number.  The
// page pool (page_pool.hpp) instead keeps one conversation's QSA layer as a growing region whose UNIT is one block of
// `idx_block` cells (= one KV page): all of that block's arrays together, one after the other, as
// `describe_conversation` sizes it.  One of the pool's blocks (a run of its pages) holds many units, and a
// conversation's blocks are anywhere.
//
// The kernels address both layouts with the same arithmetic.  A cell's row is `(page * n_kv_heads + kv_head) *
// page_size + cell % page_size` as always; its address in array X is `X + base + row * row_bytes(X)`, where:
//   - packed: base is 0 and `page` comes from the page table;
//   - units:  base is `units[block]`, `page` is 0, and X is the array's offset within a unit plus kKvUnitOrigin.
// So the packed layout computes exactly what it did, and a unit's arrays need no stride of their own.  A kernel keeps
// base + row in one integer (`kv_place`), so its shared memory does not grow.  The origin keeps every array pointer
// non-null (the kernels tell the format from which arrays are present) and makes an access that forgot the base
// fault instead of reading another conversation's state.
//
// The unit is "one block ahead": the indexer writes the spare key (`dead`) into the block after the last completed
// one, so a conversation's units must reach block `n_kv / idx_block + 1` (grow it to `pos + idx_block + 1` tokens).
// KV streaming (kv_stream.hpp) does not combine with units: the pool replaces it.
#pragma once

#include "strata/models/qwen38/kernels/kv_q4.hpp"
#include "strata/models/qwen38/kernels/kv_q8.hpp"
#include "strata/models/qwen38/kernels/kv_stream.hpp"
#include "strata/models/qwen38/kernels/qsa.hpp"
#include "strata/models/qwen38/kernels/qsa_decode_attn.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::kernels {

/// The address every array pointer of the unit layout is relative to: low enough to be unmapped on every GPU.
inline constexpr uint64_t kKvUnitOrigin = 4096;

/// Byte offsets of one block's arrays within its unit (K, V, then the scales, then the pooled indexer key), and the
/// unit's size.  An array the format lacks has offset 0 and takes no bytes.
struct KvUnitLayout {
    uint64_t k = 0, v = 0, k_scale = 0, v_scale = 0, key = 0;
    uint64_t bytes = 0;
};

/// Gets the unit of format `f`: `idx_block` cells of `kv_*_bytes_per_cell` and one `idx_dim` fp32 key, every array
/// 16-byte aligned.  Throws std::invalid_argument when a KV page is not one indexer block (`page_size != idx_block`)
/// or holds more than 16 rows (kv_place).
inline KvUnitLayout kv_unit_layout(const QsaShapes& s, KvFormat f) {
    if (s.page_size != s.idx_block || s.n_head_kv * s.page_size > 16)
        throw std::invalid_argument("kv_unit_layout: a unit must be one indexer block of at most 16 rows");
    const uint64_t rows = (uint64_t) s.n_head_kv * s.page_size;
    const uint64_t f16 = (uint64_t) s.head_dim * 2, q8 = (uint64_t) s.head_dim;
    const uint64_t q4 = kv_q4_bytes_per_head((int) s.head_dim);
    const uint64_t scale = (uint64_t) s.head_dim / KV_Q8_GROUP * 2;
    const uint64_t k_row = f == kKvF16 ? f16 : f == kKvQ4 ? q4 : q8;
    const uint64_t v_row = f == kKvF16 ? f16 : f == kKvInt8 ? q8 : q4;
    KvUnitLayout u;
    uint64_t at = 0;
    const auto place = [&](uint64_t& offset, uint64_t bytes) {
        offset = at;
        at += (bytes + 15) / 16 * 16;
    };
    place(u.k, rows * k_row);
    place(u.v, rows * v_row);
    if (f == kKvInt8 || f == kKvHybrid)
        place(u.k_scale, rows * scale);
    if (f == kKvInt8)
        place(u.v_scale, rows * scale);
    place(u.key, (uint64_t) s.idx_dim * sizeof(float));
    u.bytes = at;
    return u;
}

/// Gets the attention's view of a layer in units: the format's arrays as offsets (plus kKvUnitOrigin) and `units`,
/// the device table that `kv_unit_table` fills.
inline QsaAttnPools kv_unit_pools(const KvUnitLayout& u, KvFormat f, const int64_t* units) {
    const auto at = [](uint64_t offset) { return (uintptr_t) (kKvUnitOrigin + offset); };
    QsaAttnPools p;
    if (f == kKvF16) {
        p.k_pool = (const uint16_t*) at(u.k);
        p.v_pool = (const uint16_t*) at(u.v);
    }
    if (f == kKvInt8 || f == kKvHybrid) {
        p.k_q = (const int8_t*) at(u.k);
        p.k_scale = (const uint16_t*) at(u.k_scale);
    }
    if (f == kKvInt8) {
        p.v_q = (const int8_t*) at(u.v);
        p.v_scale = (const uint16_t*) at(u.v_scale);
    }
    if (f == kKvQ4)
        p.k_q4 = (const uint8_t*) at(u.k);
    if (f == kKvQ4 || f == kKvHybrid)
        p.v_q4 = (const uint8_t*) at(u.v);
    p.units = units;
    return p;
}

/// Gets the pooled-key array of a layer in units (its offset plus kKvUnitOrigin), the `pooled` the indexer takes.
inline float* kv_unit_keys(const KvUnitLayout& u) { return (float*) (uintptr_t) (kKvUnitOrigin + u.key); }

/// Gets the host image of a `units` table for `blocks` blocks: block b is unit `b % per_run` of the run of pages that
/// starts at `runs[b / per_run]` (a run holds `run_bytes / unit_bytes` units of KvUnitLayout::bytes each), for a
/// conversation's region in the page pool, whose blocks are such runs (PagePool::conversation_blocks).
inline std::vector<int64_t> kv_unit_table(const std::vector<uint8_t*>& runs, uint64_t run_bytes, uint64_t unit_bytes,
                                          int64_t blocks) {
    const uint64_t per_run = run_bytes / unit_bytes;
    std::vector<int64_t> table((size_t) blocks);
    for (int64_t b = 0; b < blocks; ++b)
        table[(size_t) b] = (int64_t) ((uintptr_t) runs.at((size_t) ((uint64_t) b / per_run)) +
                                       (uint64_t) b % per_run * unit_bytes - kKvUnitOrigin);
    return table;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
/// Gets cell `cell`'s place for KV head `kvh`, in one integer.  `units` null (packed): its row through `page_table`,
/// or -1 for a block the KV streaming has not made resident (page -1).  In units: its unit's base plus its row in the
/// unit, which `kv_row` takes apart (a base is a multiple of 16 and a unit has at most 16 rows).
__device__ __forceinline__ long long kv_place(const int32_t* page_table, const int64_t* units, int cell, int kvh,
                                              int n_kv_heads, int page_size) {
    const int block = cell / page_size;
    if (units)
        return __ldg(units + block) + kvh * page_size + cell % page_size;
    const long long page = (long long) __ldg(page_table + block);
    return page < 0 ? -1 : (page * n_kv_heads + kvh) * page_size + (cell % page_size);
}

/// Gets array `a` moved by `base` bytes.
template <class T> __device__ __forceinline__ T* kv_at(T* a, long long base) {
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(a) + base);
}

/// Gets the first of `row_elems` elements of array `a` at a cell's place (kv_place); the same arithmetic in both
/// layouts, as the packed layout's base is 0 and its row the whole place.
template <class T>
__device__ __forceinline__ T* kv_row(const int64_t* units, T* a, long long place, long long row_elems) {
    const long long row_mask = units ? 15 : -1;
    return kv_at(a, place & ~row_mask) + (place & row_mask) * row_elems;
}

/// Gets block `b`'s pooled indexer key, `idx_dim` floats: `pooled` row b, or in units the key of b's unit.
template <class T> __device__ __forceinline__ T* kv_key(T* pooled, const int64_t* units, long long b, int idx_dim) {
    return units ? kv_at(pooled, __ldg(units + b)) : pooled + b * idx_dim;
}
#endif

}  // namespace strata::kernels

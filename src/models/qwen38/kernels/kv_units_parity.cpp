// src/models/qwen38/kernels/kv_units_parity.cpp - M2b (docs/MEMORY_MANAGER_PLAN.md): the QSA kernels on the page
// pool's units (kv_units.hpp) against the same kernels on today's packed pools (GPU, no model).
//
// For every KV format, one random conversation of an odd number of cells is written in both layouts: its first part
// by the prompt path's append, the rest one cell at a time by the decode append.  The units sit in pages that are
// separate allocations, used in a shuffled order.  Everything must be BITWISE the same in both layouts: every cell's
// stored bytes, the gather, the decode attention (one query, a batch, the slots form), the prompt attention, and the
// indexer's pooled keys and block scores.  Prints what it compared; exit 1 on any difference.
#include "strata/common/kernels/rope.hpp"
#include "strata/models/qwen38/kernels/kv_q4.hpp"
#include "strata/models/qwen38/kernels/kv_q8.hpp"
#include "strata/models/qwen38/kernels/kv_units.hpp"
#include "strata/models/qwen38/kernels/native_qsa_indexer.hpp"
#include "strata/models/qwen38/kernels/native_qsa_score.hpp"
#include "strata/models/qwen38/kernels/qsa.hpp"
#include "strata/models/qwen38/kernels/qsa_decode_attn.hpp"
#include "strata/models/qwen38/kernels/qsa_prompt_attn.hpp"
#include "strata/models/qwen38/kernels/qsa_select.hpp"
#include "strata/models/qwen38/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <class T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

template <class T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

template <class T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

/// Records one comparison of two device buffers of `n` elements.
template <class T> void expect_same(const char* what, const T* a, const T* b, size_t n) {
    const std::vector<T> x = download(a, n), y = download(b, n);
    const bool same = std::memcmp(x.data(), y.data(), n * sizeof(T)) == 0;
    std::printf("  %-34s %s\n", what, same ? "identical" : "DIFFERENT");
    if (!same)
        ++g_fail;
}

const char* format_name(k::KvFormat f) {
    return f == k::kKvF16 ? "fp16" : f == k::kKvInt8 ? "int8" : f == k::kKvQ4 ? "q4_0" : "k8v4";
}

/// One QSA layer's K/V in one layout, with the appends the engine makes for its format (layer.cpp, prefill.cpp).
struct Layer {
    k::KvFormat format;
    k::QsaAttnPools p;   ///< the readers' view; the writers take the same pointers

    template <class T> static T* w(const T* x) { return const_cast<T*>(x); }

    /// Appends cells [pos0, pos0 + T) of K and V ([T][n_head_kv][head_dim]) as the prompt path does.
    void append_prompt(const float* K, const float* V, int64_t pos0, int64_t T, const k::QsaShapes& s) const {
        if (format == k::kKvQ4) {
            k::kv_append_q4(w(p.k_q4), w(p.v_q4), p.page_table, pos0, T, K, V, s, nullptr, nullptr, nullptr, p.units);
        } else if (format == k::kKvHybrid) {
            strata::prefill::kv_append(K, K, T, pos0, p.page_table, s.page_size, nullptr, nullptr, w(p.k_q),
                                       w(p.k_q), w(p.k_scale), w(p.k_scale), nullptr, nullptr, nullptr, p.units);
            k::kv_append_q4(w(p.v_q4), w(p.v_q4), p.page_table, pos0, T, V, V, s, nullptr, nullptr, nullptr, p.units);
        } else {
            strata::prefill::kv_append(K, V, T, pos0, p.page_table, s.page_size, w(p.k_pool), w(p.v_pool), w(p.k_q),
                                       w(p.v_q), w(p.k_scale), w(p.v_scale), nullptr, nullptr, nullptr, p.units);
        }
    }
    /// Appends the cell at step[kStepPos] as the decode step does.
    void append_step(const float* kcur, const float* vcur, const int32_t* step, const k::QsaShapes& s) const {
        if (format == k::kKvF16) {
            k::kv_append_step(w(p.k_pool), w(p.v_pool), p.page_table, step, kcur, vcur, s, nullptr, nullptr, p.units);
        } else if (format == k::kKvInt8) {
            k::kv_append_q8_step(w(p.k_q), w(p.v_q), w(p.k_scale), w(p.v_scale), p.page_table, step, kcur, vcur, s,
                                 nullptr, nullptr, p.units);
        } else if (format == k::kKvQ4) {
            k::kv_append_q4_step(w(p.k_q4), w(p.v_q4), p.page_table, step, kcur, vcur, s, nullptr, nullptr, p.units);
        } else {
            k::kv_append_q8_step(w(p.k_q), w(p.k_q), w(p.k_scale), w(p.k_scale), p.page_table, step, kcur, kcur, s,
                                 nullptr, nullptr, p.units);
            k::kv_append_q4_step(w(p.v_q4), w(p.v_q4), p.page_table, step, vcur, vcur, s, nullptr, nullptr, p.units);
        }
    }
    /// Gathers the cells `ids` names into fp16 scratch, as the non-fused attention does.
    void gather(const int32_t* ids, const int32_t* step, int64_t cap, const k::QsaShapes& s, uint16_t* ks,
                uint16_t* vs) const {
        if (format == k::kKvF16) {
            k::kv_gather_step(p.k_pool, p.v_pool, p.page_table, ids, step, cap, s, ks, vs, nullptr, p.units);
        } else if (format == k::kKvInt8) {
            k::kv_gather_q8_step(p.k_q, p.v_q, p.k_scale, p.v_scale, p.page_table, ids, step, cap, s, ks, vs, nullptr,
                                 p.units);
        } else if (format == k::kKvQ4) {
            k::kv_gather_q4_step(p.k_q4, p.v_q4, p.page_table, ids, step, cap, s, ks, vs, nullptr, p.units);
        } else {
            k::kv_gather_q8_step(p.k_q, p.k_q, p.k_scale, p.k_scale, p.page_table, ids, step, cap, s, ks, ks, nullptr,
                                 p.units);
            k::kv_gather_q4_step(p.v_q4, p.v_q4, p.page_table, ids, step, cap, s, vs, vs, nullptr, p.units);
        }
    }
};

/// One QSA layer's indexer state in one layout: its own tail, dead row and block position, and the pooled keys in a
/// packed buffer or in the units.
k::QsaIndexerBuffers make_indexer(const k::QsaShapes& s, int64_t blocks, float* unit_keys, const int64_t* units) {
    k::QsaIndexerBuffers b;
    b.tail = dalloc<float>((size_t) ((s.idx_block - 1) * s.idx_dim));
    b.dead = dalloc<float>((size_t) s.idx_dim);
    b.pooled = units ? unit_keys : dalloc<float>((size_t) (blocks * s.idx_dim));
    b.block_pos = dalloc<int32_t>(1);
    b.units = units;
    return b;
}

/// Gets the bytes of one block of each array (K, V, K scales, V scales) of a unit: the packed layout's page strides.
std::vector<uint64_t> get_array_bytes(const k::KvUnitLayout& u) {
    const uint64_t ks_end = u.v_scale ? u.v_scale : u.key;
    return {u.v - u.k, (u.k_scale ? u.k_scale : u.key) - u.v, u.k_scale ? ks_end - u.k_scale : 0,
            u.v_scale ? u.key - u.v_scale : 0};
}

/// Gets the packed layout for `blocks` blocks with the identity page table.
Layer make_packed(k::KvFormat f, const k::KvUnitLayout& u, int64_t blocks) {
    std::vector<int32_t> table((size_t) blocks);
    for (int64_t b = 0; b < blocks; ++b)
        table[(size_t) b] = (int32_t) b;
    const std::vector<uint64_t> bytes = get_array_bytes(u);
    uint8_t* a[4];
    for (int i = 0; i < 4; ++i)
        a[i] = bytes[i] ? dalloc<uint8_t>((size_t) (bytes[i] * blocks)) : nullptr;
    // kv_unit_pools places the format's arrays; the packed ones take the same slots
    k::QsaAttnPools p = k::kv_unit_pools(u, f, nullptr);
    const auto place = [&](auto& field, int array) {
        if (field)
            field = (std::remove_reference_t<decltype(field)>) a[array];
    };
    place(p.k_pool, 0);
    place(p.k_q, 0);
    place(p.k_q4, 0);
    place(p.v_pool, 1);
    place(p.v_q, 1);
    place(p.v_q4, 1);
    place(p.k_scale, 2);
    place(p.v_scale, 3);
    p.page_table = upload(table);
    return {f, p};
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, NH = (int) s.n_head;
    // 301 cells: 76 blocks with a partial last one; 5 units per page, so 16 pages for the 77 units (one ahead)
    const int cells = 301, prompt_cells = 123, per_page = 5;
    const int64_t blocks = cells / s.idx_block + 2;
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K((size_t) cells * H * D), V(K.size());
    for (float& x : K)
        x = nd(rng);
    for (float& x : V)
        x = nd(rng);
    const float *d_K = upload(K), *d_V = upload(V);

    // the queries and their selections: all cells, and odd random subsets (ascending, as topk returns them)
    const int nq = 3;
    const int64_t cap = cells;
    std::vector<float> q((size_t) nq * NH * D);
    for (float& x : q)
        x = nd(rng);
    std::vector<int32_t> ids((size_t) nq * cap, 0), steps((size_t) nq * k::kStepCount);
    const int widths[nq] = {cells, 97, 211};
    for (int z = 0; z < nq; ++z) {
        std::vector<int32_t> all(cells);
        for (int c = 0; c < cells; ++c)
            all[c] = c;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + widths[z]);
        std::copy(all.begin(), all.begin() + widths[z], ids.begin() + (size_t) z * cap);
        k::qsa_step_fill(&steps[(size_t) z * k::kStepCount], cells - 1, s);
        steps[(size_t) z * k::kStepCount + k::kStepWidth] = widths[z];
    }
    const float* d_q = upload(q);
    std::vector<float> raw((size_t) cells * s.idx_dim), gamma((size_t) s.idx_dim);
    std::vector<float> q_idx((size_t) nq * s.idx_n_head * s.idx_dim);
    for (float& x : raw)
        x = nd(rng);
    for (float& x : gamma)
        x = 0.5f + std::abs(nd(rng));
    for (float& x : q_idx)
        x = nd(rng);
    const float *d_raw = upload(raw), *d_gamma = upload(gamma), *d_q_idx = upload(q_idx);
    std::vector<float> cos_tab((size_t) (cells * s.n_rot / 2)), sin_tab(cos_tab.size());
    k::build_rope_table((int) s.n_rot, k::qsa_freq_base(), cells, cos_tab.data(), sin_tab.data());
    const float *d_cos = upload(cos_tab), *d_sin = upload(sin_tab);
    const int32_t *d_ids = upload(ids), *d_steps = upload(steps);

    for (const k::KvFormat f : {k::kKvInt8, k::kKvF16, k::kKvQ4, k::kKvHybrid}) {
        std::printf("%s:\n", format_name(f));
        const k::KvUnitLayout u = k::kv_unit_layout(s, f);
        const Layer packed = make_packed(f, u, blocks);
        const std::vector<uint64_t> array_bytes = get_array_bytes(u);
        // the units: pages of `per_page` units and some slack, separate allocations, in a shuffled order
        const uint64_t page_bytes = per_page * u.bytes + 256;
        std::vector<uint8_t*> pages((size_t) ((blocks + per_page - 1) / per_page));
        for (uint8_t*& page : pages)
            page = dalloc<uint8_t>((size_t) page_bytes);
        std::shuffle(pages.begin(), pages.end(), rng);
        const int64_t* d_units = upload(k::kv_unit_table(pages, page_bytes, u.bytes, blocks));
        const Layer units{f, k::kv_unit_pools(u, f, d_units)};

        // ---- the appends
        int32_t* d_step = dalloc<int32_t>(k::kStepCount);
        for (const Layer* l : {&packed, &units})
            l->append_prompt(d_K, d_V, 0, prompt_cells, s);
        for (int c = prompt_cells; c < cells; ++c) {
            int32_t step[k::kStepCount];
            k::qsa_step_fill(step, c, s);
            ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");
            for (const Layer* l : {&packed, &units})
                l->append_step(d_K + (size_t) c * H * D, d_V + (size_t) c * H * D, d_step, s);
        }
        ck(cudaDeviceSynchronize(), "append");
        {   // every block's arrays, byte for byte: packed page b against unit b
            const uint64_t offset[4] = {u.k, u.v, u.k_scale, u.v_scale};
            const k::QsaAttnPools& pp = packed.p;
            const void* arrays[4] = {
                pp.k_pool ? (const void*) pp.k_pool : pp.k_q ? (const void*) pp.k_q : (const void*) pp.k_q4,
                pp.v_pool ? (const void*) pp.v_pool : pp.v_q ? (const void*) pp.v_q : (const void*) pp.v_q4,
                pp.k_scale, pp.v_scale};
            const int64_t used = (cells + s.idx_block - 1) / s.idx_block;
            int bad = 0;
            for (int a = 0; a < 4; ++a) {
                if (array_bytes[a] == 0)
                    continue;
                const std::vector<uint8_t> flat =
                    download((const uint8_t*) arrays[a], (size_t) (array_bytes[a] * used));
                for (int64_t b = 0; b < used; ++b) {
                    const uint8_t* unit = pages[(size_t) (b / per_page)] + (b % per_page) * u.bytes;
                    const std::vector<uint8_t> got = download(unit + offset[a], (size_t) array_bytes[a]);
                    bad += std::memcmp(got.data(), &flat[(size_t) (b * array_bytes[a])], got.size()) != 0;
                }
            }
            std::printf("  %-34s %s\n", "appended K/V bytes", bad ? "DIFFERENT" : "identical");
            g_fail += bad != 0;
        }

        // ---- the gather
        uint16_t* ks[2] = {dalloc<uint16_t>((size_t) cap * H * D), dalloc<uint16_t>((size_t) cap * H * D)};
        uint16_t* vs[2] = {dalloc<uint16_t>((size_t) cap * H * D), dalloc<uint16_t>((size_t) cap * H * D)};
        packed.gather(d_ids + cap, d_steps + k::kStepCount, cap, s, ks[0], vs[0]);
        units.gather(d_ids + cap, d_steps + k::kStepCount, cap, s, ks[1], vs[1]);
        expect_same("gather K", ks[0], ks[1], (size_t) widths[1] * H * D);
        expect_same("gather V", vs[0], vs[1], (size_t) widths[1] * H * D);

        // ---- the attention
        const size_t out = (size_t) nq * NH * D;
        float* scratch = dalloc<float>((size_t) (k::qsa_decode_attn_scratch_floats(cap, s) * nq));
        float* attn[2] = {dalloc<float>(out), dalloc<float>(out)};
        const Layer* both[2] = {&packed, &units};
        for (int i = 0; i < 2; ++i)
            k::qsa_decode_attn_step(d_q, both[i]->p, d_ids, d_steps, cap, s, scratch, attn[i], nullptr);
        expect_same("decode attention, one query", attn[0], attn[1], (size_t) NH * D);
        for (int i = 0; i < 2; ++i)
            k::qsa_decode_attn_batch(d_q, both[i]->p, d_ids, d_steps, cap, s, scratch, attn[i], nq, nullptr);
        expect_same("decode attention, a batch", attn[0], attn[1], out);
        bool slots = true;
        for (int i = 0; i < 2; ++i) {
            k::QsaAttnSlotPools sp;
            for (int z = 0; z < nq; ++z)
                sp.p[z] = both[i]->p;
            slots = k::qsa_decode_attn_slots(d_q, sp, d_ids, d_steps, cap, s, scratch, attn[i], nq, nullptr) && slots;
        }
        if (slots)
            expect_same("decode attention, slots", attn[0], attn[1], out);
        else
            std::printf("  %-34s not run on this device\n", "decode attention, slots");
        bool prompt = true;
        for (int i = 0; i < 2; ++i)
            prompt = k::qsa_prompt_attn_batch(d_q, both[i]->p, d_ids, d_steps, cap, s, attn[i], nq, nullptr) && prompt;
        if (prompt)
            expect_same("prompt attention", attn[0], attn[1], out);
        else
            std::printf("  %-34s not run on this device\n", "prompt attention");
        ck(cudaDeviceSynchronize(), "attention");

        // ---- the indexer: the native appends (prompt, then steps) and the reference append into the same units
        const k::RopeScaling rope;
        cudaStream_t stream = nullptr;   // the native indexer wants an explicit stream
        ck(cudaStreamCreate(&stream), "stream");
        const k::QsaIndexerBuffers ib[2] = {make_indexer(s, blocks, nullptr, nullptr),
                                            make_indexer(s, blocks, k::kv_unit_keys(u), d_units)};
        for (const k::QsaIndexerBuffers& b : ib)
            k::native_qsa_indexer_append_batch(d_raw, prompt_cells, 0, 0, d_gamma, k::qsa_rms_eps(), b, s, cells,
                                               rope, stream);
        for (int c = prompt_cells; c < cells; ++c) {
            int32_t step[k::kStepCount];
            k::qsa_step_fill(step, c, s);
            ck(cudaStreamSynchronize(stream), "step");
            ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");
            for (const k::QsaIndexerBuffers& b : ib)
                k::native_qsa_indexer_append(d_raw + (size_t) c * s.idx_dim, d_step + k::kStepPos, 0, d_gamma,
                                             k::qsa_rms_eps(), b, s, cells, rope, stream);
        }
        ck(cudaStreamSynchronize(stream), "indexer append");
        const int64_t keys = cells / s.idx_block + 1;   // the completed blocks and the spare
        const auto compare_keys = [&](const char* what, const k::QsaIndexerBuffers& a) {
            const std::vector<float> flat = download(a.pooled, (size_t) (keys * s.idx_dim));
            int bad = 0;
            for (int64_t r = 0; r < keys; ++r) {
                const uint8_t* unit = pages[(size_t) (r / per_page)] + (r % per_page) * u.bytes;
                const std::vector<float> got = download((const float*) (unit + u.key), (size_t) s.idx_dim);
                bad += std::memcmp(got.data(), &flat[(size_t) (r * s.idx_dim)], got.size() * sizeof(float)) != 0;
            }
            std::printf("  %-34s %s\n", what, bad ? "DIFFERENT" : "identical");
            g_fail += bad != 0;
        };
        compare_keys("pooled keys, native append", ib[0]);

        float* scores[2] = {dalloc<float>((size_t) (nq * blocks)), dalloc<float>((size_t) (nq * blocks))};
        for (int i = 0; i < 2; ++i)
            k::qsa_block_scores(ib[i].pooled, ib[i].dead, d_q_idx, d_steps, nq, blocks, s, scores[i], nullptr, -1,
                                ib[i].units);
        expect_same("block scores", scores[0], scores[1], (size_t) (nq * blocks));
        bool tc = true;
        for (int i = 0; i < 2; ++i)
            tc = k::qsa_block_scores_tc(ib[i].pooled, ib[i].dead, d_q_idx, d_steps, nq, blocks, s, scores[i], nullptr,
                                        -1, ib[i].units) && tc;
        if (tc)
            expect_same("block scores, matrix cores", scores[0], scores[1], (size_t) (nq * blocks));
        else
            std::printf("  %-34s not run on this device\n", "block scores, matrix cores");
        float* cell_scores[2] = {dalloc<float>((size_t) cells), dalloc<float>((size_t) cells)};
        for (int i = 0; i < 2; ++i)
            k::qsa_index_step(ib[i].pooled, d_q_idx, nullptr, s, d_steps, blocks, cell_scores[i], nullptr,
                              ib[i].units);
        expect_same("cell scores (qsa_index)", cell_scores[0], cell_scores[1], (size_t) cells);
        for (int i = 0; i < 2; ++i)
            k::native_qsa_score(ib[i].pooled, d_q_idx, nullptr, s, d_steps, keys, cells, cell_scores[i], stream,
                                ib[i].units);
        ck(cudaStreamSynchronize(stream), "native score");
        expect_same("cell scores (native)", cell_scores[0], cell_scores[1], (size_t) cells);

        // the reference append (indexer_key_append) over the whole conversation, into fresh state
        const k::QsaIndexerBuffers rb[2] = {make_indexer(s, blocks, nullptr, nullptr),
                                            make_indexer(s, blocks, k::kv_unit_keys(u), d_units)};
        for (int c = 0; c < cells; ++c) {
            int32_t step[k::kStepCount];
            k::qsa_step_fill(step, c, s);
            ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");
            for (const k::QsaIndexerBuffers& b : rb)
                k::indexer_key_append(d_raw + (size_t) c * s.idx_dim, d_step + k::kStepPos, 0, d_gamma,
                                      k::qsa_rms_eps(), b, s, d_cos, d_sin, nullptr);
        }
        ck(cudaDeviceSynchronize(), "reference indexer append");
        compare_keys("pooled keys, reference append", rb[0]);
    }
    std::printf("kv_units_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

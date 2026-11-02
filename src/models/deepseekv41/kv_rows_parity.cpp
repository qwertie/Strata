// src/models/deepseekv41/kv_rows_parity.cpp - DeepSeek V4.1's cache rows in the page pool's blocks against the same
// rows back to back (kv_rows.hpp, docs/MEMORY_MANAGER_PLAN.md step D2): the writer (finish's packed rows, in pieces
// that cross blocks), the readers (index_score in the window, chunk and candidate paths; sparse_attn every way: the
// scratch, the staged kernels of 1 to 8 heads a block, a chunk's slab, and with DS4_ATTN_SPLIT=1 the split-K kernel)
// and the host's copies block by block must give the same bytes and bits.  The blocks are pages of 92,160 bytes, in
// a shuffled order, with other bytes between them.  Needs a GPU, no model.
#include "ds4_kernels.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace dk = strata::ds4;

namespace {

int g_fail = 0;

#define CU(call)                                                                                \
    do {                                                                                        \
        const cudaError_t e_ = (call);                                                          \
        if (e_ != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)

constexpr int HD = 512, IX = 128, NH = 64, IXH = 32, WIN = 128, RD = 64, SEL = 512, NC = 2500;
constexpr uint32_t kPage = 92160;   // the DeepSeek pack's page (conversation_layout_test)

std::mt19937 rng(20261102);
float frand(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }

template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc((void**) &p, h.size() * sizeof(T)));
    CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> std::vector<T> host(const T* p, size_t n) {
    CU(cudaDeviceSynchronize());
    std::vector<T> h(n);
    CU(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void check(const std::string& what, bool same) {
    if (!same)
        ++g_fail;
    std::printf("%-58s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

/// One cache of NC rows twice: back to back, and in shuffled pages of a pool with a page more than they need
/// between every two.
struct TwoLayouts {
    dk::KvRows flat, paged;
    uint8_t* pool = nullptr;
    std::vector<uint8_t*> table;

    TwoLayouts(int fmt, int d) {
        flat = dk::make_kv_rows(nullptr, fmt, d);
        CU(cudaMalloc((void**) &flat.base, (size_t) NC * flat.bytes));
        CU(cudaMemset(flat.base, 0, (size_t) NC * flat.bytes));
        paged = flat;
        paged.base = nullptr;
        paged.per_block = kPage / flat.bytes;
        const int blocks = (NC + (int) paged.per_block - 1) / (int) paged.per_block;
        CU(cudaMalloc((void**) &pool, (size_t) 2 * blocks * kPage));
        CU(cudaMemset(pool, 0xa5, (size_t) 2 * blocks * kPage));
        std::vector<int> slot(blocks);
        for (int b = 0; b < blocks; ++b)
            slot[(size_t) b] = 2 * b + 1;
        std::shuffle(slot.begin(), slot.end(), rng);
        for (int b = 0; b < blocks; ++b)
            table.push_back(pool + (size_t) slot[(size_t) b] * kPage);
        // (the unused rows of a block are the flat cache's zeros too)
        for (uint8_t* p : table)
            CU(cudaMemset(p, 0, kPage));
        paged.blocks = dev(table);
        paged.host_blocks = table.data();
    }
    /// Gets every row of `rows` through the host's block-by-block copies.
    static std::vector<uint8_t> read(const dk::KvRows& rows) {
        CU(cudaDeviceSynchronize());
        std::vector<uint8_t> h((size_t) NC * rows.bytes);
        rows.for_each_run(0, NC, [&](size_t off, size_t bytes, const uint8_t* at) {
            CU(cudaMemcpy(h.data() + off, at, bytes, cudaMemcpyDeviceToHost));
            return true;
        });
        return h;
    }
};

/// Packs the NC rows of x (d floats each) into both layouts with `quant` in the pieces `cuts` gives, as the window
/// and chunk paths' finish does.
void write_rows(TwoLayouts& c, const float* x, int d, int quant, const float* w, const float* freqs,
                const std::vector<int>& cuts) {
    for (size_t i = 0; i + 1 < cuts.size(); ++i)
        for (const dk::KvRows* rows : {&c.flat, &c.paged}) {
            dk::Finish f;
            f.w = w;
            f.rd = RD;
            f.pos = cuts[i] * 4;
            f.pos_step = 4;
            f.freqs = freqs;
            f.quant = quant;
            f.pdst = *rows;
            f.pdst_row = cuts[i];
            dk::finish(const_cast<float*>(x) + (size_t) cuts[i] * d, cuts[i + 1] - cuts[i], d, d, f, nullptr);
        }
}

}  // namespace

int main() {
    std::printf("the cache rows in the page pool's blocks against rows back to back (DS4_ATTN_SPLIT=%s)\n",
                std::getenv("DS4_ATTN_SPLIT") ? std::getenv("DS4_ATTN_SPLIT") : "unset");
    TwoLayouts ckv(dk::kKvFp4, HD), ik(dk::kKvFp4p, IX);
    std::printf("blocks: %u compressed rows (%zu blocks), %u index keys (%zu blocks)\n", ckv.paged.per_block,
                ckv.table.size(), ik.paged.per_block, ik.table.size());

    // the writer: a prompt chunk's rows, then windows' of 1 to 16 rows, so that pieces cross blocks
    std::vector<int> cuts = {0, 700};
    while (cuts.back() < NC)
        cuts.push_back(std::min(NC, cuts.back() + 1 + (int) (rng() % 16)));
    std::vector<float> xk((size_t) NC * HD), xi((size_t) NC * IX), wi(IX), freqs(RD / 2);
    for (float& v : xk)
        v = frand(2.0f);
    for (float& v : xi)
        v = frand(2.0f);
    for (float& v : wi)
        v = 1.0f + frand(0.5f);
    for (float& v : freqs)
        v = frand(0.1f);
    float *d_xk = dev(xk), *d_xi = dev(xi), *d_wi = dev(wi), *d_freqs = dev(freqs);
    write_rows(ckv, d_xk, HD, dk::kKvFp4, nullptr, d_freqs, cuts);
    write_rows(ik, d_xi, IX, dk::kKvFp4p, d_wi, d_freqs, cuts);
    const std::vector<uint8_t> ckv_flat = host(ckv.flat.base, (size_t) NC * ckv.flat.bytes);
    const std::vector<uint8_t> ik_flat = host(ik.flat.base, (size_t) NC * ik.flat.bytes);
    const auto written = [](const std::vector<uint8_t>& v) {
        return std::any_of(v.begin(), v.end(), [](uint8_t b) { return b != 0; });
    };
    check("finish: packed compressed rows (" + std::to_string(cuts.size() - 1) + " pieces)",
          written(ckv_flat) && ckv_flat == TwoLayouts::read(ckv.paged));
    check("finish: packed index keys", written(ik_flat) && ik_flat == TwoLayouts::read(ik.paged));
    check("the host's block-by-block copies of rows back to back", ckv_flat == TwoLayouts::read(ckv.flat));

    cudaStream_t s = nullptr;
    // (rows of `n` values `stride` apart: the values past n are not written)
    const auto same = [&](const float* a, const float* b, size_t n, int rows = 1, size_t stride = 0) {
        const std::vector<float> ha = host(a, rows * std::max(n, stride)), hb = host(b, rows * std::max(n, stride));
        for (int r = 0; r < rows; ++r)
            if (std::memcmp(ha.data() + r * stride, hb.data() + r * stride, n * 4) != 0)
                return false;
        return true;
    };

    // the indexer: a window's queries, a chunk's (positions made on the device), and a chunk's over candidates
    {
        const int T = 16, BLK = 64, NCAND = 8;
        std::vector<float> q((size_t) T * IXH * IX), w((size_t) T * 64);
        for (float& v : q)
            v = frand(1.0f);
        for (float& v : w)
            v = frand(0.3f);
        std::vector<int> cand((size_t) T * NCAND);
        for (int& v : cand)
            v = (int) (rng() % (unsigned) (NC / BLK));
        float *d_q = dev(q), *d_w = dev(w), *o1 = nullptr, *o2 = nullptr;
        int* d_cand = dev(cand);
        const long long ss = NC + 64;
        CU(cudaMalloc((void**) &o1, (size_t) T * ss * 4));
        CU(cudaMalloc((void**) &o2, (size_t) T * ss * 4));
        const auto run = [&](const char* what, int n, const int* cd, int blk, const dk::Rows& lim) {
            CU(cudaMemset(o1, 0x7f, (size_t) T * ss * 4));
            CU(cudaMemset(o2, 0x3f, (size_t) T * ss * 4));
            dk::index_score(d_q, d_w, ik.flat, IXH, IX, T, n, cd, NCAND, blk, lim, o1, ss, s);
            dk::index_score(d_q, d_w, ik.paged, IXH, IX, T, n, cd, NCAND, blk, lim, o2, ss, s);
            check(std::string("index_score: ") + what, same(o1, o2, (size_t) n, T, (size_t) ss));
        };
        dk::Rows lim;
        for (int t = 0; t < T; ++t) {
            lim.n[t] = NC - 37 * t;
            lim.k[t] = 512;
        }
        run("a window", NC, nullptr, 1, lim);
        dk::Rows chunk;
        chunk.gen = 1;
        chunk.p0 = 4LL * NC - 4 * T;
        chunk.R = 4;
        chunk.kcap = 512;
        chunk.ccap = NCAND;
        chunk.blk = BLK;
        run("a chunk", (int) ((chunk.p0 + T) / chunk.R), nullptr, 1, chunk);
        dk::Rows cands = chunk;
        cands.gen = 2;
        run("a chunk's candidates", NCAND * BLK, d_cand, BLK, cands);
        for (void* p : {(void*) d_q, (void*) d_w, (void*) d_cand, (void*) o1, (void*) o2})
            CU(cudaFree(p));
    }

    // the sparse attention: a decode step and short windows (gen 0) every way, and a chunk's slab (gen 1)
    {
        const int TM = 32;
        std::vector<float> q((size_t) TM * NH * HD), sink(NH);
        for (float& v : q)
            v = frand(0.2f);
        for (float& v : sink)
            v = frand(1.0f);
        std::vector<uint16_t> ext((size_t) (WIN + TM) * HD);
        for (uint16_t& v : ext) {
            const float f = frand(1.0f);
            uint32_t u;
            std::memcpy(&u, &f, 4);
            v = (uint16_t) (u >> 16);
        }
        std::vector<int> sel((size_t) TM * SEL);
        for (int& v : sel)
            v = (int) (rng() % (unsigned) NC);
        float *d_q = dev(q), *d_sink = dev(sink), *o1 = nullptr, *o2 = nullptr;
        uint16_t* d_ext = dev(ext);
        int* d_sel = dev(sel);
        dk::AttnScratch scr;
        scr.max_t = TM;
        scr.cap = SEL;
        CU(cudaMalloc((void**) &scr.rows, (size_t) TM * SEL * HD * 2));
        CU(cudaMalloc((void**) &o1, q.size() * 4));
        CU(cudaMalloc((void**) &o2, q.size() * 4));
        const float scale = 1.0f / 22.627417f;
        // (T queries' outputs from the rows back to back in o1, from the pages in o2: the same bits?)
        const auto run = [&](int T, const dk::AttnRows& ar, int g) {
            const size_t n = (size_t) T * NH * HD;
            CU(cudaMemset(o1, 0x7f, n * 4));
            CU(cudaMemset(o2, 0x3f, n * 4));
            dk::sparse_attn(d_q, d_ext, ckv.flat, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o1, s, dk::kKvFp4, g, &scr);
            dk::sparse_attn(d_q, d_ext, ckv.paged, d_sel, SEL, ar, d_sink, NH, HD, T, scale, o2, s, dk::kKvFp4, g,
                            &scr);
            return same(o1, o2, n);
        };
        for (int T : {1, 2, 5, 16}) {
            dk::AttnRows ar;
            for (int t = 0; t < T; ++t) {
                ar.win0[t] = t + 1;
                ar.nwin[t] = WIN;
                ar.nsel[t] = SEL - 31 * t;
            }
            for (int g : {-1, 0, 1, 2, 4, 8})
                check("sparse_attn: " + std::to_string(T) + " queries, heads_per_block " + std::to_string(g),
                      run(T, ar, g));
        }
        dk::AttnRows ar;   // a chunk's slab at 4 (NC - TM): every query its 512 rows
        ar.gen = 1;
        ar.p0 = 4LL * (NC - TM);
        ar.win = WIN;
        ar.R = 4;
        ar.kcap = SEL;
        for (int g : {-1, 0})
            check("sparse_attn: a chunk's slab of " + std::to_string(TM) + ", heads_per_block " + std::to_string(g),
                  run(TM, ar, g));
        for (void* p : {(void*) d_q, (void*) d_sink, (void*) d_ext, (void*) d_sel, (void*) scr.rows, (void*) o1,
                        (void*) o2})
            CU(cudaFree(p));
    }
    CU(cudaGetLastError());
    std::printf("%s\n", g_fail ? "FAILED" : "all identical");
    return g_fail ? 1 : 0;
}

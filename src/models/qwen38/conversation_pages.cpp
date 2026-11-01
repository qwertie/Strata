// src/models/qwen38/conversation_pages.cpp - a Qwen3.8 conversation's state in a GPU page pool
// (conversation_pages.hpp).
#include "strata/models/qwen38/conversation_pages.hpp"

#include "strata/core/layout.hpp"
#include "strata/models/qwen38/conversation_snapshot.hpp"
#include "strata/models/qwen38/layer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace strata::qwen38 {

ConversationPages::~ConversationPages() {
    for (int64_t* t : tables_)
        cudaFree(t);
}

bool ConversationPages::get_layout(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss,
                                   strata::core::StateLayout& layout, std::string& err) {
    using namespace strata::core;
    const int64_t n_qsa = g.n_qsa_layers();
    if (n_qsa == 0 || ss.qsa_ord0 != 0 || ss.qsa_alloc != n_qsa || ss.gdn_alloc != g.n_gdn_layers() ||
        ss.gdn_state != nullptr) {
        err = "the page pool needs a whole-model session carved without its state (session_set_pool_state)";
        return false;
    }
    const strata::kernels::QsaShapes s = qsa_shapes(g);
    const auto fmt = (strata::kernels::KvFormat) qsa_kv_format(ss.qsa_states[0]);
    try {
        const strata::kernels::KvUnitLayout unit = strata::kernels::kv_unit_layout(s, fmt);
        const uint64_t key_bytes = (uint64_t) s.idx_dim * sizeof(float);
        if (!describe_conversation(g, (unit.bytes - key_bytes) / (uint64_t) s.idx_block, layout, err))
            return false;
        layout.regions.erase(layout.regions.begin());   // the small running state stays in the session
        if (layout.regions.size() != (size_t) (1 + n_qsa) || layout.regions[1].unit_bytes != unit.bytes) {
            err = "the page pool's layout does not match the K/V's units (" + std::to_string(unit.bytes) + " B)";
            return false;
        }
    } catch (const std::exception& e) {
        err = std::string("the page pool: ") + e.what();
        return false;
    }
    return true;
}

bool ConversationPages::init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss,
                             strata::core::PagePool& pool, strata::core::ConversationId id, std::string& err) {
    using namespace strata::core;
    const strata::kernels::QsaShapes s = qsa_shapes(g);
    const int64_t n_qsa = g.n_qsa_layers();
    unit_ = strata::kernels::kv_unit_layout(s, (strata::kernels::KvFormat) qsa_kv_format(ss.qsa_states[0]));
    page_bytes_ = pool.page_bytes();
    if (const std::string why = pool.grow_conversation(id, 0); !why.empty()) {
        err = "the page pool: " + why;
        return false;
    }
    const std::vector<PageId>& gdn = pool.conversation_pages(id, 0);
    ss.gdn_state = (float*) pool.page_address(gdn.at(0));
    idx_block_ = s.idx_block;
    max_blocks_ = ss.max_cells / s.idx_block + 2;
    for (int64_t* t : tables_)   // (an init that failed before)
        (void) cudaFree(t);
    tables_.assign((size_t) n_qsa, nullptr);
    pages_in_table_.assign((size_t) n_qsa, 0);
    for (int64_t j = 0; j < n_qsa; ++j) {
        if (cudaMalloc((void**) &tables_[(size_t) j], (size_t) max_blocks_ * sizeof(int64_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "the page pool: no room for the K/V's unit tables";
            return false;
        }
        cudaMemset(tables_[(size_t) j], 0, (size_t) max_blocks_ * sizeof(int64_t));
        qsa_use_units(g, ss.qsa_states[j], tables_[(size_t) j]);
    }
    if (cudaMemset(ss.gdn_state, 0, gdn.size() * page_bytes_) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("the page pool: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    pool_ = &pool;
    id_ = id;
    return true;
}

bool ConversationPages::reserve(int64_t tokens, std::string& err) {
    // blocks up to the last position's, and the spare one after it that the indexer writes ahead
    const int64_t need = std::min((std::max<int64_t>(tokens, 1) - 1) / idx_block_ + 2, max_blocks_) * idx_block_;
    if (need <= reserved_)
        return true;
    if (const std::string why = pool_->grow_conversation(id_, need); !why.empty()) {
        err = "the conversation's memory is full: " + why;
        return false;
    }
    reserved_ = need;
    const uint64_t per_page = page_bytes_ / unit_.bytes;
    bool added = false;
    for (size_t j = 0; j < tables_.size(); ++j) {
        const std::vector<strata::core::PageId>& pages = pool_->conversation_pages(id_, 1 + j);
        if (pages.size() == pages_in_table_[j])
            continue;
        std::vector<uint8_t*> at;
        for (const strata::core::PageId p : pages)
            at.push_back(pool_->page_address(p));
        for (size_t i = pages_in_table_[j]; i < at.size(); ++i)
            cudaMemset(at[i], 0, page_bytes_);
        const int64_t blocks = std::min((int64_t) (at.size() * per_page), max_blocks_);
        const int64_t first = std::min((int64_t) (pages_in_table_[j] * per_page), blocks);
        const std::vector<int64_t> table = strata::kernels::kv_unit_table(at, page_bytes_, unit_, blocks);
        cudaMemcpy(tables_[j] + first, table.data() + first, (size_t) (blocks - first) * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
        pages_in_table_[j] = pages.size();
        added = true;
    }
    // the kernels that read the tables and pages run on non-blocking streams, which do not wait for these copies
    if (added && cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("the conversation's pages: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

void ConversationPages::truncate(int64_t tokens) {
    const int64_t keep = tokens > 0 ? std::min(((tokens - 1) / idx_block_ + 2) * idx_block_, reserved_) : 0;
    pool_->truncate_conversation(id_, keep);
    reserved_ = keep;
    for (size_t j = 0; j < tables_.size(); ++j)
        pages_in_table_[j] = pool_->conversation_pages(id_, 1 + j).size();
}

uint64_t ConversationPages::bytes() const {
    if (!is_ready())
        return 0;
    uint64_t pages = 0;
    for (size_t r = 0; r <= tables_.size(); ++r)   // the fixed region, then each QSA layer's
        pages += pool_->conversation_pages(id_, r).size();
    return pages * page_bytes_;
}

}  // namespace strata::qwen38

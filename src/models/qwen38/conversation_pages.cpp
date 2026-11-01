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
                                   const strata::core::QsaState* drafter, strata::core::StateLayout& layout,
                                   std::string& err) {
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
        if (drafter != nullptr)
            layout.add(RegionKind::per_block,
                       strata::kernels::kv_unit_layout(s, (strata::kernels::KvFormat) qsa_kv_format(*drafter)).bytes, 1,
                       s.idx_block);
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
    unit_bytes_.clear();
    run_bytes_.clear();
    for (size_t r = 1; r < pool.layout().regions.size(); ++r) {
        unit_bytes_.push_back(pool.layout().regions[r].unit_bytes);
        run_bytes_.push_back(pool.block_bytes(r));
    }
    has_drafter_ = (int64_t) unit_bytes_.size() == n_qsa + 1;
    if (const std::string why = pool.grow_conversation(id, 0); !why.empty()) {
        err = "the page pool: " + why;
        return false;
    }
    ss.gdn_state = (float*) pool.page_address(pool.conversation_blocks(id, 0).at(0));
    idx_block_ = s.idx_block;
    max_blocks_ = ss.max_cells / s.idx_block + 2;
    tables_.resize(unit_bytes_.size(), nullptr);
    runs_in_table_.assign(unit_bytes_.size(), 0);
    for (size_t j = 0; j < tables_.size(); ++j) {
        if (tables_[j] == nullptr &&
            cudaMalloc((void**) &tables_[j], (size_t) max_blocks_ * sizeof(int64_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            tables_[j] = nullptr;
            err = "the page pool: no room for the K/V's unit tables";
            return false;
        }
        cudaMemset(tables_[j], 0, (size_t) max_blocks_ * sizeof(int64_t));
        if ((int64_t) j < n_qsa)
            qsa_use_units(g, ss.qsa_states[j], tables_[j]);
    }
    if (cudaMemset(ss.gdn_state, 0, pool.block_bytes(0)) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("the page pool: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    pool_ = &pool;
    id_ = id;
    return true;
}

bool ConversationPages::reserve(int64_t tokens, std::string& err) {
    const int64_t need = get_pool_tokens(tokens);
    if (need <= reserved_)
        return true;
    if (const std::string why = pool_->grow_conversation(id_, need); !why.empty()) {
        err = "the conversation's memory is full: " + why;
        return false;
    }
    reserved_ = need;
    return write_tables(true, err);
}

bool ConversationPages::refresh(strata::core::SessionState& ss, std::string& err) {
    ss.gdn_state = (float*) pool_->page_address(pool_->conversation_blocks(id_, 0).at(0));
    std::fill(runs_in_table_.begin(), runs_in_table_.end(), 0);
    return write_tables(false, err);
}

/// Writes the units of the pool's blocks that the device tables lack (runs_in_table_ on), zeroing the new blocks'
/// pages first with `zero_new`.
bool ConversationPages::write_tables(bool zero_new, std::string& err) {
    bool added = false;
    for (size_t j = 0; j < tables_.size(); ++j) {
        const std::vector<strata::core::PageId>& runs = pool_->conversation_blocks(id_, 1 + j);
        if (runs.size() == runs_in_table_[j])
            continue;
        std::vector<uint8_t*> at;
        for (const strata::core::PageId p : runs)
            at.push_back(pool_->page_address(p));
        for (size_t i = runs_in_table_[j]; zero_new && i < at.size(); ++i)
            cudaMemset(at[i], 0, run_bytes_[j]);
        const uint64_t per_run = run_bytes_[j] / unit_bytes_[j];
        const int64_t blocks = std::min((int64_t) (at.size() * per_run), max_blocks_);
        const int64_t first = std::min((int64_t) (runs_in_table_[j] * per_run), blocks);
        const std::vector<int64_t> table = strata::kernels::kv_unit_table(at, run_bytes_[j], unit_bytes_[j], blocks);
        cudaMemcpy(tables_[j] + first, table.data() + first, (size_t) (blocks - first) * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
        runs_in_table_[j] = runs.size();
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
        runs_in_table_[j] = pool_->conversation_blocks(id_, 1 + j).size();
}

void ConversationPages::release() {
    pool_->remove_conversation(id_);
    pool_ = nullptr;
    reserved_ = 0;
}

int64_t ConversationPages::get_pool_tokens(int64_t tokens) const {
    // blocks up to the last position's, and the spare one after it that the indexer writes ahead
    return std::min((std::max<int64_t>(tokens, 1) - 1) / idx_block_ + 2, max_blocks_) * idx_block_;
}

uint64_t ConversationPages::bytes() const {
    if (!is_ready())
        return 0;
    uint64_t bytes = 0;
    for (size_t r = 0; r <= tables_.size(); ++r)   // the fixed region, then each table's
        bytes += pool_->conversation_blocks(id_, r).size() * pool_->block_bytes(r);
    return bytes;
}

}  // namespace strata::qwen38

// src/models/deepseekv41/expert_files.cpp - a GGUF pack's routed experts read without the file cache
// (expert_files.hpp).
#include "expert_files.hpp"

#include <algorithm>
#include <atomic>

namespace strata::ds4 {
namespace {

/// The experts read at once: their matrices' reads run side by side on the reading threads.
constexpr size_t kBatch = 4;
/// The reading threads (the old engine's RAM store reads with 8).
constexpr int kReaders = 8;

}  // namespace

ExpertFiles::~ExpertFiles() {
    io_.reset();   // (its threads may still hold staging buffers)
    if (staging_ != nullptr)
        platform::anon_free(staging_, kBatch * 3 * span_);
    for (const platform::fd_t f : direct_)
        platform::close_fd(f);
    for (const platform::fd_t f : buffered_)
        platform::close_fd(f);
}

bool ExpertFiles::open(const std::vector<std::string>& paths, std::vector<ExpertFileLayout> layers, std::string& err) {
    layers_ = std::move(layers);
    for (const std::string& path : paths) {
        direct_.push_back(platform::open_direct(path));
        buffered_.push_back(platform::open_par(path));
        if (buffered_.back() == platform::kBadFd) {
            err = path + ": cannot open";
            return false;
        }
    }
    for (const ExpertFileLayout& l : layers_)   // a matrix with its sector-aligned surroundings
        for (int r = 0; r < 3; ++r)
            span_ = std::max<size_t>(span_, (size_t) platform::direct_span(platform::kDirectAlign - 1, l.bytes[r]));
    staging_ = (uint8_t*) platform::anon_alloc(kBatch * 3 * span_, -1, 1);
    if (staging_ == nullptr) {
        err = "no memory for the expert reads' staging";
        return false;
    }
    io_ = std::make_unique<IoQueue>(kReaders);
    return true;
}

bool ExpertFiles::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const int32_t l = (int32_t) layer, e = (int32_t) expert;
    return copy_blobs(&l, &e, &dst, 1);
}

bool ExpertFiles::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    const std::lock_guard<std::mutex> lock(mu_);
    std::atomic<bool> bad{false};
    for (size_t i0 = 0; i0 < n && !bad.load(); i0 += kBatch) {
        IoQueue::Job jobs[kBatch * 3];
        std::atomic<int> left{0};
        int nj = 0;
        for (size_t i = i0; i < std::min(n, i0 + kBatch); ++i) {
            const ExpertFileLayout& l = layers_[(size_t) layers[i]];
            for (int r = 0; r < 3; ++r) {
                IoQueue::Job& j = jobs[nj];
                j.dfd = direct_[(size_t) l.file[r]];
                j.fd = buffered_[(size_t) l.file[r]];
                j.off = l.off[r] + (uint64_t) experts[i] * l.stride[r];
                j.len = l.bytes[r];
                j.dst = staging_ + (size_t) nj * span_;
                j.keep = dst[i] + l.tr[r];
                j.left = &left;
                j.bad = &bad;
                ++nj;
            }
        }
        left.store(nj);
        io_->push(jobs, nj);
        while (left.load(std::memory_order_acquire) != 0)
            io_->wait_any(500);
    }
    return !bad.load();
}

}  // namespace strata::ds4

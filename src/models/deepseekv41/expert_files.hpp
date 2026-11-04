// DeepSeek V4.1's routed experts read from GGUF or EXL3 files without the
// OS file cache, for strata-ds41's RAM page pool (docs/MEMORY_MANAGER_PLAN.md, step D5), which reads through it the
// experts it does not hold.
#pragma once

#include "ds4_model.hpp"
#include "ds4_pool.hpp"
#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::ds4 {

/// The routed experts of a GGUF or EXL3 pack as an ExpertSource that copies each into the blob layout the model reads
/// (ExpertFileLayout): its pieces read without the file cache into sector-aligned staging buffers and moved into place.  Copies only: blob() has no memory to give.
class ExpertFiles final : public strata::core::ExpertSource {
public:
    ExpertFiles() = default;
    ~ExpertFiles() override;
    ExpertFiles(const ExpertFiles&) = delete;
    ExpertFiles& operator=(const ExpertFiles&) = delete;

    /// Opens `paths` (Model::get_file_paths) for the experts of `layers` (Model::get_expert_files); false (and `err`)
    /// when a file cannot be opened.
    bool open(const std::vector<std::string>& paths, std::vector<ExpertFileLayout> layers, std::string& err);

    const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    /// Safe from several threads: one batch is read at a time.
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;

private:
    std::vector<platform::fd_t> direct_, buffered_;   ///< per file: unbuffered, and buffered (when unbuffered fails)
    std::vector<ExpertFileLayout> layers_;
    std::unique_ptr<IoQueue> io_;
    std::mutex mu_;
    uint8_t* staging_ = nullptr;                      ///< kBatch experts' pieces, each in a span of span_ bytes
    size_t span_ = 0;
};

}  // namespace strata::ds4

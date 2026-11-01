// src/common/state/pool_copy.cpp - a conversation's pages between page pools and files (pool_copy.hpp).
#include "strata/common/state/pool_copy.hpp"

#include "strata/base/gpu/runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {

/// The file's first bytes; then the region count, and per region its block bytes and block count (uint64_t each).
constexpr char kMagic[8] = {'S', 'T', 'R', 'P', 'A', 'G', 'E', '1'};
/// The host buffer the bytes of a GPU's pages pass through on their way to and from a file.
constexpr size_t kBounceBytes = 8u << 20;

using File = std::unique_ptr<std::FILE, int (*)(std::FILE*)>;

/// Gets a conversation's block count per region.
std::vector<uint64_t> count_blocks(const PagePool& pool, ConversationId id) {
    std::vector<uint64_t> counts;
    for (size_t r = 0; r < pool.layout().regions.size(); ++r)
        counts.push_back(pool.conversation_blocks(id, r).size());
    return counts;
}

/// Copies `bytes` of pages (in any memory) to the file `f` (`to_file`) or from it, through `bounce` a piece at a time.
bool copy_through(std::FILE* f, uint8_t* pages, uint64_t bytes, bool to_file, std::vector<uint8_t>& bounce) {
    for (uint64_t at = 0; at < bytes; at += bounce.size()) {
        const size_t n = (size_t) std::min<uint64_t>(bounce.size(), bytes - at);
        if (to_file ? !gpu_copy(bounce.data(), pages + at, n) || std::fwrite(bounce.data(), 1, n, f) != n
                    : std::fread(bounce.data(), 1, n, f) != n || !gpu_copy(pages + at, bounce.data(), n))
            return false;
    }
    return true;
}

}  // namespace

bool copy_conversation(const PagePool& from, ConversationId from_id, const PagePool& to, ConversationId to_id,
                       std::string& err) {
    const size_t regions = to.layout().regions.size();
    if (from.layout().regions.size() != regions || from.page_bytes() != to.page_bytes()) {
        err = "the two page pools differ in layout or page size";
        return false;
    }
    for (size_t r = 0; r < regions; ++r) {
        const std::vector<PageId>& src = from.conversation_blocks(from_id, r);
        const std::vector<PageId>& dst = to.conversation_blocks(to_id, r);
        if (src.size() < dst.size()) {
            err = "the conversation's copy has fewer blocks than its new place";
            return false;
        }
        for (size_t b = 0; b < dst.size(); ++b)
            if (!gpu_copy(to.page_address(dst[b]), from.page_address(src[b]), to.block_bytes(r))) {
                err = "a copy of the conversation's pages failed";
                return false;
            }
    }
    return true;
}

bool save_conversation(const PagePool& pool, ConversationId id, const std::string& path, std::string& err) {
    File f(std::fopen(path.c_str(), "wb"), &std::fclose);
    const std::vector<uint64_t> counts = count_blocks(pool, id);
    std::vector<uint64_t> head{counts.size()};
    for (size_t r = 0; r < counts.size(); ++r)
        head.insert(head.end(), {pool.block_bytes(r), counts[r]});
    std::vector<uint8_t> bounce(kBounceBytes);
    bool ok = f != nullptr && std::fwrite(kMagic, 1, sizeof kMagic, f.get()) == sizeof kMagic &&
              std::fwrite(head.data(), sizeof(uint64_t), head.size(), f.get()) == head.size();
    for (size_t r = 0; ok && r < counts.size(); ++r)
        for (const PageId p : pool.conversation_blocks(id, r))
            ok = ok && copy_through(f.get(), pool.page_address(p), pool.block_bytes(r), true, bounce);
#if defined(_WIN32)
    ok = ok && std::fflush(f.get()) == 0 && _commit(_fileno(f.get())) == 0;
#else
    ok = ok && std::fflush(f.get()) == 0 && fsync(fileno(f.get())) == 0;
#endif
    if (f != nullptr)
        ok = std::fclose(f.release()) == 0 && ok;
    if (ok)
        return true;
    std::remove(path.c_str());
    err = "could not write the paused conversation to " + path;
    return false;
}

bool load_conversation(const std::string& path, const PagePool& pool, ConversationId id, std::string& err) {
    File f(std::fopen(path.c_str(), "rb"), &std::fclose);
    const std::vector<uint64_t> counts = count_blocks(pool, id);
    char magic[sizeof kMagic] = {};
    std::vector<uint64_t> head(1 + 2 * counts.size());
    bool ok = f != nullptr && std::fread(magic, 1, sizeof magic, f.get()) == sizeof magic &&
              std::memcmp(magic, kMagic, sizeof magic) == 0 &&
              std::fread(head.data(), sizeof(uint64_t), head.size(), f.get()) == head.size() &&
              head[0] == counts.size();
    std::vector<uint8_t> bounce(kBounceBytes);
    for (size_t r = 0; ok && r < counts.size(); ++r) {
        const uint64_t block = head[1 + 2 * r], stored = head[2 + 2 * r];
        const std::vector<PageId>& blocks = pool.conversation_blocks(id, r);
        ok = block == pool.block_bytes(r) && stored >= blocks.size();
        for (size_t b = 0; ok && b < blocks.size(); ++b)
            ok = copy_through(f.get(), pool.page_address(blocks[b]), block, false, bounce);
        // past the file's blocks the conversation does not hold (cells it had for guesses when it was saved)
        const int64_t skip = ok ? (int64_t) ((stored - blocks.size()) * block) : 0;
#if defined(_WIN32)
        ok = ok && _fseeki64(f.get(), skip, SEEK_CUR) == 0;
#else
        ok = ok && fseeko(f.get(), (off_t) skip, SEEK_CUR) == 0;
#endif
    }
    if (!ok)
        err = "could not read the paused conversation from " + path;
    return ok;
}

}  // namespace strata::core

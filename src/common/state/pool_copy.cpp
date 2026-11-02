// src/common/state/pool_copy.cpp - a conversation's pages between page pools and files (pool_copy.hpp).
#include "strata/common/state/pool_copy.hpp"

#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/os.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {

/// The start of every pause file's name (get_pause_file_prefix).
constexpr char kPausePrefix[] = "paused-";

/// The file's first bytes; then the region count, and per region its block bytes and block count (uint64_t each).
constexpr char kMagic[8] = {'S', 'T', 'R', 'P', 'A', 'G', 'E', '1'};
/// The host buffer the bytes of a GPU's pages pass through on their way to and from a file.
constexpr size_t kBounceBytes = 8u << 20;

using File = std::unique_ptr<std::FILE, int (*)(std::FILE*)>;

/// One region of a span, or the share of a fixed region a span takes: the bytes of each of its blocks from `offset`.
struct Piece {
    size_t region = 0;
    uint64_t offset = 0, bytes = 0;
};

/// Gets the validated pieces of `pool` selected by the legacy fixed share, explicit ranges and consecutive regions.
bool get_pieces(const PagePool& pool, const RegionSpan& span, std::vector<Piece>& pieces, std::string& err) {
    if (span.part != SIZE_MAX)
        pieces.push_back({span.part, span.offset, span.bytes});
    for (const RegionRange& range : span.ranges) {
        if (range.region >= pool.layout().regions.size()) {
            err = "a conversation span names a region outside its layout";
            return false;
        }
        const uint64_t block = pool.block_bytes(range.region);
        pieces.push_back({range.region, range.offset,
                          range.bytes ? range.bytes : block - std::min(range.offset, block)});
    }
    for (size_t r = span.first; r < pool.layout().regions.size() && r - span.first < span.count; ++r)
        pieces.push_back({r, 0, pool.block_bytes(r)});
    for (const Piece& piece : pieces) {
        if (piece.region >= pool.layout().regions.size() || piece.offset > pool.block_bytes(piece.region) ||
            piece.bytes > pool.block_bytes(piece.region) - piece.offset) {
            err = "a conversation span extends beyond a region's block";
            return false;
        }
    }
    return true;
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
                       std::string& err, RegionSpan from_span, RegionSpan to_span) {
    std::vector<Piece> src_pieces, dst_pieces;
    if (!get_pieces(from, from_span, src_pieces, err) || !get_pieces(to, to_span, dst_pieces, err))
        return false;
    bool same = src_pieces.size() == dst_pieces.size() && from.page_bytes() == to.page_bytes();
    for (size_t i = 0; same && i < dst_pieces.size(); ++i)
        same = src_pieces[i].bytes == dst_pieces[i].bytes;
    if (!same) {
        err = "the two page pools differ in layout or page size";
        return false;
    }
    for (size_t i = 0; i < dst_pieces.size(); ++i) {
        const Piece& s = src_pieces[i];
        const Piece& d = dst_pieces[i];
        const std::vector<PageId>& src = from.conversation_blocks(from_id, s.region);
        const std::vector<PageId>& dst = to.conversation_blocks(to_id, d.region);
        if (src.size() < dst.size()) {
            err = "the conversation's copy has fewer blocks than its new place";
            return false;
        }
        for (size_t b = 0; b < dst.size(); ++b)
            if (!gpu_copy(to.page_address(dst[b]) + d.offset, from.page_address(src[b]) + s.offset, d.bytes)) {
                err = "a copy of the conversation's pages failed";
                return false;
            }
    }
    return true;
}

bool save_conversation(const PagePool& pool, ConversationId id, const std::string& path, std::string& err,
                       RegionSpan span) {
    std::vector<Piece> pieces;
    if (!get_pieces(pool, span, pieces, err))
        return false;
    File f(std::fopen(path.c_str(), "wb"), &std::fclose);
    std::vector<uint64_t> head{pieces.size()};
    for (const Piece& c : pieces)
        head.insert(head.end(), {c.bytes, pool.conversation_blocks(id, c.region).size()});
    std::vector<uint8_t> bounce(kBounceBytes);
    bool ok = f != nullptr && std::fwrite(kMagic, 1, sizeof kMagic, f.get()) == sizeof kMagic &&
              std::fwrite(head.data(), sizeof(uint64_t), head.size(), f.get()) == head.size();
    for (const Piece& c : pieces)
        for (const PageId p : pool.conversation_blocks(id, c.region))
            ok = ok && copy_through(f.get(), pool.page_address(p) + c.offset, c.bytes, true, bounce);
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

bool load_conversation(const std::string& path, const PagePool& pool, ConversationId id, std::string& err,
                       RegionSpan span) {
    std::vector<Piece> pieces;
    if (!get_pieces(pool, span, pieces, err))
        return false;
    File f(std::fopen(path.c_str(), "rb"), &std::fclose);
    char magic[sizeof kMagic] = {};
    std::vector<uint64_t> head(1 + 2 * pieces.size());
    bool ok = f != nullptr && std::fread(magic, 1, sizeof magic, f.get()) == sizeof magic &&
              std::memcmp(magic, kMagic, sizeof magic) == 0 &&
              std::fread(head.data(), sizeof(uint64_t), head.size(), f.get()) == head.size() &&
              head[0] == pieces.size();
    std::vector<uint8_t> bounce(kBounceBytes);
    for (size_t i = 0; ok && i < pieces.size(); ++i) {
        const uint64_t block = head[1 + 2 * i], stored = head[2 + 2 * i];
        const std::vector<PageId>& blocks = pool.conversation_blocks(id, pieces[i].region);
        ok = block == pieces[i].bytes && stored >= blocks.size();
        for (size_t b = 0; ok && b < blocks.size(); ++b)
            ok = copy_through(f.get(), pool.page_address(blocks[b]) + pieces[i].offset, block, false, bounce);
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

std::string get_pause_file_prefix(const std::string& dir) {
    const auto start = std::chrono::system_clock::now().time_since_epoch().count();
    return (std::filesystem::path(dir) / (kPausePrefix + std::to_string(strata::platform::get_process_id()) + "-" +
                                          std::to_string(start) + "-"))
        .string();
}

int remove_stale_pause_files(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    int removed = 0;
    for (const fs::directory_entry& f : fs::directory_iterator(dir, ec)) {
        const std::string name = f.path().filename().string();
        if (name.rfind(kPausePrefix, 0) != 0 || f.path().extension() != ".bin")
            continue;
        const uint64_t pid = std::strtoull(name.c_str() + sizeof kPausePrefix - 1, nullptr, 10);
        if (!strata::platform::is_process_running(pid) && fs::remove(f.path(), ec))
            ++removed;
    }
    return removed;
}

}  // namespace strata::core

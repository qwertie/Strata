// DeepSeek's routed experts read without the file cache (expert_files.hpp) from a file of known bytes: matrices at
// offsets and of sizes that are not sector multiples, several layers and files, batches larger than the reader's,
// each expert's bytes landing at its blob offsets and nothing else of the blob touched.  No GPU, no model files.
#include "expert_files.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::ds4;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

/// Gets the byte a test file holds at `at` (a different pattern per file).
uint8_t get_byte(int file, uint64_t at) {
    return (uint8_t) ((at * 2654435761u >> 13) + (uint64_t) file * 77);
}

/// Writes `bytes` of file `file`'s pattern to `path`.
void write_file(const std::string& path, int file, uint64_t bytes) {
    std::vector<uint8_t> data(bytes);
    for (uint64_t i = 0; i < bytes; ++i)
        data[i] = get_byte(file, i);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    check(f != nullptr && std::fwrite(data.data(), 1, data.size(), f) == data.size(), "the test file is written");
    std::fclose(f);
}

}  // namespace

int main() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ds4_expert_files_test";
    std::filesystem::create_directories(dir);
    const std::vector<std::string> paths = {(dir / "a.bin").string(), (dir / "b.bin").string()};
    write_file(paths[0], 0, 3u << 20);
    write_file(paths[1], 1, 2u << 20);
    // layer 0 in file a, layer 1's down matrix in file b; offsets, strides and sizes off the 4096 grid
    std::vector<ExpertFileLayout> layers(2);
    layers[0] = {5, {0, 0, 0}, {1000, 300001, 600003}, {50001, 50003, 30007}, {40961, 40963, 30001}, {0, 41216, 82432},
                 112640};
    layers[1] = {5, {0, 0, 1}, {1500000, 1800005, 7}, {40000, 40000, 20000}, {33333, 33335, 17777}, {256, 33792, 67584},
                 85504};
    {   // (the readers close the files before they are removed)
        ExpertFiles files;
        std::string err;
        check(files.open(paths, layers, err), "the files open");
        const uint8_t fill = 0xA5;
        std::vector<std::vector<uint8_t>> blobs;
        std::vector<int32_t> ls, es;
        std::vector<uint8_t*> dst;
        for (int32_t l = 0; l < 2; ++l)
            for (int32_t e = 0; e < 5; ++e) {
                blobs.emplace_back(layers[(size_t) l].blob, fill);
                ls.push_back(l);
                es.push_back(e);
            }
        for (std::vector<uint8_t>& b : blobs)
            dst.push_back(b.data());
        check(files.copy_blobs(ls.data(), es.data(), dst.data(), dst.size()), "10 experts are read in batches");
        std::vector<uint8_t> one(layers[1].blob, fill);
        check(files.copy_blob(1, 3, one.data()) && one == blobs[8], "one expert reads as in a batch");
        for (size_t i = 0; i < blobs.size(); ++i) {
            const ExpertFileLayout& l = layers[(size_t) ls[i]];
            bool same = true, untouched = true;
            std::vector<char> inside(l.blob, 0);
            for (int r = 0; r < 3; ++r)
                for (uint64_t k = 0; k < l.bytes[r]; ++k) {
                    inside[l.tr[r] + k] = 1;
                    const uint64_t at = l.off[r] + (uint64_t) es[i] * l.stride[r] + k;
                    same = same && blobs[i][l.tr[r] + k] == get_byte(l.file[r], at);
                }
            for (uint64_t k = 0; k < l.blob; ++k)
                untouched = untouched && (inside[k] != 0 || blobs[i][k] == fill);
            check(same, "each matrix holds the file's bytes at its offset");
            check(untouched, "the bytes between the matrices stay as they were");
        }
        ExpertFiles missing;
        check(!missing.open({(dir / "none.bin").string()}, layers, err) && !err.empty(), "a missing file is refused");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::printf("ds4_expert_files_test: %d checks passed\n", checks);
    return 0;
}

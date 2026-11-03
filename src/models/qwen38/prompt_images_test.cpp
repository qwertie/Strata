#include "strata/models/qwen38/prompt_images.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

/// Checks grid positions, image identity and malformed records without a model or GPU.
int main() {
    const auto path = std::filesystem::temp_directory_path() / "strata_prompt_images_test.sve";
    const std::vector<int64_t> ids = {10, 248056, 248056, 248056, 248056, 11};
    const int32_t header[] = {0x31455653, 4, 2, 2, 2};
    const float pixels[] = {1, 2, 3, 4, 5, 6, 7, 8};
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(header), sizeof header);
        out.write(reinterpret_cast<const char*>(pixels), sizeof pixels);
    }
    std::vector<int32_t> positions(30);
    std::vector<float> rows;
    std::vector<const float*> pointers;
    std::vector<strata::core::ConversationImageKey> hashes;
    std::string err;
    const auto read = [&](const std::vector<uint64_t>& keys) {
        err.clear();
        return strata::qwen38::read_prompt_images(path.string(), ids, 2, keys, positions, rows, pointers,
                                                  hashes, err);
    };
    bool ok = read({});
    ok = ok && hashes.size() == 1 && hashes[0].start == 1 && pointers[0] == nullptr && pointers[1][0] == 1 &&
         pointers[4][1] == 8 && pointers[5] == nullptr;
    ok = ok && positions[3] == 1 && positions[6] == 1 && positions[7] == 1 && positions[8] == 2 &&
         positions[12] == 1 && positions[13] == 2 && positions[14] == 2 && positions[15] == 3 && positions[27] == 7;
    const uint64_t first = hashes[0].hash;
    ok = ok && read({123}) && hashes[0].hash != first;
    const uint64_t keyed = hashes[0].hash;
    {
        std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
        const float changed = 9;
        out.seekp(sizeof header);
        out.write(reinterpret_cast<const char*>(&changed), sizeof changed);
    }
    ok = ok && read({}) && hashes[0].hash != first && read({123}) && hashes[0].hash == keyed;
    std::filesystem::resize_file(path, sizeof header + 1);
    ok = ok && !read({}) && err == "short embeddings file";
    std::filesystem::remove(path);
    std::cout << (ok ? "prompt_images_test: passed\n" : "prompt_images_test: FAILED\n");
    return ok ? 0 : 1;
}

// The drafts' confidence cut that strata-dsv41's rounds and strata-ds41's proposals both make
// (Model::count_confident): the floor on each draft's confidence, the cut on their product, the most asked for.  No
// GPU, no model files.
#include "ds4_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::ds4::Model;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

}  // namespace

int main() {
    const std::vector<float> conf = {0.99f, 0.9f, 0.8f, 0.6f, 0.95f};
    check(Model::count_confident(conf, 5, 0.65f, 0.0f) == 3, "the floor stops at the first draft below it");
    check(Model::count_confident(conf, 2, 0.65f, 0.0f) == 2, "no more than the most asked for");
    check(Model::count_confident(conf, 0, 0.65f, 0.0f) == 0, "none asked for");
    check(Model::count_confident(conf, 5, 0.0f, 0.75f) == 2, "the product cut: 0.99 x 0.9 x 0.8 is below 0.75");
    check(Model::count_confident(conf, 5, 0.0f, 0.0f) == 5, "no cut: every draft");
    check(Model::count_confident({0.5f}, 5, 0.65f, 0.0f) == 0, "a first draft below the floor: none");
    check(Model::count_confident({}, 5, 0.0f, 0.0f) == 0, "no drafts");
    std::printf("draft_cut_test: %d checks passed\n", checks);
    return 0;
}

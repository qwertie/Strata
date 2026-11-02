// include/strata/models/deepseekv41/paged_model.hpp - strata-ds41: DeepSeek V4.1 on the shared paged engine
// (strata/common/serve/paged_engine.hpp; docs/MEMORY_MANAGER_PLAN.md, step D4).  One conversation on one GPU, without
// the DSpark drafter: its window rings, partial groups, compressed rows and index keys are pages of the GPU's page
// pool, which grow with it; the routed experts are placed as the old engine (strata-dsv41) places them.
#pragma once

namespace strata::ds4 {

/// Runs strata-ds41: the paged engine's options (strata::serve::take_paged_options), then strata-dsv41's model
/// options (read_option), `--eos LIST` and `--serve` (implied).  Gets the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::ds4

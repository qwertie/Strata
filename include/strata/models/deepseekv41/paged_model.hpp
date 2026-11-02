// include/strata/models/deepseekv41/paged_model.hpp - strata-ds41: DeepSeek V4.1 on the shared paged engine
// (strata/common/serve/paged_engine.hpp; docs/MEMORY_MANAGER_PLAN.md, steps D4 to D6).  One GPU, without the DSpark
// drafter: each conversation's window rings, partial groups, compressed rows and index keys are pages of the GPU's
// page pool, which grow with it, and the routed experts take the rest of the GPU's pool and the RAM's (D5).  It holds
// the conversations of its --batch slots and decodes one at a time; they pause to RAM and to the SSD with their
// checkpoints, and a prompt that leaves a conversation before its end goes back to one of them (D6).
#pragma once

namespace strata::ds4 {

/// Runs strata-ds41: the paged engine's options (strata::serve::take_paged_options), then strata-dsv41's model
/// options (read_option), its own (--batch, --expert-cache, --adapt-*), `--eos LIST` and `--serve` (implied).  Gets
/// the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::ds4

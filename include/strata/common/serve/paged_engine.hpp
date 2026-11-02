// include/strata/common/serve/paged_engine.hpp - the engine whose conversations live in page pools, several decoding
// at once, for any model that implements PagedModel (paged_model.hpp; docs/MEMORY_MANAGER_PLAN.md, steps M2d to M9
// and D3).  strata-q38 runs it for Qwen3.8 (include/strata/models/qwen38/paged_model.hpp).
//
// It answers the server's line protocol (`--serve` is implied) like `strata --serve --batch N`: GEN, BGEN, STOP,
// BSTOP, QUIT, LIST_CONVERSATIONS, GET_CONTEXT_WINDOWS, GET_MEMORY_USE and CHANGE_SETTINGS (BYIELD is dropped: a
// prompt read never gives way).  Each conversation's state takes page-pool pages as it grows; the experts are in the
// same pool (ExpertPages): a growing conversation takes the least-used experts' pages, and between requests the
// experts take back the pages conversations left.  Up to `concurrent` conversations decode together, each step one
// pass through the model with a row per conversation and the rows left over shared among their guesses
// (strata::spec::Step); a prompt is read in passes of its own, a chunk at a time between the others' steps.  A request
// continues the conversation that holds the longest start of its prompt, reading only the rest.  A request's
// `priority=N` key (default 0) orders the conversations of a step, higher first.
//
// The RAM is a second page pool (RamExperts): the experts the GPU's pool does not hold, then copies of those it does
// while room allows, and the paused conversations.  The experts in neither are read from the SSD without the file
// cache.  A conversation that leaves its place on the GPU (another request takes the place, or a growing conversation
// needs its pages) is paused, its pages copied to the RAM's pool or to a file on the SSD, and comes back when a
// request continues it.
//
// On a layer split each card runs a range of layers and has a page pool of its own, with the experts of its layers
// and each conversation's state for them.  A paused conversation is one conversation in the RAM's pool (each card's
// regions one after another) or one file per card.  A card with pages to spare holds experts of other cards' layers
// that their cards have no room for, and computes them in one hop (`remote_experts`; auto: where a hop over the card's
// link costs less than the CPU computing an expert).  When a card's memory setting changes, the cut moves if the
// planner says so: between two steps every conversation is paused to RAM, the cards take their new layers, and the
// conversations come back.
//
// CHANGE_SETTINGS (memory_report.hpp) changes, between two steps, each pool's size (`memory_mib.gpu<N>`, per card
// `memory_mib.gpu<N>.<card>` for a later card on the GPU of an earlier one,
// `memory_mib.ram`: a pool takes and gives back memory in chunks of 256 MiB; a shrink evicts the experts of the
// chunks it gives back and moves or pauses the conversations there), `tokens_min`, `tokens_max`, `ssd_pause_mib`,
// `drafter` (off gives its weights back), `concurrent`, `rows` and `lookahead`.  A shrink is refused only when the
// biggest decoding conversation (at the end of its request) or the token minimum would not fit.
#pragma once

#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/serve/paged_model.hpp"
#include "strata/common/spec/step.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::serve {

/// The conversation tokens whose pages experts leave free when --tokens-min is not given.
constexpr int64_t kDefaultTokensMin = 8192;
/// The SSD space paused conversations may take when --ssd-pause-mib is not given.
constexpr int64_t kDefaultSsdPauseMib = 8192;
/// The most evicted experts put back in one step of the serve loop, so that a request waits little for it.
constexpr int kRefillBatch = 16;
/// The time the experts put back into the GPU's pool in one step of the serve loop may take on its link: a slow link
/// puts back fewer (get_refill_batch).
constexpr double kRefillBudgetMs = 10.0;
/// The step in which the pools take memory and give it back (PagePoolConfig::chunk_bytes).  An expert may not cross a
/// chunk's end, so each chunk ends in a run too short for any expert: 0.36% of the Strix Halo's GPU pool in chunks of
/// this size (docs/MEMORY_MANAGER_PLAN.md, M2f).
constexpr uint64_t kChunkBytes = 256ull << 20;

/// The paged engine's own options.
struct PagedSettings {
    int64_t tokens_min = kDefaultTokensMin;        ///< --tokens-min
    int64_t tokens_max = -1;                       ///< --tokens-max; -1: --max-context
    /// --gpu-mib, per card of a layer split (one value: every card's); empty: what each GPU has free less what it keeps
    std::vector<int64_t> gpu_mib;
    int64_t ram_mib = -1;                          ///< --ram-mib; -1: get_ram_room
    int64_t ssd_pause_mib = kDefaultSsdPauseMib;   ///< --ssd-pause-mib
    std::string pause_dir;                         ///< --pause-dir; empty: the temporary folder's <program name>
    std::string copy_back = "auto";                ///< --copy-back
    std::string remote_experts = "auto";           ///< --remote-experts
    bool drafter = true;                           ///< --drafter on|off: its weights loaded (with --lookahead)
    strata::spec::StepSettings step;               ///< --rows, --lookahead and --concurrent (default --batch)
};

/// Runs the engine over `model`, whose start made `cards` (the GPUs' pools) and `ram` (the RAM's pool: experts and
/// paused conversations, its conversation layout the whole model's regions): carves each conversation's state on each
/// card, then answers the server's commands until QUIT or the end of stdin.  `copy_back`: an expert a GPU's pool
/// evicts is copied to `ram`.  `slots`: the batch slots (--batch; 0: one conversation at a time).  Gets the exit
/// status.
int serve_paged_model(PagedModel& model, PagedCards& cards, strata::core::RamExperts& ram, bool copy_back,
                      const PagedSettings& settings, int slots);

/// Gets why the engine cannot decode with `s` on a model whose steps run at most `max_rows` rows (empty: it can).
std::string get_step_settings_problem(const strata::spec::StepSettings& s, int max_rows);
/// Gets the memory a pool on device `dev` may take besides what it holds: what the GPU has free less `keep`.  On
/// Windows, for a GPU whose memory is the system RAM, what is free is what is free of the share the BIOS gives it
/// (get_share_room, `mapped` the RAM's page-locked pool), not the runtime's free memory: that is the share and what
/// the GPU may borrow beyond it, and HIP takes page-locked host memory off it twice (on the Strix Halo, 8 GiB of
/// hipHostMalloc took 16 GiB off it).  Its memory counts in Windows' commit charge, so no more than get_commit_room
/// either.  Cards on one GPU each see what is free of it.
uint64_t get_gpu_room(int dev, uint64_t keep, uint64_t mapped);
/// Gets the RAM a pool may take by default besides what the pools hold: what is available less `headroom`, and no more
/// than get_commit_room.
uint64_t get_ram_room(uint64_t headroom);

}  // namespace strata::serve

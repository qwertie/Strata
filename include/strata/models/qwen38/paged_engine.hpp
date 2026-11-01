// include/strata/models/qwen38/paged_engine.hpp - strata-q38: the Qwen3.8 engine whose conversations live in a page
// pool (docs/MEMORY_MANAGER_PLAN.md, steps M2d, M2e, M3, M4 and M5).
//
// It answers the server's line protocol (`--serve` is implied) like `strata --serve --batch N`: GEN, BGEN, STOP,
// BSTOP, QUIT, LIST_CONVERSATIONS, GET_CONTEXT_WINDOWS and GET_MEMORY_USE (BYIELD is dropped: a prompt read never
// gives way).  One GPU.  Each conversation's GDN state and K/V (and the MTP drafter's K/V) take page-pool pages as it
// grows; the experts are in the same pool (ExpertPages): a growing conversation takes the least-used experts' pages,
// and between requests the experts take back the pages conversations left.  Up to --batch conversations decode
// together, each step one pass through the model with a row per conversation and the rows left over shared among
// their guesses (strata::spec::Step); a prompt is read in passes of its own, a chunk at a time between the others'
// steps.  A request continues the conversation that holds the longest start of its prompt, reading only the rest.  A
// request's `priority=N` key (default 0) orders the conversations of a step, higher first.
//
// The RAM is a second page pool (RamExperts): the experts the GPU's pool does not hold, then copies of those it does
// while room allows, and the paused conversations.  The experts in neither are read from the SSD without the file
// cache.  A conversation that leaves its place on the GPU (another request takes the place, or a growing conversation
// needs its pages) is paused, its pages copied to the RAM's pool or to a file on the SSD, and comes back when a
// request continues it.
#pragma once

namespace strata::qwen38 {

/// Runs strata-q38 with `strata`'s command line, plus `--tokens-min N`: the conversation tokens whose pages the
/// experts leave free (default 8192); `--rows N`: the most rows one step computes (default 8, at least --batch);
/// `--lookahead N`: the most guesses a conversation adds to a step (0, the default: no drafting; more needs --mtp,
/// whose drafter stops guessing below --spec-min-p).  `--batch N` is how many conversations decode at once.
/// `--expert-cache N` is the most experts the GPU holds, as many as N slots of the largest expert hold; `auto`, as
/// many as fit.  The RAM: `--ram-mib N`, the RAM's pool (default: what the RAM and the commit limit have free, less
/// the resident headroom, but no more than every expert takes; `--resident-budget-gib N` stands for it);
/// `--tokens-max N`, the tokens the conversations paused in RAM may hold (as many pages as one conversation of N
/// takes; default --max-context); `--ssd-pause-mib N`, the SSD space paused conversations may take (default 8192);
/// `--pause-dir DIR`, where their files go (default the temporary folder's strata-q38); `--copy-back auto|on|off`,
/// whether an expert the GPU evicts is copied back to RAM (auto: when that measures at least twice as fast as
/// reading the SSD).  `--mmap-experts`, `--resident-experts` and `--resident-cpu-experts` are taken and do
/// nothing: the RAM's pool replaces them.  Returns the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::qwen38

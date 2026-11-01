// include/strata/models/qwen38/paged_engine.hpp - strata-q38: the Qwen3.8 engine whose conversations live in a page
// pool (docs/MEMORY_MANAGER_PLAN.md, steps M2d, M2e and M3).
//
// It answers the server's line protocol (`--serve` is implied) like `strata --serve --batch N` with drafting off: GEN,
// BGEN, STOP, BSTOP, QUIT, LIST_CONVERSATIONS, GET_CONTEXT_WINDOWS and GET_MEMORY_USE (BYIELD is dropped: a prompt
// read never gives way).  One GPU, no drafting yet.  Each conversation's GDN state and K/V take page-pool pages as it
// grows; the experts are in the same pool (ExpertPages): a growing conversation takes the least-used experts' pages,
// and between requests the experts take back the pages conversations left.  Up to --batch conversations decode
// together, each step one pass through the model with a row per conversation (strata::spec::Step); a prompt is read
// in passes of its own, a chunk at a time between the others' steps.  A request continues the idle conversation that
// holds the longest start of its prompt, reading only the rest.
#pragma once

namespace strata::qwen38 {

/// Runs strata-q38 with `strata`'s command line, plus `--tokens-min N`: the conversation tokens whose pages the
/// experts leave free (default 8192); `--rows N`: the most rows one step computes (default 8, at least --batch);
/// `--lookahead N`: the most guesses a conversation adds to a step (0, the default: no drafting; there is no drafter
/// yet).  `--batch N` is how many conversations decode at once.  `--expert-cache N` is the most experts the GPU
/// holds, as many as N slots of the largest expert hold; `auto`, as many as fit.  Returns the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::qwen38

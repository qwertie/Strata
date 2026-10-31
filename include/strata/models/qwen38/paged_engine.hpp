// include/strata/models/qwen38/paged_engine.hpp - strata-q38: the Qwen3.8 engine whose conversation lives in a page
// pool (docs/MEMORY_MANAGER_PLAN.md, steps M2d and M2e).
//
// It answers the server's line protocol (`--serve` is implied) like `strata --serve` with drafting off: GEN, STOP,
// QUIT, LIST_CONVERSATIONS, GET_CONTEXT_WINDOWS and GET_MEMORY_USE.  One GPU, one conversation at a time, no drafting
// and no batch slots; the conversation's GDN state and K/V take page-pool pages as it grows, and a new conversation
// replaces the old one (a prompt that continues it reads only its new tokens).  The experts are in the same pool
// (ExpertPages): a growing conversation takes the least-used experts' pages, and between requests the experts take
// back the pages a shorter conversation left.
#pragma once

namespace strata::qwen38 {

/// Runs strata-q38 with `strata`'s command line, plus `--tokens-min N`: the conversation tokens whose pages the
/// experts leave free (default 8192).  `--expert-cache N` is the most experts the GPU holds, as many as N slots of the
/// largest expert hold; `auto`, as many as fit.  Returns the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::qwen38

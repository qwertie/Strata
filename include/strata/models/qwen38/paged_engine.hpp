// include/strata/models/qwen38/paged_engine.hpp - strata-q38: the Qwen3.8 engine whose conversation lives in a page
// pool (docs/MEMORY_MANAGER_PLAN.md, step M2d).
//
// It answers the server's line protocol (`--serve` is implied) like `strata --serve` with drafting off: GEN, STOP,
// QUIT, LIST_CONVERSATIONS, GET_CONTEXT_WINDOWS and GET_MEMORY_USE.  One GPU, one conversation at a time, no drafting
// and no batch slots; the conversation's GDN state and K/V take page-pool pages as it grows, and a new conversation
// replaces the old one (a prompt that continues it reads only its new tokens).  The experts stay in the expert cache.
#pragma once

namespace strata::qwen38 {

/// Runs strata-q38 with `strata`'s command line, plus `--conversation-mib N`: the page pool's size (default: the
/// memory left after the model and the prompt path, at most what a conversation at --max-context takes; an `auto`
/// expert cache leaves only --vram-reserve-mib, so a discrete card needs the option).  Returns the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::qwen38

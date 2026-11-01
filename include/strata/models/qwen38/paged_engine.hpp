// include/strata/models/qwen38/paged_engine.hpp - strata-q38: the Qwen3.8 engine whose conversations live in a page
// pool (docs/MEMORY_MANAGER_PLAN.md, steps M2d, M2e, M3, M4, M5 and M6).
//
// It answers the server's line protocol (`--serve` is implied) like `strata --serve --batch N`: GEN, BGEN, STOP,
// BSTOP, QUIT, LIST_CONVERSATIONS, GET_CONTEXT_WINDOWS, GET_MEMORY_USE and CHANGE_SETTINGS (BYIELD is dropped: a
// prompt read never gives way).  Each conversation's GDN state and K/V (and the MTP drafter's K/V) take
// page-pool pages as it grows; the experts are in the same pool (ExpertPages): a growing conversation takes the
// least-used experts' pages, and between requests the experts take back the pages conversations left.  Up to
// --concurrent conversations decode together, each step one pass through the model with a row per conversation and
// the rows left over shared among their guesses (strata::spec::Step); a prompt is read in passes of its own, a chunk
// at a time between the others' steps.  A request continues the conversation that holds the longest start of its
// prompt, reading only the rest.  A request's `priority=N` key (default 0) orders the conversations of a step, higher
// first.
//
// The RAM is a second page pool (RamExperts): the experts the GPU's pool does not hold, then copies of those it does
// while room allows, and the paused conversations.  The experts in neither are read from the SSD without the file
// cache.  A conversation that leaves its place on the GPU (another request takes the place, or a growing conversation
// needs its pages) is paused, its pages copied to the RAM's pool or to a file on the SSD, and comes back when a
// request continues it.
//
// On a layer split (`--layer-split K[,K2..]|auto`, docs/MEMORY_MANAGER_PLAN.md step M8) each card runs a range of
// layers and has a page pool of its own, with the experts of its layers and each conversation's state for them; the
// verify windows and the prompt path hand their rows from card to card.  `--split-device 0,..` puts several cards on
// one GPU, each with its own weights and pool: the test of a split on a PC with one GPU.  A paused conversation is one
// conversation in the RAM's pool (each card's regions one after another) or one file per card.  A card with pages to
// spare holds experts of other cards' layers that their cards have no room for, and computes them in one hop
// (`--remote-experts auto|on|off`; auto: where a hop over the card's link costs less than the CPU computing an
// expert).  The MTP drafter runs on the last card.  When a card's memory setting changes, the cut moves if the
// planner says so (step M9): between two steps every conversation is paused to RAM, the cards take their new layers,
// and the conversations come back.
//
// CHANGE_SETTINGS (memory_report.hpp) changes, between two steps, each pool's size (`memory_mib.gpu<N>`, per card
// `memory_mib.gpu<N>.<card>` for a later card on the GPU of an earlier one,
// `memory_mib.ram`: a pool takes and gives back memory in chunks of 256 MiB; a shrink evicts the experts of the
// chunks it gives back and moves or pauses the conversations there), `tokens_min`, `tokens_max`, `ssd_pause_mib`,
// `drafter` (off gives its weights back), `concurrent`, `rows` and `lookahead`.  A shrink is refused only when the
// biggest decoding conversation (at the end of its request) or the token minimum would not fit.
#pragma once

namespace strata::qwen38 {

/// Runs strata-q38 with `strata`'s command line, plus `--tokens-min N`: the conversation tokens whose pages the
/// experts leave free (default 8192); `--rows N`: the most rows one step computes (default 8, at least the
/// conversations at once); `--lookahead N`: the most guesses a conversation adds to a step (default 3 with --mtp,
/// whose drafter stops guessing below --spec-min-p, and 0, no drafting, without it); `--drafter on|off`: with a
/// lookahead, whether the drafter starts loaded (off: loaded for the conversations' layout, then given back, until
/// CHANGE_SETTINGS turns it on).  `--batch N` is the batch slots, and `--concurrent N` how many of them decode at once
/// (default all).  `--expert-cache N` is the most experts the GPU holds, as many as N slots of the largest expert
/// hold; `auto`, as many as fit; on a layer split, the profile's first N, each on the card of its layer.  `--gpu-mib
/// N[,N2..]`: the GPU's pool, one per card of a layer split (one value: each card's) (default: what is free less
/// --vram-reserve-mib and the prompt path's buffers, but no more than those experts and a conversation at
/// --max-context take).  The RAM:
/// `--ram-mib N`, the RAM's pool (default: what the RAM has free less the resident headroom, but no more than every
/// expert takes, and like the GPU's default leaving a tenth of Windows' commit limit, at least 8 GiB, uncommitted;
/// `--resident-budget-gib N` stands for it);
/// `--tokens-max N`, the tokens the conversations paused in RAM may hold (as many pages as one conversation of N
/// takes; default --max-context); `--ssd-pause-mib N`, the SSD space paused conversations may take (default 8192);
/// `--pause-dir DIR`, where their files go (default the temporary folder's strata-q38); `--copy-back auto|on|off`,
/// whether an expert the GPU evicts is copied back to RAM (auto: when that measures at least twice as fast as
/// reading the SSD).  `--mmap-experts`, `--resident-experts` and `--resident-cpu-experts` are taken and do
/// nothing: the RAM's pool replaces them.  Returns the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::qwen38

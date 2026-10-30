# The memory manager

A plan for a new Qwen3.8 engine whose memory and parallelism can be seen and changed while it runs. It is built on the
shared code of the refactor ([REFACTOR_PLAN.md](REFACTOR_PLAN.md)) and adds to it, so that Qwen3.6 and DeepSeek can
move onto it later. Status: planning (2026-10-30), branch `memory-manager`.

## What the user gets

A panel in the server's web page shows, for each memory, what it holds now, and lets the user move the boundaries
while the model runs:

```
RTX 3080  | Expert cache                 | Conversations | Drafter | Free     |
20 GiB    | 15.51 GiB                    | 2.19 GiB      | 1.0 GiB | 1.30 GiB |
RTX 3060  | Expert cache         | Conversations | Free |
12 GiB    | 11.08 GiB            | 0.82 GiB      | 0.10 |
System RAM | Expert cache                         | Paused conversations: 2 | Free     |
48 GB      | 28.88 GiB                            | 2.00 GiB (55% used)     | 17 GB    |
SSD        | Experts not loaded | Engram    | Paused conversations: 0 |
           | 0 GiB              | 26.55 GiB | 4 GiB (0% used)         |
Conversations at once: 2 of 3 | Rows per step: 8 | Lookahead: 4
```

- Sizes change with the model running. The one limit: the biggest active conversation must still fit.
- Each change is either for this session only or saved as the default for the next start.
- On a PC whose graphics memory is system RAM (Strix Halo), the graphics and RAM bars are one bar with RAM's share
  marked, because the two are the same memory.

## The settings

| Setting | What it limits | Changes live |
|---|---|---|
| Memory used, per GPU and in RAM | the total the engine takes from each memory; it gives memory back or takes more | yes |
| Conversation tokens: minimum and maximum | the K/V of all conversations kept in RAM and on the GPUs, in tokens, in any mix (one conversation of 2,000,000 tokens or 8 of 250,000); the minimum is kept free of experts so a new conversation never waits | yes |
| SSD space for paused conversations | how much of the disk paused conversations may take | yes |
| Drafter on/off (per GPU) | its weights: about 1 GiB for Qwen3.8 | yes (unloading frees the memory) |
| Conversations at once | how many conversations decode in the same step | yes |
| Rows per step | how many tokens one pass through the model computes (at most 8 today) | yes |
| Lookahead | the most guesses one conversation may add in a step; 0 turns drafting off | yes |

Rows per step must be at least the conversations at once.

There is no expert-cache setting: experts get every page the conversations are not using, the most-used on the GPUs,
then in RAM, and the rest are read from the SSD as needed. So a small conversation leaves almost all the memory to
experts, and a large one takes pages from them up to the maximum. There is no separate low-RAM mode either: when the
memory used is too small for all the experts, some stay on the SSD, read without the operating system's file cache
(`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux), so the cache does not take the RAM back. That replaces
today's `--resident-budget-gib`, `--mmap-experts` and the file tier's options.

### How rows are shared (the owner's rule)

Each step gives every active conversation one row for its next token. The rows left over go to guesses, split as
evenly as possible, and no conversation gets more than the lookahead:

| Conversations at once (3), rows (8), active now | Rows for guesses | Split |
|---|---:|---|
| 3 | 5 | 2, 2, 1 |
| 2 | 6 | 3, 3 |
| 1 | 7 | 7, or the lookahead if lower |

- A share is the most a conversation may use: its drafter may stop earlier when it is unsure (the confidence floor of
  today's `--spec-min-p`). Rows left unused stay empty.
- The extra row of an uneven split goes to the highest-priority conversation, and rotates between equals.
- Reading a prompt is not counted in these rows: it reads chunks of hundreds of tokens in passes of its own.
- Later, a share may be spent as a tree of guesses instead of a chain (see [Later](#later)).

Whether guesses touch fewer experts than other conversations' tokens (and so cost less when experts are in RAM) is
measured once this works, by comparing settings such as 1×8 rows against 3×8 and 3×3. No setting is chosen from that
assumption beforehand.

## How it works

### One page pool per memory

Each GPU's memory and the RAM the engine uses are each one pool of equal-size pages (about 2 MiB; a Qwen3.8 IQ3_S
expert is 1.95 MiB). A page holds an expert, or a block of a conversation's tokens, or is free.

- A pool takes its memory in chunks (about 256 MiB). Pages need not be next to each other, so growing the memory used
  adds chunks; shrinking it first empties chunks (their experts are evicted, conversation pages are copied to other
  chunks or paused) and then gives them back. No virtual-memory API is needed, so this works on AMD as well as NVIDIA
  (today's live resize, the `VRAM` command, is NVIDIA-only).
- Conversations and experts trade pages without a setting: a growing conversation takes the least-used expert's
  page, up to the conversation maximum. Experts take pages back only after a conversation ends or is paused, and only
  after a short wait, so that the two do not thrash.
- On the SSD, an expert is read with its sector-aligned surroundings into a staging buffer, because reads without the
  file cache must be aligned. DeepSeek's Engram reader (its own threads reading the SSD) is the model for this.
- A conversation's memory is described with the existing state descriptor
  (`include/strata/common/state/state_layout.hpp`): fixed regions (the GDN state, the drafter's state) take whole
  pages once; per-token regions (K/V) take a page per block of tokens as the conversation grows. The pool holds the
  blocks; a conversation keeps a list of its pages.
- Weights that are always needed (the dense layers, the head) are outside the pools.

### Tiers and pausing

- A conversation is **active** (decoding, its pages on a GPU), **paused in RAM** or **paused on the SSD**. A paused
  conversation is copied page by page; coming back, it is copied up again before its next step.
- Every conversation has a priority (an API field, else the order it arrived). When memory is short, the
  lowest-priority paused conversation moves down a tier first, then the lowest-priority active one is paused. A
  running conversation may be paused for a higher-priority one.
- The SSD tier replaces today's separate mechanisms: Qwen3.8's RAM conversation cache, the session files, and
  DeepSeek's RAM/disk slots.

### Guesses in the page pool

- A guess writes its K/V into its conversation's last page; a rejected guess's cells are simply not counted. No
  separate session is needed for drafting.
- The drafter's own state per conversation is one more region of the conversation.

### The step

The engine's loop is a model-neutral step: pick the active conversations (by priority), share the rows by the rule
above, ask the drafter for each conversation's guesses, run the rows through the model with a parent table (which row
each row follows), accept, and update the pages. The model provides only:

- running a set of rows with their parent table (a chain today, a tree later);
- its state regions (the descriptor);
- its drafter's "propose up to N guesses".

For GDN models the verify kernel keeps one running state per row for rollback, as today; a tree needs each row to
start from its parent's state instead of the previous row's.

## The protocol

New engine commands (between and during requests, like `LIST_CONVERSATIONS`):

- `MEM` reports every pool: per memory, its size and its pages by use (experts, active conversations, paused
  conversations, drafter, free), and the settings above. The old engines answer it too, read-only, from what they know
  (sizes from their startup, `LIST_CONVERSATIONS` bytes), so the panel shows them.
- `SET <name> <value> [...]` changes settings live: `memory_mib.<memory>`, `tokens_min`, `tokens_max`,
  `ssd_pause_mib`, `drafter on|off`, `concurrent`, `rows`, `lookahead`. The engine answers with the new `MEM` or `ERR <why>` (for example
  "the biggest active conversation needs 3.1 GiB").

The server forwards both through one endpoint (`GET/POST /v1/memory`); "save as default" writes the run config
(`strata-<model>.json`), the one file the web page, the app and the Terminal all read.

## Where the code goes

| Part | Folder |
|---|---|
| Page pool, page lists, tiers, priorities | `src/common/state/` |
| Step: picking conversations, sharing rows, accept | `src/common/spec/` and `src/common/serve/` |
| `MEM`/`SET` parsing and answers | `src/common/serve/` |
| Expert pages (evict, load, count use) | `src/common/experts/` |
| Copies between pool, RAM and SSD | `src/base/gpu/` (the runtime interface), `src/base/platform/` (files) |
| Qwen3.8 rows with parent table, its regions, its drafter | `src/models/qwen38/` |
| `/v1/memory`, saving defaults | `strata_server/`, `strata_common/` (run config) |
| The panel | `strata_server/web/` |

## Steps

| Step | Work | Checked by |
|---|---|---|
| M0 | `MEM` in the three existing engines (read-only), `/v1/memory`, the panel without controls | the panel matches the engines' own numbers on Qwen3.8, Qwen3.6, DeepSeek |
| M1 | The page pool and page lists: chunks, growing and shrinking, experts and conversations trading pages, with unit tests (CPU) | unit tests: allocation, growth, emptying a chunk, a full pool, no thrashing |
| M2 | The new Qwen3.8 engine: one conversation in the pool, no drafting, same answers as today's engine | token-identical at temperature 0 against the old engine |
| M3 | Several conversations, the step with rows per step and conversations at once | answers identical per conversation; speed against `--batch` |
| M4 | Drafting with the row-sharing rule and the lookahead | identical answers; speed against today's MTP |
| M5 | Pausing to RAM and SSD, priorities; experts read from the SSD without the file cache | pause and resume give the same answers; files saved before load; with too little memory for the experts, the same answers as with enough, and RAM use stays at the setting |
| M6 | Live `SET`, the panel's controls, "this session" and "save as default" | resize while decoding; the biggest-conversation limit refuses |
| M7 | Measurements: guesses against other conversations' tokens, by where experts live; defaults chosen from them | numbers in DETAILS.md, measured on the Strix Halo and, if possible, the NVIDIA test PC |

M0 is useful on its own and touches no engine logic. M2 to M6 are the new engine; the old Qwen3.8 path stays until
the new one is at least as fast.

## Later

- **Tree drafting**: a conversation's share of rows spent as a tree of guesses (several likely continuations
  checked in one pass). The parent table of the step allows it from the start; it needs the drafter to propose trees,
  the verify kernels to take tree-shaped parents, and for GDN layers a running state per branch.
- Qwen3.6 and DeepSeek on the new step and pools.
- More than 8 rows per step (larger verify windows).

## Open questions

- Page size: one size for experts and conversation blocks (2 MiB), or two pools per memory?
- Should a paused conversation's pages on the SSD use the session file format (`conversation_file.hpp`), so
  "save" and "pause to SSD" are one thing?
- What the panel shows for an old engine's settings it cannot change live: greyed out, or "applies after a restart"
  with a restart button?

# The memory manager

A plan for a new Qwen3.8 engine whose memory and parallelism can be seen and changed while it runs. It is built on the
shared code of the refactor ([REFACTOR_PLAN.md](REFACTOR_PLAN.md)) and adds to it, so that Qwen3.6 and DeepSeek can
move onto it later. Status: M0 to M7 done, M8 in part (2026-11-01), branch
`memory-manager`.

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
- On a PC whose graphics memory is system RAM (Strix Halo), the bars follow the BIOS's division: the GPU's bar is the
  share the BIOS gives the GPU, and what the GPU borrows beyond it is a segment of the RAM's bar.

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

Each GPU's memory and the RAM the engine uses are each one pool of equal-size pages. A page holds an expert, or a
block of a conversation's tokens, or is free.

- The page size is the greatest common divisor of the model's expert sizes, so every expert is a whole number of
  pages and no page is part empty (step M2f). The experts of one layer are all one size; sizes differ by layer. Qwen3.8
  IQ3_S has 7 sizes from 1,510,400 to 2,662,400 bytes: pages of 25,600 bytes, 59 to 104 per expert. DeepSeek's 2.00
  bpw set has 8 sizes from 7,741,440 to 10,229,760 bytes: pages of 92,160 bytes, 84 to 111 per expert. A page is a
  multiple of 256 bytes and at least 16 KiB; a model whose divisor is not gets the size from 16 to 512 KiB that wastes
  least.
- So a pool has millions of pages (about 2 million for all of Qwen3.8 IQ3_S's experts). The pool keeps runs of
  consecutive pages (free, one expert, one block of a conversation, or a conversation's hole), not pages, and finds room
  through ordered indexes.
- Each expert size has a home point: the middle of the room the expert profile's hottest experts of that size need,
  the sizes in order above the conversations' reserve. An expert takes the free run nearest its home, so the experts
  of one size sit together and an evicted one leaves a hole that fits the next one of its size. When there is no free
  run, evicting the least-used expert of the same size always makes room. Nothing moves experts to defragment: the
  unit tests trade pages at random (20,000 steps in a small pool, 300 steps of three conversations of up to 100,000
  tokens in a pool of 1.9 million pages) and the experts afterwards fill the pool as well as they fill an empty one.
- The conversations' end is the bottom of the pool: the `--tokens-min` reserve, which experts leave free, and the
  conversations' fixed regions (a recurrent state of 112 MiB takes one run; when no free run is long enough, the
  lowest stretch of experts is evicted for it). A growing conversation past its free runs evicts the least-used expert
  of any size and fills that hole with its blocks before it takes another; the hole goes back whole when its last
  block leaves.
- A conversation's growing region takes blocks of several consecutive pages when one page would waste much: Qwen3.8's
  int8 K/V unit (an indexer block of 4 tokens, 2 KV heads) is 4,736 bytes, so a page of 25,600 bytes would hold 5 of
  them and waste 7.5%; a block of 5 pages holds 27 and wastes 0.1%. (f16: 8,704 bytes, 32% of a page wasted, 0.8% in
  blocks of 12 pages; q4: 2,816 bytes, 1%; k8v4: 3,776 bytes, 11.5%, 0.4% in blocks of 4. The drafter's K/V has the
  main K/V's format, int8 under k8v4.)
- `GET_MEMORY_USE` gives a pool's pages, how many are allocated, and how many are free in runs shorter than the
  smallest expert (fragmented: free memory no expert can use); the panel shows both as shares of the pool.
- A pool takes its memory in chunks (about 256 MiB; strata-q38's pool on the GPU is one chunk, the expert cache's
  arena, because a run may not cross a chunk's end). Pages need not be next to each other, so growing the memory used
  adds chunks; shrinking it first empties chunks (their experts are evicted, conversation pages are copied to other
  chunks or paused) and then gives them back. No virtual-memory API is needed, so this works on AMD as well as NVIDIA
  (today's live resize, the `VRAM` command, is NVIDIA-only).
- Conversations and experts trade pages without a setting, up to the conversation maximum. Experts take pages back
  only after a conversation ends or is paused, and only after a short wait, so that the two do not thrash.
- On the SSD, an expert is read with its sector-aligned surroundings into a staging buffer, because reads without the
  file cache must be aligned. DeepSeek's Engram reader (its own threads reading the SSD) is the model for this.
- A conversation's memory is described with the existing state descriptor
  (`include/strata/common/state/state_layout.hpp`): fixed regions (the GDN state, the drafter's state) take whole
  pages once; per-token regions (K/V) take a block of consecutive pages per group of tokens as the conversation
  grows. The pool holds the blocks; a conversation keeps a list of its blocks.
- Weights that are always needed (the dense layers, the head) are outside the pools.
- On a PC whose graphics memory is system RAM (Strix Halo), the GPU's pool is the share the BIOS gives the GPU, not
  what the runtime reports as the GPU's size. The runtime reports the share plus what the GPU may borrow from the
  operating system's RAM (on the test PC, 107.6 GiB: a 63.8 GiB share and 47.6 GiB that may be borrowed, while Windows
  sees 63.6 GiB of RAM). Once the share is full, GPU allocations spill into that RAM, so any spill counts against the
  RAM pool and its setting. Otherwise the manager would silently take Windows' memory; DeepSeek reaching Windows'
  commit limit on this PC is likely this.

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

### Experts leaving a memory

RAM normally holds the experts the GPUs do not (the complement), so together they hold more different experts. An
expert evicted from a GPU is therefore copied back to RAM, in the background between steps, instead of being dropped
and read from the SSD again later.

- While RAM has room, it also keeps copies of experts that are on a GPU; evicting those costs nothing. RAM holds
  only the complement when it runs short.
- An autodetected setting drops the expert instead unless copying it back from the card is clearly faster (by
  default at least twice as fast) than reading it from the SSD again: when the two are close, dropping wins, because
  an evicted expert may not be needed again for a while. Both speeds are measured at start (a PCIe 3.0 x4 card, about
  3.2 GB/s, is close to a fast NVMe drive, and a Thunderbolt 4 enclosure, about 2.5 GB/s, is slower than one). The
  user may override it.

### Several GPUs

Today Qwen3.8 has a layer split, Qwen3.6 runs on one GPU, and DeepSeek has its own split with optional tensor
parallelism ([MULTI_GPU.md](MULTI_GPU.md)). The new step has one shared way:

- **A layer split for every model.** Each card owns a range of layers; its pool holds their conversation pages and
  state regions (the descriptor's one layout per device), and their experts by default. The hand-off between cards
  goes through pinned RAM, so cards without peer-to-peer access (GeForce on Windows) work. The planner that picks the
  cut (most of the expert profile on the GPUs, the faster card last) moves from `generate.cpp` to shared code; models
  give per-layer sizes (dense weights, experts, K/V bytes per token).
- **Experts may live on any card.** A card with free pages may hold another card's experts and compute them there,
  all of one layer's remote experts in one hop. This replaces today's helper caches and peer tier with one rule.
- **Links are measured, not assumed.** At start, each card's round-trip latency and its bandwidth each way are
  measured (the engine already probes each card's PCIe share). Remote experts are used on a card only when one hop
  costs less than the CPU computing those experts from RAM, and refills onto a slow card are paced by its bandwidth (a
  2 MB expert takes about 0.6 ms over PCIe 3.0 x4).
- **Memory per card.** The conversation token limits stay global; each card's share follows from its layers. The
  biggest active conversation must fit on every card.
- **Moving the cut while running.** When a card's memory setting changes, the planner may move the cut. The experts of
  moved layers stay where they are (now remote) until use moves them. A layer's dense weights (about 56 MB on
  Qwen3.8) and every conversation's pages for it are copied in the background, the tokens added meanwhile copied
  after, and ownership switches between two steps, one layer at a time.
- **Keeping cards busy.** With several conversations, the step splits them into groups, so that one card runs group
  A while the other runs group B (today's `--batch-groups`, chosen automatically). The rows settings apply per group.
- **Left out:** tensor parallelism (an exchange between cards at every layer, slow over PCIe without peer-to-peer
  access; DeepSeek keeps its `--tp` until there are cards with a fast link to measure), and one model on cards of two
  makes.

### The panel's bars

A memory smaller than 32 GiB is drawn as a bar of its size divided by 32 GiB of the full width (a 16 GiB card fills half);
larger memories fill the width. So RAM is seen to be bigger than a typical card, and cards do not shrink to slivers
beside a lot of RAM.

### Changing sizes by dragging

The boundaries between a bar's segments can be dragged (step M6).

- While dragging, the bar shows the new boundary and its numbers, and nothing is sent to the engine. A size the
  engine would refuse (for example below what the biggest active conversation needs) shows in red with the reason.
- The change is sent (`CHANGE_SETTINGS`) only when the mouse button is released. Escape cancels; dropping it where it
  started changes nothing; releasing on a refused size changes nothing.
- After a change, a small prompt offers "This session" (the default) or "Save as default", and an Undo stays for a
  few seconds.
- Easy to find: each draggable boundary has a visible grip and a resize cursor, and a tooltip says what dragging it
  changes. A one-line hint ("Drag the dividers to change sizes") shows under the panel until the first change.
- Other ways to do the same: clicking a segment opens its size as a number field; a divider can take keyboard focus
  and move with the arrow keys (Enter applies); touch works like the mouse.

## The protocol

New engine commands (between and during requests, like `LIST_CONVERSATIONS`):

- `GET_MEMORY_USE` reports every pool: per memory, its size and its pages by use (experts, active conversations, paused
  conversations, drafter, free), and the settings above. The old engines answer it too, read-only, from what they know
  (sizes from their startup, `LIST_CONVERSATIONS` bytes), so the panel shows them.
- `CHANGE_SETTINGS <name>=<value> [...]` changes settings live: `memory_mib.<memory>`, `tokens_min`, `tokens_max`,
  `ssd_pause_mib`, `drafter on|off`, `concurrent`, `rows`, `lookahead`. The engine answers with the new `GET_MEMORY_USE` report or `ERR <why>` (for example
  "the biggest active conversation needs 3.1 GiB").

The server forwards both through one endpoint (`GET/POST /v1/memory`); "save as default" writes the run config
(`strata-<model>.json`), the one file the web page, the app and the Terminal all read.

## Where the code goes

| Part | Folder |
|---|---|
| Page pool, page lists, tiers, priorities | `src/common/state/` |
| Step: picking conversations, sharing rows, accept | `src/common/spec/` and `src/common/serve/` |
| `GET_MEMORY_USE`/`CHANGE_SETTINGS` parsing and answers | `src/common/serve/` |
| Expert pages (evict, load, count use) | `src/common/experts/` |
| Copies between pool, RAM and SSD | `src/base/gpu/` (the runtime interface), `src/base/platform/` (files) |
| Qwen3.8 rows with parent table, its regions, its drafter | `src/models/qwen38/` |
| `/v1/memory`, saving defaults | `strata_server/`, `strata_common/` (run config) |
| The panel | `strata_server/web/` |

## Steps

| Step | Work | Checked by |
|---|---|---|
| M0 | `GET_MEMORY_USE` in the three existing engines (read-only), `/v1/memory`, the panel without controls | the panel matches the engines' own numbers on Qwen3.8, Qwen3.6, DeepSeek. Done: `dffd3a25` |
| M0b | The desktop app shows the same panel: the panel is one framework-free module that the web page uses directly and the app (`apps/host`, `NativeMonitor`) wraps | the app and the web page draw the same bars from `/v1/memory`. Done: `strata_server/web/memory-panel.js` (and `memory-panel.css`) draws the bars, the conversations the engine holds (strata-q38's list: state, tokens, memory) and the settings; the web page imports it, and the app's Monitor screen wraps it (`MemoryPanel.tsx`; `NativeMonitor.tsx` is shown by no screen), reading the chosen model's `/v1/memory` through the host (`/api/native/memory`) and bundling the same two files. `apps/host/tests/memory-panel.test.mjs` draws strata-q38's own answer (three conversations, the Strix Halo) |
| M1 | The page pool and page lists: chunks, growing and shrinking, experts and conversations trading pages, with unit tests (CPU). Done: `src/common/state/page_pool.cpp` | unit tests: allocation, growth, emptying a chunk, a full pool, no thrashing (`page_pool_test`) |
| M2a | The pool holds Qwen3.8's state: fixed regions in consecutive pages (its GDN state is 112 MiB), truncation, `describe_conversation`. Done: `45eb5a07` | unit tests |
| M2b | Qwen3.8's attention kernels read K/V and the indexer keys from pool pages: a page stride per array (today one row index serves four packed arrays), in the decode, prompt, append and gather kernels. Done: `c4d1578a` (NVIDIA-only variants compiled, not run: run `kv_units_parity` in the NVIDIA session) | a parity test (pool layout bit-identical to today's layout); the old engine's tokens and speed unchanged |
| M2c | The Qwen3.8 model loader and a one-conversation step move out of `generate.cpp`'s `main()` (about 6,200 lines of loading and the decode lambdas) into `src/models/qwen38`; the old engine calls them | the old engine's tokens and speed unchanged Loader done: `7a698d73` (`strata::qwen38::Model::load()`) |
| M2c2 | The one-conversation step: the hit tables and verify hits into `Model`, and a `Conversation` (prompt read, one decode step) wrapping `Prefill` and `Verifier`, called by the old CLI path | the old engine's tokens and speed unchanged. Done (the serve path still has its own step) |
| M2d | The new engine (`strata-q38`, a thin `main()` over `src/models/qwen38` and the shared serve loop): one conversation in the pool, no drafting | token-identical at temperature 0 against the old engine; speed; the pool's pages grow with the conversation. Done: identical tokens with one-row windows; through the server on the Strix Halo, prompts 884-1,108 tok/s against 798-1,042 and decode 26-28 against 25-27; conversation memory 145 MB at 66 tokens and 245 MB at 8.3K, against 3,092 MB reserved by the old engine |
| M2e | Experts in the pool's pages: the expert cache's slots go through the pool, evicted experts go to the RAM tier or are dropped | the same tokens; conversations take expert pages and give them back. Done (`ExpertPages`, `src/common/experts/expert_pages.cpp`): one pool on the GPU, pages of 332,800 bytes for Qwen3.8 IQ3_S (the size per model that wastes least, 4.4% here); on a GPU whose memory is the system RAM the pool stays within the BIOS share (Windows: DXGI's dedicated memory); an evicted expert is dropped (the CPU reads it from the RAM copy or the files) and comes back between requests; `--tokens-min` (default 8192) replaces `--conversation-mib`. Copying evicted experts back to RAM is left for M5, which gives the RAM copy room of its own |
| M2f | Exact-fit pages: the page size is the greatest common divisor of the expert sizes; a pool of millions of pages; experts placed by size around home points; conversations fill expert holes; the pool's allocated and fragmented shares in `GET_MEMORY_USE` and the panel | the same tokens; no fragmentation in random trading; experts that leave come back. Done (`src/common/state/page_pool.cpp`): pages of 25,600 bytes for Qwen3.8 IQ3_S (0% of the experts' pages unused, against 4.4% with M2e's 332,800), the pool keeps runs of pages and ordered indexes, each expert size has a home point and an expert takes the free run nearest it or evicts one of its size, a conversation's K/V blocks are runs of 5 pages (27 int8 units, 0.1% unused, against 7.5% in one page). On the Strix Halo the same 7,846 experts take 15.97 GB instead of 16.69 GB; with the trading test's pool (18.6 GiB) 9,719 experts fit instead of 9,298, and after an 8.5K-token conversation all of them came back (M2e's build stayed 34 MB short). Fragmented share 0% (0.36% with 256 MiB chunks, so the GPU's pool is one chunk). Speed against the M4 build, 4 alternating rounds at 204,800 context, medians: 2K prompt 878 -> 862 tok/s, decode 24.65 -> 24.35; 8K prompt 1,062 -> 1,057, decode 24.13 -> 24.58 (noise). Unit tests: 1.9M pages, 23,960 experts placed in 0.05 s, 300 random steps of three conversations in 0.7 s, every expert back afterwards |
| M3 | Several conversations, the step with rows per step and conversations at once | answers identical per conversation; speed against `--batch`. Done (`strata::spec::Step`, `src/common/spec/step.cpp`): `strata-q38 --batch N --rows R --lookahead L` speaks the old engine's batch protocol (BGEN, BADM, BT, BDONE, BSTOP), so the server's "parallel" works unchanged; each step is one verify window with a row per conversation, each reading its own pages; `share_guess_rows` (the 2, 2, 1 rule) is ready for M4. A conversation takes pages when it gets its first request, nothing per slot at start: on the Strix Halo at 204,800 tokens, four held conversations took 0.49 GB (112 MiB of GDN state each) against the old engine's 12.4 GB, at the same or better speed (aggregate decode, 3 alternating rounds: 1, 2, 3 conversations 26.1, 40.9, 49.5 tok/s against 24.8, 39.7, 48.0). A conversation decoded beside others gives its tokens alone up to a late token in some prompts (mid: token 24 to 63 of 96), also with every expert on the GPU and in the old engine's `--batch`: the multi-row windows round differently |
| M4 | Drafting with the row-sharing rule and the lookahead | identical answers; speed against today's MTP. Done (`ConversationDrafter`, `src/models/qwen38/conversation_drafter.cpp`): `strata-q38 --mtp DIR --lookahead N` loads the MTP drafter (only with a lookahead) and each step asks it for each conversation's share of guesses, stopping below `--spec-min-p`; the drafter's K/V is one more region of each conversation in the pool (801 MiB of drafter against 1,034 MiB with its own K/V ring), and it reads a step's kept rows one step late, in the same pass as its next guesses. A request's `priority=N` orders a step's conversations; the extra rows go to the highest priority and rotate between equals. Answers (5 prompts, 128 tokens): with one guess a step, the same as without drafting up to the token where the old engine's own MTP also drifts (short 66, mid 119; two answers whole); with up to 3 guesses (the server's windows) they drift earlier (14, 24), at near-ties (the top candidates within 0.1-0.3 logits). On the Strix Halo (204,800 context, 3 alternating rounds, medians, tok/s): one conversation, a story / code: old MTP 34.4 / 44.7 (53% / 70% of drafts right), rows 4 lookahead 3 36.6 / 44.5 (58% / 70%), 1x3 35.3 / 46.5, 1x8 31.4 / 44.8 (40% / 56%), no drafting 25.7 / 24.1; three conversations: 3x8 54.7, 3x3 49.8, no drafting 49.7, the old engine's `--batch` 48.9 |
| M5 | Pausing to RAM and SSD, priorities; experts read from the SSD without the file cache | pause and resume give the same answers; files saved before load; with too little memory for the experts, the same answers as with enough, and RAM use stays at the setting. Done, but for pausing a running conversation (`RamExperts`, `src/common/experts/ram_experts.cpp`; `pool_copy`, `src/common/state/pool_copy.cpp`): strata-q38 keeps a second page pool in page-locked RAM (`--ram-mib`; by default what RAM and the commit limit have free less the resident headroom, 39-43 GiB on the Strix Halo) with the GPU's page size and conversation layout. At start it takes the experts the GPU's pool does not hold (the profile's order), then copies of the GPU's; an expert decoding reads from the SSD (unbuffered: `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`, through the file source) takes the place of the least recently used one of its size, never one the current layer used, else goes to a staging ring. This replaces strata-q38's RAM copy, arena and file tier (`--resident-experts` and `--mmap-experts` do nothing there; `--resident-budget-gib` stands for `--ram-mib`). An expert the GPU evicts is copied back to RAM when that measures at least twice as fast as an SSD read (`--copy-back auto`, best of three at start; on the Strix Halo 29-57 GB/s against 2.3-2.6 GB/s, so it is copied). A conversation that loses its place on the GPU (another request takes the place, or a growing one needs the pages: the lowest priority, then the least recently used, first) is paused: its blocks are copied to the RAM's pool while the conversations paused there take no more than one conversation of `--tokens-max` tokens (default `--max-context`), the lowest-priority one moving down to the SSD first; else into a file, written and flushed before its pages are given away (`--pause-dir`, at most `--ssd-pause-mib`, default 8192). A request whose prompt starts with a paused conversation resumes it; a seat whose GDN state comes back in other pages drops the verify windows' captured graphs. `GET_MEMORY_USE` reports the RAM's pool (experts, free, staging ring, allocated and fragmented pages, the paused budget) and the SSD (experts in neither pool, paused conversations, their budget). On the Strix Halo: answers identical to M2f (5 prompts, now with pauses between them); a 2,000-token conversation paused behind another and continued gives the same 48 tokens paused in RAM (140 MiB in 26 ms, back in 28 ms), on the SSD (135 ms, back in 65 ms) and moved from RAM down to the SSD; with a 4.4 GiB GPU pool, a 1 GiB RAM setting gives the same tokens as a 42 GiB one (with `--pcie-frac 0` and the prompt's CPU share off, as which experts the GPU reads over the bus depends on what RAM holds), the working set 2.9-3.8 GiB against 44.4-45.3 GiB. Against the old engine's low-RAM mode (`--resident-budget-gib`), the same 3000-slot GPU cache, two stories of 160 tokens, 2 rounds: 8 GiB of RAM decodes 9.0-10.3 tok/s against 7.4-10.5 (2.9-3.5 before the pool kept what decoding reads), working set 11.1 against 10.0 GiB; 1 GiB decodes 3.1-3.3 against 5.3-8.5, working set 4.1 against 2.8 GiB: there the old engine's adaptive tier moves the conversation's experts into VRAM, and strata-q38's GPU pool keeps the profile's. Left: the GPU pool taking the most-used experts while decoding; pausing a running conversation for a higher-priority one; the copy-back runs when the expert is evicted (between steps), not on a thread of its own |
| M5b | The GPU pool takes the experts decoding uses; a running conversation paused for a higher priority; stale pause files; the RAM outside the pools | the same tokens with the adaptive tier off; the low-RAM settings at least as fast as the old engine's; a preempted and a preempting answer complete. Done: strata-q38 counts each expert's routings while decoding (the old engine's counts, `--adapt-every`, `--adapt-swaps`, `--adapt-min-gain`, `--adapt-decay`) and every 4 steps moves the most-routed experts the GPU's pool lacks into the pages of the least-routed ones of their layer, by the old engine's rule, now shared (`choose_expert_swaps`, `include/strata/common/experts/expert_swaps.hpp`; its margin of 1.5 routings keeps experts from trading back and forth); the experts routed since the last round count as used, so a growing conversation evicts the least recently used ones, not the profile's coldest (M2e's open issue). A request's `priority` (an API field) can preempt: a BGEN of a higher priority to a busy slot ends that slot's request with `BDONE <slot> <n> preempted`, its conversation is paused, and the server sends it again with its tokens so far once a slot is free, which resumes it (`READY ... preempt`; the server picks the slot of the lowest priority below the new one's). Pause files carry the engine's process id, and a starting engine deletes those of engines no longer running. The RAM outside the pools was mostly the file source's read buffers, which each reading thread keeps at the size of its largest read: the prompt path's 32 threads kept 16-28 MiB each and the engine's thread 156 MiB from the start-up fill; reads on the prompt path now take at most 8 MiB. On the Strix Halo (same 3000-slot GPU cache, two stories of 160 tokens, 3 alternating rounds, medians, tok/s): 1 GiB of RAM 5.96 / 7.93 against the old engine's 5.25 / 7.91 and M5's 3.02 / 3.12; 8 GiB 9.58 / 11.13 against 7.39 / 10.58 and M5's 9.92 / 10.52; all the RAM 15.00 / 14.04 against 14.58 / 13.88. The moves cost 0.65-1.0 s per 160 tokens. Working set at 1 GiB: 3.76 GiB (M5 4.13, the old engine 2.83); what is left above the old engine is the page-locked region (2.36 GiB, fully resident, against 1.7 GiB) and the prompt path's smaller read buffers. Copy-back stays inline: it took at most 5.3 ms per 160 tokens (the RAM's pool rarely has room for an evicted expert colder than what it holds), so a thread of its own would save nothing measurable. With the adaptive tier off (`--adapt-swaps 0`) the answers are M5's (5 prompts); with it on, a 1 GiB RAM setting gives the same tokens as 42 GiB (with `--pcie-frac 0` and the prompt's CPU share off, 1,145 experts moved). One batch slot, temperature 0, tier off: a story preempted after 20 tokens by a request of priority 5 and continued gives the same 96 tokens as uninterrupted, and the preempting answer is the same too (with the tier on, the preempting answer differed after its first sentence: the GPU's experts then are another set than in the uninterrupted run). Left: the copies into the GPU's pool wait for each expert (a queued copy could overlap them, about 1 s per 160 tokens at most); a preempting request that fails after its BGEN can leave the slot's bookkeeping in the server to the other request's BDONE wait; a pause file whose process id a new process took stays. |
| M6 | Live `CHANGE_SETTINGS`, the panel's controls (web page and app), "this session" and "save as default" through the server; the app's own memory and parallelism settings (`parallel`, `mtp`, `vram_reserve_mib`, ...) give way to these, so the server is the only writer | resize while decoding; the biggest-conversation limit refuses. Done: strata-q38 answers `CHANGE_SETTINGS` (READY says `settings`) with the new `GET_MEMORY_USE` report or `ERR <why>`, which changes nothing. Both pools now take their memory in chunks of 256 MiB (the GPU's: the cache's arena is the first chunk, the others are allocated a page larger and start above it at a whole number of pages, so that the kernels' arena-plus-slot addressing holds; no virtual-memory API, so it works on AMD), and `memory_mib.<id>` grows a pool by chunks or shrinks it by emptying chunks: their experts are evicted, the conversations' blocks there are copied down (the K/V tables and the GDN state's address are rewritten, `ConversationPages::refresh`), and when they do not fit, idle conversations are paused, then the lowest-priority batch slot is preempted. A shrink is refused only below what the token minimum and the biggest decoding conversation at the end of its request need (`smallest=` on the `MEMORY` line, with `setting=` and `largest=`). Also live: `tokens_min`, `tokens_max` (the RAM's paused budget, as in M5), `ssd_pause_mib`, `drafter on|off` (off gives the weights back: 1.25 GiB more free on the Strix Halo; on loads them again in 0.4 s), `concurrent` (up to `--batch`; the waiting conversations take turns), `rows` (the verify windows are now sized for 8) and `lookahead`. New flags for the defaults: `--gpu-mib`, `--concurrent`, `--drafter`. The server: `POST /v1/memory {"set", "save": "session"|"default"}` (409 with the engine's reason; "default" writes the args of `strata-<model>.json` and keeps the .bak); `GET` adds `can_change`, `defaults`, `config`. The panel (`memory-panel.js`) drags the end of each pool and of the SSD's paused room (preview, red with the reason outside the range, nothing sent until release, Escape, arrow keys), has controls for the other settings, the session/default choice, Undo and the refusal message; the app forwards it (`POST /api/native/memory`), and its settings save sends `parallel` (as `concurrent`) and `mtp` (as `drafter`) through the server when the running engine can change them. On the Strix Halo: decoding a story while the GPU's pool went from 2 GiB to one chunk (an idle 8,500-token conversation paused, the story's pages moved down), the RAM's pool to half, both back, gave the same 96 tokens as without the changes, and the paused conversation continued to the same tokens (`--pcie-frac 0`, prompt CPU share off); each resize took 0.2-0.4 s with the story decoding, 2-16 ms on the engine's side through the server; `concurrent` 1 and back gave both answers unchanged; without changes the 5 prompts give M5b's answers (with `--pcie-frac 0` and the prompt CPU share off, as the RAM's pool now ends at a whole chunk). Left: more `concurrent` than `--batch` needs a restart; Windows counts the RAM's page-locked pool in the BIOS share's use (81 GiB of 63.8), so growth after start is limited by the free memory, not the share, and the server's Strix Halo bars read the share as full; a drafter switched on mid-conversation guesses from K/V it did not write until the next new conversation |
| M6b | M6's open issues: the Strix Halo's share counted with the RAM's pool in it, a resize that gets less than asked answered as success, the verify windows sized for 8 rows | the bars and the pools' ranges match Windows' own counters; a shortfall reaches the panel. Done: Windows counts page-locked host memory in the GPU's local use (DXGI: 4 GiB of hipHostMalloc raised it by 4 GiB) and in the process's shared GPU use (PDH), not its dedicated use, and HIP takes it off its free memory twice (8 GiB: 16 GiB). `get_share_room` takes the RAM pool's page-locked bytes off, and on a GPU whose share is known the GPU pool's room is the share's, not the runtime's free memory. The RAM's `MEMORY` line says `mapped=` (the page-locked pool), and the server takes it off what the GPU borrows: on the Strix Halo the RAM's bar counted the 20.95 GiB pool twice (as RAM experts and as 22.08 GiB borrowed by the GPU); now 1.05-1.13 GiB is borrowed (the engine's other page-locked buffers), and `in_share` (58.3 GiB) is what the engine holds in the 63.8 GiB share. Found on the way: ROCclr on Windows keeps freed memory in its resource cache and gives none back, so M6's shrinks freed nothing (a RAM pool shrink from 21.45 to 7.95 GiB left Windows' free RAM at 22.6 GiB); strata-q38 sets `GPU_RESOURCE_CACHE_SIZE=0` (an explicit setting wins), and the same shrink raised free RAM from 22.4 to 36.0 GiB, a GPU shrink from 49.75 to 40 GiB the share's free part from 2.2 to 12.0 GiB. Also found: since M6 the GPU's pool held only the experts of its first chunk's slot count (10,485) at start and left the rest of the pool free (on the Strix Halo 32 GiB of a 49.75 GiB pool); the others now come in by `refill` once the engine is idle (all of them within 20 s). And the default sizes let Windows' commit charge reach its limit (the GPU's memory counts in it on the Strix Halo: 8 GiB of hipMalloc raised it by 8.3 GiB; the RAM pool took what was left less 4 GiB, 23.9 GiB, and the PC froze): the pools now leave a tenth of the commit limit, at least 8 GiB, uncommitted (`get_commit_room`). A resize that gets less than asked: `CHANGE_SETTINGS`' answer has a `SHORTFALL <what>` line, `POST /v1/memory` a `"shortfall"`, and the panel says "Changed ... But it got less than asked: ...". The verify windows' buffers: 85.8 MiB at 8 rows against about 46 MiB at 1 (5.6 MiB a row, from the logs of 2- to 8-row windows), 0.06% of the share, so they stay sized for 8 |
| M7 | Measurements: guesses against other conversations' tokens, by where experts live; defaults chosen from them | numbers in DETAILS.md, measured on the Strix Halo and, if possible, the NVIDIA test PC. Done on the Strix Halo ([DETAILS.md](DETAILS.md#rows-per-step-lookahead-and-conversations-at-once-strata-q38), `tools/measure_rows.py`; the NVIDIA PC's run is prepared in `tools/nvidia_session/README.md`): 8 configurations from 1x1 to 3x8 with story, code and chat prompts, every expert on the GPU, most in RAM, about half on the SSD, 3 alternating rounds. A guess routes to fewer experts than another conversation's token (295-353 against 462 a row, of 480 for a token alone), but is kept only 50-72% of the time. One conversation: 3 guesses a step (4 rows) as fast as 2 and faster than 7 for prose (GPU: story 32.4 tok/s, 21.2 without guesses); several: 8 rows beat one row each (three conversations 52.0 against 40.8 tok/s on the GPU, 32.9 against 24.9 in RAM, 14.2 against 13.5 on the SSD). Defaults: rows 8 (kept), lookahead 3 with `--mtp` (it was 0 unless given), concurrent all the slots, `--tokens-min` 8192 and `--copy-back auto` kept (neither measurably costs or saves here) |
| M8 | Several GPUs: the shared layer split and planner, experts on any card, measured links, groups | tokens identical to one GPU; speed against today's layer split on the NVIDIA test PC (3060 + 3080, x16 and x4 if possible) In part (2026-11-01): the planner and the rules are shared code, unit-tested on synthetic cards; strata-q38 still runs on one card. Done: the layer split's planner (`plan_layer_split`, `src/common/experts/layer_split.cpp`) moved out of the old engine's loader, which calls it (its search, ties and retries unchanged); a model gives it per-layer sizes (`LayerSizes`: dense weights, one expert, fixed state, bytes per token) and each card its memory, speed and link (`make_paged_split_problem`). Links are measured (`src/base/gpu/links.cu`: each way and a small copy's round trip, to the host and between two cards); the rules (`src/common/experts/card_rules.cpp`): a hop's cost against the CPU's, refills paced by a card's link, copy-back only when at least twice as fast as an SSD read, and `place_experts` (each card its own layers' experts, then a card with room left the hottest experts of another card's layers where one hop is cheaper than the CPU, one card per layer). strata-q38 measures its GPU's link at start and paces refills and copy-back by these rules. Unit tests (`layer_split_test`, `card_rules_test`): a 3060 12 GB on PCIe 3.0 x4 and a 3080 20 GB on x16 cut at K=16 (14,080 of 24,576 experts on the cards); the 3080 at 9 GiB moves the cut to K=26; four equal cards 12,24,36; a Thunderbolt enclosure (2.5 GB/s) and an x4 card drop evicted experts (an NVMe drive reads as fast), an x16 card copies them back; one hop of 8 rows costs 0.13 ms over x4 and 0.23 ms over Thunderbolt, so it pays from 2 and 4 routed experts against 0.1 ms each on the CPU. On the Strix Halo (one GPU): strata-q38's 5 prompts give the M7 build's tokens (`--gpu-mib 14336 --ram-mib 8192 --pcie-frac 0`); its link measured 68.5-76.0 GB/s to the GPU, 61.1 back and 0.08-0.18 ms a round trip in two starts. Left: strata-q38 on several cards (a page pool and conversation regions per card, the verify windows and prompt path chained over the cards' layer ranges as the old engine's split does, pausing and resizing per card, `memory_mib.gpu<N>` per card), experts computed on another card in one hop (the kernels' side: today's helper caches compute a layer's experts on another GPU), groups, and a test mode that runs two stages with their own pools on one GPU. The NVIDIA session's `memory` phase (`tools/nvidia_session/README.md`) checks the old engine's split against M7's build now, and runs the two-card checks once strata-q38 takes a split |
| M9 | Moving the layer cut while running | a cut moved during decoding gives the same tokens Not started: it needs M8's strata-q38 on several cards. The planner's side is there (the same `plan_layer_split` with the cards' new memory settings; the unit test moves the cut when the 3080's setting drops), and the NVIDIA session's `memory-cut-move` check (a story decoded while `CHANGE_SETTINGS` halves the first card's memory and sets it back) is ready |

M0 is useful on its own and touches no engine logic. M2 to M6 are the new engine; the old Qwen3.8 path stays until
the new one is at least as fast.

## Later

- **Tree drafting**: a conversation's share of rows spent as a tree of guesses (several likely continuations
  checked in one pass). The parent table of the step allows it from the start; it needs the drafter to propose trees,
  the verify kernels to take tree-shaped parents, and for GDN layers a running state per branch.
- Qwen3.6 and DeepSeek on the new step and pools.
- More than 8 rows per step, and so more than 8 conversations at once: the verify kernels widened beyond 8 rows.
  Not several passes per step, which would read the weights more than once. Kernel work for a later pull request,
  not this one.

## Open questions

- The desktop app's Windows and Linux installers download the engine from upstream's signed feed. App users get the
  new engine only if upstream takes it, or if the fork publishes a feed of its own. The Terminal and source installs
  can use the fork's build.

- Page size (decided in M2f): one page size, the greatest common divisor of the expert sizes; conversation blocks
  take several consecutive pages where one page would waste much of a page. Open: a conversation's hole wastes the
  pages past its last whole block (a 59-page hole holds 11 blocks of 5 pages, 4 pages unused while the hole is
  partly used); filling a whole hole with K/V units would need a unit table per hole instead of per block.
- Should a paused conversation's pages on the SSD use the session file format (`conversation_file.hpp`), so
  "save" and "pause to SSD" are one thing? M5 writes its own (`pool_copy`: the blocks of each region, as the pool
  holds them), and strata-q38 takes no session files.
- What the panel shows for an old engine's settings it cannot change live: greyed out, or "applies after a restart"
  with a restart button?

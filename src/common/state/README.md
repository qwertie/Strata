# src/common/state/

Per-conversation state: the state descriptor (what memory a model keeps per conversation), the page pool, snapshots,
swapping to RAM or disk, conversation files. Headers: `include/strata/common/state/`.

The conversation cache, buffers, files and memory checks, the session-save reclaim, the conversation listing
(`token_listing`) and the progress heartbeat came here from `src/core/`. The snapshot code (`conversation_snapshot`,
`conversation_state`) is still Qwen3.8's, in `src/models/qwen38/`.

## The state descriptor (`state_layout.hpp`)

A model describes the memory one conversation keeps as a `StateLayout`: a list of `StateRegion`s, each one buffer.

| `RegionKind` | Size at N tokens | Examples |
|---|---|---|
| `fixed` | `unit_bytes` | a GDN/SSM recurrent state, a conv history, the last hidden row, DeepSeek's window ring |
| `per_token` | `rows` x N x `unit_bytes` | K or V (`rows` = KV heads, head-major) |
| `per_block` | `rows` x (N / `block`) x `unit_bytes` | DeepSeek's compressed K/V and index keys, Qwen3.8's pooled index keys |

In a session, each row of a growing region has room for the session's whole context (`cells`). The shared code does
the bookkeeping on the description alone:

- `bytes_for(N)`: the size of a session at its context, or of a conversation of N tokens (`LIST_CONVERSATIONS`).
- `alloc_state`: one zeroed device buffer per region (`StateBuffers`).
- `copy_state`: the first N tokens' state from one place to another: session to session on the device, or session to
  a packed host image (`get_host_image`) and back. A packed image is what parking a conversation in RAM or writing it
  to a file needs. `for_each_region_copy` gives the rectangles without copying (the unit test copies them with
  `memcpy`).
- `clear_fixed_state`: a new conversation's start.

A model split across devices describes one layout per device. Device memory and copies go through
`include/strata/base/gpu/runtime.hpp`, not the CUDA runtime.

### How the engines map onto it

- **Qwen3.6** (`src/models/qwen36/q36_engine.cu`) uses it: per layer K and V (`per_token`) or the conv history and
  the recurrent state (`fixed`), the MTP block's K and V, the last hidden row. `alloc_session`, `copy_session` (a
  conversation read in one batch session that continues in another), `reset` and the session size it reports are
  the shared calls.
- **Qwen3.8** (`src/program/generate.cpp`, `src/models/qwen38/`) uses it for its running state:
  `describe_running_state` in `src/models/qwen38/conversation_state.cpp` lists the `fixed` regions a session owns
  (the GDN rows, the PLE history, and per QSA layer the indexer tail, dead row and block position), and the
  checkpoint save, restore and slot copy are `copy_state` calls between the session and a checkpoint's vectors or
  another session. Its K/V is not on it yet: the session (`session_init`) carves the K/V and the scratch buffers
  from one allocation, the K/V is paged (int8 with scales under `--kv int8`) and partly streamed from host memory
  (`--kv-resident`), and `conversation_snapshot.cpp` copies its five parts (K, V, their scales and the pooled index
  keys) through segmented host buffers. Those parts map onto `per_token` regions and the pooled keys onto
  `per_block` (plus the spare row the next block fills). They move when the paged K/V pool comes: a page pool
  replaces a growing region's rows, and `copy_state` then copies page by page.
- **DeepSeek V4.1** (`src/models/deepseekv41/`) does not use it yet. Its slot (`slot_bytes`, `slot_save`,
  `slot_load` in `ds4_model.cu`) maps onto one layout per device: per layer the window ring and the compressor's
  partial group (`c_kv`, `c_score`) as `fixed` regions, per KV source the compressed K/V and the index keys as
  `per_block` regions with `block` = the source's ratio. Describing it would delete little: the slot file is no
  packed image (a length before each region, the identity and tokens before them, the checkpoints' host copies
  after them), a tensor-parallel group's members each hold a copy of the ring and partial group that a load writes
  into all of them, a source's rows are written to every device that keeps them, and the copies are synchronous
  `cudaMemcpy` calls that `copy_state` would make asynchronous. It moves with its conversation slots in RAM and on
  disk (`ds4_main.cpp`), when those become the conversation cache's.

## The page pool (`page_pool.hpp`)

Step M1 of [docs/MEMORY_MANAGER_PLAN.md](../../../docs/MEMORY_MANAGER_PLAN.md); no engine uses it yet. A
`PagePool` is one memory (a GPU's, or the RAM the engine uses) cut into equal pages (2 MiB by default), taken from an
allocator in chunks (256 MiB by default). The allocator, the clock and the page copy are hooks, so the pool calls no
GPU runtime and its test (`page_pool_test.cpp`) runs on host memory.

- A page is free, holds an expert (`ExpertKey`: layer and expert), or holds part of one region of a conversation.
  A `fixed` region takes whole pages once; a growing region takes one page per block of tokens, all its rows in the
  page (`StateRegion::tokens_in` gives the tokens a page holds). A conversation keeps the list of its pages per
  region (`conversation_pages`).
- A growing conversation takes the lowest free page, then the page of the expert used longest ago, up to the token
  maximum. Experts take the highest free page, leave the token minimum's pages free, and wait for the hysteresis
  after a conversation gave a page back. The `evicted` hook tells the expert's owner, which copies it to RAM or drops
  it.
- `resize` adds chunks, or gives back the last ones: their experts are evicted and their conversation pages copied
  into the others through the `copy` hook. When those pages have nowhere to go it names the conversations to pause.
  A resize or a limit that the active conversations or the minimum would not fit is refused with the reason.
- `report` gives the pages and bytes by use for the `GET_MEMORY_USE` command.

The pool is not thread-safe: the engine's step loop is its only caller.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

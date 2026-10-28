# src/common/state/

Per-conversation state: the state descriptor (what memory a model keeps per conversation), the page pool, snapshots,
swapping to RAM or disk, conversation files. Headers: `include/strata/common/state/`.

The conversation cache, buffers, files and memory checks, the session-save reclaim, the conversation listing
(`token_listing`) and the progress heartbeat came here from `src/core/`. The snapshot code (`conversation_snapshot`,
`conversation_state`) is still Qwen3.8's, in `src/models/qwen38/`. **A5** makes it model-neutral, moves it here and
replaces DeepSeek's own conversation slots.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

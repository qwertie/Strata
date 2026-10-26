# src/common/state/

Per-conversation state: the state descriptor (what memory a model keeps per conversation), the page pool, snapshots,
swapping to RAM or disk, conversation files. Headers: `include/strata/common/state/`.

**A3** moves in the model-neutral conversation files from `src/core/`. **A5** makes the snapshot code model-neutral and
replaces DeepSeek's own conversation slots.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

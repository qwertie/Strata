# src/models/qwen38/

The Qwen3.8-Flash-Next engine. Headers: `include/strata/models/qwen38/`.

`ngram/` reads the per-layer-embedding n-gram table; `plan/` is the memory planner. The files at the top came from
`src/core/`: the layer, the session, the MTP draft layer, the verify window, the native dense weights and head, the
DFlash2 drafter (with its parity test and benches), the conversation snapshots (until **A5** makes them
model-neutral), and the batch-window and drafting helpers.

**B1** moves in its own kernels from `src/kernels/`.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

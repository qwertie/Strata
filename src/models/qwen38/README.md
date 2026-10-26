# src/models/qwen38/

The Qwen3.8-Flash-Next engine. Headers: `include/strata/models/qwen38/`.

**A3** moves in the Qwen3.8-only files of `src/core/`, plus `src/ngram/` (the per-layer-embedding n-gram table) and
`src/plan/` (the memory planner). **B1** moves in its own kernels from `src/kernels/`.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

# src/common/experts/

The expert store for mixture-of-experts models: the RAM tier, VRAM slots, prefetch and eviction. Headers:
`include/strata/common/experts/`.

**A5** moves in the model-neutral parts of `src/core/expert_cache.cpp` and `expert_source.cpp`, and replaces DeepSeek's
own store (`src/dsv41/ds4_store.hpp`). **B3** adds the multi-GPU parts (`peer_experts.cpp`, `remote_experts.cpp`).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

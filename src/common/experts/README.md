# src/common/experts/

The expert store for mixture-of-experts models: the RAM tier, VRAM slots, prefetch and eviction. Headers:
`include/strata/common/experts/`.

What is here so far is the model-neutral host code, which needs no GPU:

- `expert_profile`: the expert profile file (`--expert-profile`, `--expert-profile-save`, `tools/make_profile.py`).
- `ram_tier`: how much RAM the RAM tier may take (MemAvailable, cgroup limits, NUMA nodes, Windows commit) and which
  experts its compact copy (the cache complement) holds.
- `exchange_storage.hpp`: the adaptive tier's exchange of experts between the RAM copy and the GPU cache.
- `hit_hook.hpp`: the seam between Qwen3.8's host loop and the VRAM tier.
- `uma_cache.hpp`: the GPU cache's size on a GPU that shares the system RAM.
- `expert_pages`: experts in the pages of a page pool (`src/common/state/page_pool.cpp`) that conversations share: a
  GPU's cache or the RAM's copy; the page size per model, experts leaving for a growing conversation and coming back
  (strata-q38, docs/MEMORY_MANAGER_PLAN.md steps M2e and M5).
- `ram_experts`: the RAM's page pool as an expert source (step M5): at start the experts the GPU's pool does not hold,
  then copies of the GPU's while room allows; an expert the GPU evicts copied back, and the paused conversations. The
  rest are read from the SSD without the file cache, and an expert decoding reads there takes the place of the least
  recently used one of its size (else it goes to a staging ring). It calls the runtime through
  `src/base/gpu/runtime.hpp`.

The SYCL build compiles `expert_profile` and `ram_tier` too, without copies of them; `expert_pages` and `ram_experts`
are for strata-q38, which the SYCL build does not make.

Still in `src/core/`: the GPU cache (`expert_cache`) and the expert sources with their prefetch and the adaptive tier's
copies (`expert_source`, `foresight_swap.hpp`). They call the CUDA runtime directly (streams, events, page-locked and
mapped host memory, the driver's virtual memory functions), which code in `src/common/` must not, and
`src/base/gpu/runtime.hpp` does not offer those calls yet. Qwen3.6 and DeepSeek do not use this store: Qwen3.6 has its
own hybrid slots in `src/models/qwen36/q36_engine.cu`, DeepSeek its own store in
`src/models/deepseekv41/ds4_store.hpp`.

The multi-GPU parts stay in `src/core/` too (B3): the peer tier (`peer_experts`) and the helper caches
(`remote_experts`, `remote_expert_opt`). Besides calling the CUDA runtime directly, they compute with Qwen3.8's kernels
(`include/strata/models/qwen38/kernels/`), a layer above this one.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

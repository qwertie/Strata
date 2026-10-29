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

The SYCL build uses these files too; it has no copies of them.

Still in `src/core/`: the GPU cache (`expert_cache`) and the expert sources with their prefetch and the adaptive tier's
copies (`expert_source`, `foresight_swap.hpp`). They call the CUDA runtime directly (streams, events, page-locked and
mapped host memory, the driver's virtual memory functions), which code in `src/common/` must not, and
`src/base/gpu/runtime.hpp` does not offer those calls yet. Qwen3.6 and DeepSeek do not use this store: Qwen3.6 has its
own hybrid slots in `src/models/qwen36/q36_engine.cu`, DeepSeek its own store in
`src/models/deepseekv41/ds4_store.hpp`.

**B3** adds the multi-GPU parts (`peer_experts.cpp`, `remote_experts.cpp`).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

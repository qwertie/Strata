# src/models/deepseekv41/

The DeepSeek V4.1-Flash engine. Headers: `include/strata/models/deepseekv41/`.

**A5** and **B2** replace its own copies of shared code; A4 moved its OS services to
`include/strata/base/platform/os.hpp`. Its plain sampling is the shared `sample_tokens`; its quant kernels
(`ds4_iq`, `ds4_gguf`, `ds4_tq`), its draft sampling (`vendor/sampler.cu`) and its worker pool (`ds4_pool.hpp`)
stay its own (see the open issues in REFACTOR_PLAN.md).

`conversation_layout.cpp` describes a conversation's state as page-pool regions (step D1 of
[docs/MEMORY_MANAGER_PLAN.md](../../../docs/MEMORY_MANAGER_PLAN.md#deepseek-v41-flash-on-the-memory-manager)); its
test is `conversation_layout_test.cpp`. The geometry (`geometry.hpp`) and the cache rows' sizes (`kv_rows.hpp`) are
headers the old engine reads too.

The model loads in three calls (`ds4_model.hpp`): `Model::load` (the dense weights, the embedding table, the
Engram's reader), `reserve_caches` (the conversation's state for the whole context) and `place_experts` (the prompt
path's buffers, the routed experts in the GPU memory left, the CPU lane and the RAM store). `strata-dsv41`
(`ds4_main.cpp`) calls all three; `strata-ds41` (`paged_model.cpp`, step D4, `main()` in `src/program/ds41_main.cpp`)
skips `reserve_caches`: it opens the GPU's page pool first and points the model at a conversation's pages
(`use_state_memory`). Both read the model's options with `read_option` (`ds4_options.cpp`).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

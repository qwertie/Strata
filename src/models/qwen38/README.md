# src/models/qwen38/

The Qwen3.8-Flash-Next engine. Headers: `include/strata/models/qwen38/`.

`model.cpp` is the model loader (`Model::load`, from `strata`'s main(), step M2c of
[docs/MEMORY_MANAGER_PLAN.md](../../../docs/MEMORY_MANAGER_PLAN.md)); `split_rules.hpp` and `batch_groups.hpp` (with
their tests) came from `src/program/` with it. `options.cpp` is the command line, `conversation.cpp` one sequence's
prompt path and verify windows (M2c2). `conversation_pages.cpp` keeps a conversation's state in a page pool and
`paged_engine.cpp` is `strata-q38`, the engine built on them (M2d; several conversations at once through the
shared step, M3). `ngram/` reads the per-layer-embedding n-gram
table; `plan/` is the memory planner. The files at the top came from `src/core/`: the layer, the session, the MTP
draft layer, the verify window, the native dense weights and head, the DFlash2 drafter (with its parity test and
benches), the conversation snapshots (until **A5** makes them model-neutral), and the batch-window and drafting
helpers.

`kernels/` and `prefill/` came from `src/kernels/` and `src/prefill/` (**B1**): the kernels of the layers only
Qwen3.8 has, or that are compiled for its shapes (`core/geometry.hpp`): QSA attention with its indexer, selection and
K/V formats (`qsa*`, `kv_*`), the gated residual (`gr`, `fused_gr`), the GDN layer and the verify window (`gdn`,
`native_gdn*`, `verify_kernels`), the per-layer embeddings (`ple`, `ngram`), the grouped GPU experts and the shared
expert, the DFlash2 drafter's small kernels and control vectors; and the prompt path (`prefill/prefill.cpp`) with its
element-wise and recurrent kernels.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

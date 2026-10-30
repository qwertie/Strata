# src/models/common/

Code shared by several model families: model geometry, tensor layout tables, and the GDN, gated attention and
MoE-routing layers and kernels. Layer 2: it may use `src/base/` and `src/common/`. Headers:
`include/strata/models/common/`.

`family_layout.hpp` came from `include/strata/core/`. `geometry.hpp` stays there while code of a lower layer uses it:
`include/strata/common/artifact/gguf_reader.hpp` reads a model file's shape into its `ModelGeometry`.

`kernels/` holds the kernels that Qwen3.6 and Qwen3.8 both use, from `src/kernels/` (**B1**): the GDN step with its
output norm (`fused_gdn`), the MoE router (`native_router`) and the routed experts' combine (`native_moe`).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

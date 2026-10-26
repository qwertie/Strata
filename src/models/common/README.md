# src/models/common/

Code shared by several model families: model geometry, tensor layout tables, and the GDN, gated attention and
MoE-routing layers and kernels. Layer 2: it may use `src/base/` and `src/common/`. Headers:
`include/strata/models/common/`.

**A3** moves in `include/strata/core/{geometry,family_layout}.hpp`. **B1** moves in the kernels that Qwen3.6 and Qwen3.8
both use.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

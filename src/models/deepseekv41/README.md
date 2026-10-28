# src/models/deepseekv41/

The DeepSeek V4.1-Flash engine. Headers: `include/strata/models/deepseekv41/`.

**A5** and **B2** replace its own copies of shared code; A4 moved its OS services to
`include/strata/base/platform/os.hpp`. Its worker pool (`ds4_pool.hpp`) waits for a speed run (see the open issues in
REFACTOR_PLAN.md).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

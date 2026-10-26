# src/base/platform/

Operating-system services: files read at an offset, mapping, NUMA, thread pinning, free RAM, worker pools. Headers:
`include/strata/base/platform/`.

**A3** moves `src/platform/` here. **A4** merges in DeepSeek's copies (`src/dsv41/ds4_os.hpp`, `ds4_pool.hpp`) and the
worker pool from `src/kernels/cpu/pool.cpp`.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

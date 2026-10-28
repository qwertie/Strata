# src/base/platform/

Operating-system services: files read at an offset, mapping, NUMA, thread pinning, free RAM, worker pools. Headers:
`include/strata/base/platform/`.

`os.hpp` is the DeepSeek engine's header-only set of OS services (A4). The worker pools of
`src/models/deepseekv41/ds4_pool.hpp` and `src/kernels/cpu/pool.cpp` will become one pool here once both engines'
speed can be measured ([REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md#open-issues)).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

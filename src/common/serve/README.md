# src/common/serve/

The engine side of the protocol that the Python server speaks: request queue, scheduler, streaming, cancel. Headers:
`include/strata/common/serve/`.

**A5** moves the protocol code here from the three engines (`src/program/generate.cpp`, `src/qwen35/q36_engine.cu`,
`src/dsv41/ds4_main.cpp`), which each have their own today.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

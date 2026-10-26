# src/base/gpu/

The device runtime interface (memory, copies, streams, events, graphs) with one implementation per backend, the GPU
hardware traits, and the HIP compat headers. Headers: `include/strata/base/gpu/`.

**A3** moves in `src/core/{device.cu,pinned.cu,graph.cpp}`, the hardware-trait headers from `include/strata/core/` and
`include/strata/kernels/gfx_arch.hpp`, and both `hip_compat/` folders. **A5** adds the runtime interface, so that shared
code stops calling the CUDA runtime directly.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

# src/base/gpu/

The device runtime interface (memory, copies, streams, events, graphs) with one implementation per backend, the GPU
hardware traits, and the HIP compat headers. Headers: `include/strata/base/gpu/`.

The HIP compat headers are two different sets, each put first on the include path of its own build:
`include/strata/base/gpu/hip_compat/` for the wave32 HIP backend (`STRATA_ENABLE_HIP`) and
`include/strata/base/gpu/hip_compat_gfx906/` for the wave64 gfx906 build (`STRATA_HIP_GFX906`).

**A3** moves in `src/core/{device.cu,pinned.cu,graph.cpp}`, the hardware-trait headers from `include/strata/core/` and
`include/strata/kernels/gfx_arch.hpp`. **A5** adds the runtime interface, so that shared code stops calling the CUDA
runtime directly.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

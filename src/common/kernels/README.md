# src/common/kernels/

Device kernels that work for any model: dequantization, quantized matrix-vector products, GEMM, the sampler, elementwise
operations, rope. Headers: `include/strata/common/kernels/`.

They came unchanged from `src/kernels/` and `src/prefill/` in **B1** (only their paths changed), with their parity tests
and benches beside them. They take the model's sizes as arguments; a few keep a compiled maximum, such as the CPU
expert path's widths in `cpu/expert.hpp`, which `src/models/qwen38/session.cpp` checks against Qwen3.8's geometry.
Like the rest of the old engine code, they still call the CUDA runtime directly.

- `cuda/`: the GPU kernels: the S-family and i-quant GEMVs (`s_gemv`, `s2_gemv*`, `iq_kernels`, `native_mmvq`), the
  BF16 GEMV, the dequantizers, activation quantization, rope, the sampler, the MoE router (`router_top10`),
  elementwise operations. `coupled_draft.hpp` and `spec_prob.hpp` (from `include/strata/core/`) are the sampler's
  speculative modes.
- `cpu/`: the CPU expert kernels (AVX1, AVX2, AVX-512; Q2_0, the i-quants, the K-quants, NVFP4, the trellis types),
  the expert layout and the CPU expert pool.
- `prefill/`: the prompt path's GEMMs (cuBLAS / hipBLASLt / rocBLAS, RDNA3 WMMA) and its expert products (llama.cpp's
  MMQ, the fused int8 kernels, the `strata_kq` library).

All three engines use them: Qwen3.6 the GEMVs, the prompt GEMMs and the CPU pool, DeepSeek the sampler and the CPU
i-quant and trellis rows.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

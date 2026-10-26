# src/common/kernels/

Device kernels that work for any model: dequantization, quantized matrix-vector products, GEMM, the sampler, elementwise
operations, rope. Headers: `include/strata/common/kernels/`.

**B1** moves in the kernels of `src/kernels/` and `src/prefill/` that are not compiled for one model's shapes. **B2**
replaces DeepSeek's own quantization kernels with these.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

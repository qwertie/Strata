# src/models/deepseekv41/

The DeepSeek V4.1-Flash engine. Headers: `include/strata/models/deepseekv41/`.

**A5** and **B2** replace its own copies of shared code; A4 moved its OS services to
`include/strata/base/platform/os.hpp`. Its plain sampling is the shared `sample_tokens`; its quant kernels
(`ds4_iq`, `ds4_gguf`, `ds4_tq`), its draft sampling (`vendor/sampler.cu`) and its worker pool (`ds4_pool.hpp`)
stay its own (see the open issues in REFACTOR_PLAN.md).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

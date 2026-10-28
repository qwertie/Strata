# src/common/sample/

The sampler chain on the host side, and logit bias (`logit_bias.hpp`, from `src/core/`). Headers:
`include/strata/common/sample/`.

The rest is filled when the engines' sampling code is merged: DeepSeek has a copy of the shared sampler header
(`include/strata/models/deepseekv41/sampler.hpp`, removed in **A4**).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

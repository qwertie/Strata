# src/common/spec/

Speculative decoding: drafting, accepting and rejecting drafts, rolling back. Headers: `include/strata/common/spec/`.

Only the Qwen3.8 engine uses it. `window_planner.hpp` chooses what each verify window of its two decode loops
(`--serve` and `--spec`) holds: the MTP's drafts, a lookup window or a lookup chain. Qwen3.6's MTP steps and
DeepSeek's NextN drafts have no lookup and keep their own short loops.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

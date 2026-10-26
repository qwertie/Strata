# strata_common/

Strata-wide Python: GPU detection, installing and finding engines, the model family catalog, the GGUF reader, the engine
protocol client. Layer 1: it may import `strata_base`.

**A2** moves in `strata_control/{gpu_roles,engine_component,presets}.py` and `tools/gguf_reader.py`, and the setup code
that `setup.py` and `strata_control/` both have.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the layers and [docs/REFACTOR_PLAN.md](../docs/REFACTOR_PLAN.md)
for the steps.

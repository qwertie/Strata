# strata_common/

Strata-wide Python: GPU detection, installing and finding engines, the model family catalog, the GGUF reader, the engine
protocol client, the client of the server's status API. Layer 1: it may import `strata_base`.

**A2** moved in the modules that the server, Control and the Terminal share, and what they import:
`serve/{launch_identity,lifeline,loader_diag,logit_bias,ready,telemetry}.py`, the engine client from `serve/server.py`
(`engine_client.py`), 19 modules of `strata_control/` and its `update_keys.json`, `strata_cli/client.py` and
`tools/gguf_reader.py` (the path map in [docs/UPSTREAM_PORTING.md](../docs/UPSTREAM_PORTING.md#path-map) lists them).
Still to come in **A2**: the setup code that `setup.py` and `strata_control/` both have.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the layers and [docs/REFACTOR_PLAN.md](../docs/REFACTOR_PLAN.md)
for the steps.

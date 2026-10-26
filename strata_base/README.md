# strata_base/

All-purpose Python: processes, files, HTTP, hashing, downloads. Layer 0: it imports nothing of Strata's.

**A2** moves in `serve/{winjob,keepawake}.py`, `strata_control/{procs,tls,private_files}.py` and the helpers that
`setup.py` and `tools/` repeat.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the layers and [docs/REFACTOR_PLAN.md](../docs/REFACTOR_PLAN.md)
for the steps.

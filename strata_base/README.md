# strata_base/

All-purpose Python: processes, files, HTTP, hashing, downloads. Layer 0: it imports nothing of Strata's.

**A2** moved in the all-purpose modules that the other packages share: `serve/winjob.py` and
`strata_control/{atomic,longpath,loopback,posix_owner,private_files,tls,transfer,winproc}.py`, and the RAM probe
(`memory.py`) and path comparisons (`path_compare.py`) that `setup.py`, `tools/` and the packages each had a copy of.
The packages' own copies of a file write through a temporary file are now `atomic.write_bytes` and `atomic.write_text`.
Still to come in **A2**: the other helpers that `setup.py` and `tools/` repeat.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the layers and [docs/REFACTOR_PLAN.md](../docs/REFACTOR_PLAN.md)
for the steps.

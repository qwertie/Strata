# strata_base/

All-purpose Python: processes, files, HTTP, hashing, downloads. Layer 0: it imports nothing of Strata's.

**A2** moved in the all-purpose modules that the other packages share: `serve/winjob.py` and
`strata_control/{atomic,longpath,loopback,posix_owner,private_files,tls,transfer,winproc}.py`. Still to come in
**A2**: the helpers that `setup.py` and `tools/` repeat.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the layers and [docs/REFACTOR_PLAN.md](../docs/REFACTOR_PLAN.md)
for the steps.

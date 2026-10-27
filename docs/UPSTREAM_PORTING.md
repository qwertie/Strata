# Carrying upstream changes over to this fork

This fork is reorganized ([REFACTOR_PLAN.md](REFACTOR_PLAN.md), [ARCHITECTURE.md](ARCHITECTURE.md)), so a change
accepted upstream ([Niko1221/Strata](https://github.com/Niko1221/Strata)) often touches files that have moved,
been merged with a copy, or been split here. This page says where each upstream path lives now, and how to carry
a change over.

## Where we are

- The `upstream` remote is `https://github.com/Niko1221/Strata.git`.
- **Last upstream commit carried over: `4b9e05a2`** (2026-10-25, "Windows uninstaller: remove the app's own
  folders …"). Update this line whenever you finish a batch.

## Steps

1. `git fetch upstream`, then list what is new: `git log --oneline --reverse 4b9e05a2..upstream/main` (use the
   commit from the line above). Work through it one upstream pull request at a time.
2. Decide: **carry over**, **skip** (with a reason), or **already done** (the fork has its own version). Record the
   decision in the [log](#log).
3. Try git first: `git cherry-pick -x <commit>` (or `-m 1` for a merge commit). Git follows a file that was moved
   whole, because its content is unchanged. It cannot follow a file that was split, merged with a copy or moved
   into a renamed module; those hunks conflict or land in a deleted file.
4. For each hunk git could not place, look up its old path in the [path map](#path-map) and apply it by hand
   where the code is now. The [special cases](#special-cases) cover changes that touch copies.
5. Run the checks in [REFACTOR_PLAN.md](REFACTOR_PLAN.md#checks-for-every-step) that the change touches.
6. Commit with the upstream reference in the message (`-x` adds it), and update the log and the "last commit"
   line above.

## Special cases

- **A change to a file and its SYCL copy** (`src/X` and `sycl/src/X`): where the two are now one shared file,
  apply it once. Where the SYCL copy is still separate, apply it to both, as upstream did.
- **A change to only the SYCL copy** of a file that is now shared: decide whether it is a SYCL-only difference.
  If it is, it goes behind the SYCL side of the shared interface; otherwise apply it to the shared file.
- **A change to code that this fork merged** (for example DeepSeek's own sampler header, replaced by the shared
  one): check whether the change also applies to the shared code; often the shared code already has it.
- **A change to the 14,800-line `main()` in `src/program/generate.cpp`**: the code has moved into libraries
  under `src/common/` and `src/models/qwen38/`. Find it by a function or log-message name from the hunk.
- **Python imports** that changed with a module's move (`from serve import …`, `from strata_control import tls`)
  must be rewritten to the new package.

## Path map

Old paths are upstream's. Status is `planned` until the step is done, then `done`. Each refactor commit that moves
code updates its rows here. Rows are grouped by step; `{a,b}` lists several files of one folder.

| Upstream path | Path here | Step | Status |
|---|---|---|---|
| `serve/` (everything not listed below) | `strata_server/` | A2 | done |
| `serve/winjob.py`, `strata_control/{atomic,longpath,loopback,posix_owner,private_files,tls,transfer,winproc}.py` | `strata_base/` | A2 | done |
| `serve/{launch_identity,lifeline,ready,telemetry}.py` | `strata_common/` | A2 | done |
| `strata_control/{acquisition,draft_layer,drafting,engine_component,gpu_fit,gpu_roles,hipblaslt_tuning,offline,paths,presets,procs,profiles,signing,targets,tune_history,tune_knobs,tune_machine,user_args,vision_reader}.py`, `strata_control/update_keys.json` | `strata_common/` | A2 | done |
| the engine client in `serve/server.py` (`StrataEngine`, `MockEngine`, `engine_args`, `child_env`, the API-key helpers, the `Engine*` errors and the module values they read) | `strata_common/engine_client.py` | A2 | done |
| `serve/{loader_diag,logit_bias}.py` | `strata_common/` | A2 | done |
| `strata_cli/bench.py` | `strata_control/bench.py` | A2 | done |
| `_source`, `_redact_argv`, `_effective`, `_launch_plan` in `strata_cli/cli.py` | `strata_control/launch_plan.py`: `read_source`, `redact_argv`, `get_effective`, `make_launch_plan` | A2 | done |
| `strata_cli/client.py` | `strata_common/client.py` (the monitor bundle's schema 7 lists it there) | A2 | done |
| `tools/gguf_reader.py` | `strata_common/gguf_reader.py` (scripts import `strata_common.gguf_reader` and put the repository on `sys.path`) | A2 | done |
| helpers repeated in `setup.py` and `tools/*.py` (`sha256`, `http_json`, `get_json`, GPU detection, engine download) | `strata_base/`, `strata_common/` | A2 | planned |
| `src/platform/`, `include/strata/platform/` (except `hip_compat/`) | `src/base/platform/`, `include/strata/base/platform/` | A3 | planned |
| `include/strata/hip_compat/`, `include/strata/platform/hip_compat/` | `include/strata/base/gpu/hip_compat/` | A3 | planned |
| `src/core/{device.cu,pinned.cu,graph.cpp}` and their headers | `src/base/gpu/` | A3 | planned |
| `include/strata/core/{sm75_series,sm120_series,tc_sm75,emulate,gfx12_platform_defaults}.hpp`, `include/strata/kernels/gfx_arch.hpp` | `include/strata/base/gpu/` | A3 | planned |
| `src/artifact/`, `include/strata/artifact/` | `src/common/artifact/`, `include/strata/common/artifact/` | A3 | planned |
| `src/spec/`, `include/strata/spec/` | `src/common/spec/`, `include/strata/common/spec/` | A3 | planned |
| `src/core/conversation_*` and `include/strata/core/{conversation_*,token_listing,progress}.hpp`, except the snapshot files below | `src/common/state/` | A3 | planned |
| `src/core/{conversation_snapshot,conversation_state}.cpp`, `include/strata/core/conversation_snapshot.hpp` (they use Qwen3.8's session) | `src/models/qwen38/` until A5 makes them model-neutral | A3 | planned |
| `include/strata/core/{geometry,family_layout}.hpp` | `include/strata/models/common/` | A3 | planned |
| `src/dsv41/`, `include/strata/dsv41/` | `src/models/deepseekv41/`, `include/strata/models/deepseekv41/` | A3 | planned |
| `src/qwen35/`, `include/strata/qwen35/` | `src/models/qwen36/`, `include/strata/models/qwen36/` | A3 | planned |
| `src/ngram/`, `src/plan/` and their headers | `src/models/qwen38/` | A3 | planned |
| the Qwen3.8-only files of `src/core/` (one row per file once A3 decides) | `src/models/qwen38/` | A3 | planned |
| `sycl/src/X`, `sycl/include/strata/X` | the same new place as `src/X`, under `sycl/` | A3 | planned |
| `src/dsv41/ds4_os.hpp` | merged into `src/base/platform/` | A4 | planned |
| `src/dsv41/ds4_pool.hpp`, `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp` | one worker pool in `src/base/platform/` | A4 | planned |
| `include/strata/dsv41/sampler.hpp` | deleted; the shared `sampler.hpp` is used | A4 | planned |
| the protocol code in `src/program/generate.cpp`, `src/qwen35/q36_engine.cu`, `src/dsv41/ds4_main.cpp` | `src/common/serve/` | A5 | planned |
| DeepSeek's conversation slots (`src/dsv41/ds4_main.cpp`, `ds4_slotmeta.hpp`) | `src/common/state/` | A5 | planned |
| the generic parts of `src/core/{expert_cache,expert_source}.cpp`, `src/dsv41/ds4_store.hpp` | `src/common/experts/` | A5 | planned |
| 7 SYCL copies that are 95%+ the same as `src/` (listed in the A6 commit) | the shared `src/` file | A6 | planned |
| `src/kernels/`, `include/strata/kernels/` | split: `src/common/kernels/`, `src/models/common/`, `src/models/qwen38/` (one row per file once B1 decides) | B1 | planned |
| `src/prefill/`, `include/strata/prefill/` | split: `src/common/kernels/`, `src/models/qwen38/` | B1 | planned |
| `src/dsv41/{ds4_iq,ds4_gguf,ds4_tq}*` | deleted; the shared quant kernels are used | B2 | planned |
| `src/core/{peer_experts,remote_experts}.cpp` | `src/common/experts/` | B3 | planned |
| the other SYCL copies of `src/` files | the shared `src/` file | C1 | planned |

## Log

| Upstream | Decision | Here | Notes |
|---|---|---|---|

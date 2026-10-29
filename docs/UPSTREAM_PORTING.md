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
  apply it once. Where the SYCL copy is still separate, apply it to both, as upstream did. The files shared since A6
  are listed in `sycl/tools/normalize.sh` (`shared`): apply upstream's change to `src/X` or `include/strata/X` only,
  drop the hunks for the `sycl/` copy, and build SYCL. If the change uses a CUDA name the SYCL build lacks, add the
  name to `sycl/include/sycl_compat/cuda_runtime.h`; do not bring the copy back.
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
| the settings file's path: `settings_path` in `setup.py` and `tools/strata_mcp.py`; `app_settings_file` in `setup.py` | `strata_control/discovery.py`: `own_settings_file`, `in_app_mode` | A2 | done |
| `same_drive`, `same_folder` in `setup.py`; `same_drive`, `same_place` in `strata_cli/launch/library.py` | `strata_base/path_compare.py` | A2 | done |
| `repoint_config`, `library_size_gb` in `setup.py`; `library_gb` in `strata_cli/launch/library.py` | `strata_control/models_move.py`: `repoint_config`, `library_gb` | A2 | done |
| `model_config_files` in `setup.py`, `config_files` in `strata_control/models_move.py` | `strata_control/discovery.py`: `config_files` | A2 | done |
| the RAM probes: `_memory_status`, `ram_gb` in `setup.py`; `memory` in `strata_control/tune_machine.py`; `_CpuRamFallback.ram` in `serve/telemetry.py`; `ram_bytes` in `strata_cli/bench_facts.py`; `own_ram_gb` in `tools/strata_mcp.py`; `total_ram_gib` in `tools/strata_bench.py` | `strata_base/memory.py` (`read_ram`, `windows_memory_status`); the old names that remain call it | A2 | done |
| `_feed_source`, `DEFAULT_FEED`, `ENV_FEED` in `strata_cli/update_cmd.py` | `strata_control/updater.py`: `feed_source`, `DEFAULT_FEED`, `ENV_FEED` | A2 | done |
| the DeepSeek engine upgrade: `upgrade_dsv41_engine` in `setup.py` and the body of `Runner.upgrade_dsv41_engine` in `strata_cli/launch/runner.py` | `strata_control/dsv41_engine.py`: `try_upgrade` | A2 | done |
| the address test in `local_feed` in `setup.py` | `strata_base/transfer.py`: `is_local` | A2 | done |
| the model catalog in `setup.py` (`MODELS`, `FAMILIES`, `CONTEXTS`, `OWN_PATH_FAMILIES`, `MULTI_FILE_FAMILIES`, the `*_FILES` and `*_SHARDS` pins, `NVFP4_REPOS`, `REPO_PINS`, `HF_REVISIONS`, `hf`, `hf_endpoint`); its copies: `FALLBACK_*` in `tools/strata_mcp.py` and `strata_cli/launch/settings.py`, the sizes in `PINNED_PROJECTORS` in `strata_control/models.py` and the DeepSeek file sizes in `strata_control/dsv41_fit.py` | `strata_common/catalog.py` (`setup.py` imports the names it had) | A2 | done |
| `SPLIT_MIN_VRAM_GB`, `SPLIT_PROMPT_VRAM_GB` in `setup.py` | `strata_common/gpu_fit.py` | A2 | done |
| `src/platform/`, `include/strata/platform/` (except `hip_compat/`) | `src/base/platform/`, `include/strata/base/platform/` | A3 | done |
| `include/strata/hip_compat/`, `include/strata/platform/hip_compat/` | `include/strata/base/gpu/hip_compat/` (the wave32 HIP backend's) and `include/strata/base/gpu/hip_compat_gfx906/` (gfx906's): two different sets | A3 | done |
| `src/core/{device.cu,pinned.cu,graph.cpp}` and their headers | `src/base/gpu/` | A3 | done |
| `src/core/{vmm.cpp,pin_budget_test.cpp,pinned_shared_test.cpp}`, `include/strata/core/{vmm,pin_budget,pinned_ring,host_graph_launch,on_device,dma_batch}.hpp` | `src/base/gpu/`, `include/strata/base/gpu/` | A3 | done |
| `include/strata/core/{sm75_series,sm120_series,tc_sm75,emulate,gfx12_platform_defaults}.hpp`, `include/strata/kernels/gfx_arch.hpp` | `include/strata/base/gpu/` | A3 | done |
| `src/artifact/`, `include/strata/artifact/` | `src/common/artifact/`, `include/strata/common/artifact/` | A3 | done |
| `src/spec/`, `include/strata/spec/` | `src/common/spec/`, `include/strata/common/spec/` | A3 | done |
| `src/core/conversation_*` and `include/strata/core/{conversation_*,token_listing,progress}.hpp`, except the snapshot files below | `src/common/state/`, `include/strata/common/state/`; the snapshot code's helpers `src/core/{conversation_copy,conversation_checked}.hpp` and its tests `src/core/{conversation_snapshot,conversation_validation}_test.cpp` go with it to `src/models/qwen38/` | A3 | done |
| `include/strata/core/session_save_reclaim.hpp`, `src/core/{session_save_reclaim,token_listing}_test.cpp` | `include/strata/common/state/`, `src/common/state/` | A3 | done |
| `src/core/{conversation_snapshot,conversation_state}.cpp`, `include/strata/core/conversation_snapshot.hpp` (they use Qwen3.8's session) | `src/models/qwen38/` until A5 makes them model-neutral | A3 | done |
| `include/strata/core/{geometry,family_layout}.hpp` | `include/strata/models/common/`; `geometry.hpp` only once `include/strata/common/artifact/gguf_reader.hpp` and the shared kernels no longer use it | A3, B1 | `family_layout.hpp` done |
| `src/dsv41/`, `include/strata/dsv41/` | `src/models/deepseekv41/`, `include/strata/models/deepseekv41/` | A3 | done |
| `src/qwen35/`, `include/strata/qwen35/` | `src/models/qwen36/`, `include/strata/models/qwen36/` | A3 | done |
| `src/ngram/`, `src/plan/` and their headers | `src/models/qwen38/{ngram,plan}/`, `include/strata/models/qwen38/{ngram,plan}/` | A3 | done |
| the Qwen3.8-only files of `src/core/` (one row per file once A3 decides) | `src/models/qwen38/`, `include/strata/models/qwen38/`: `{layer,session,mtp,verify,native_dense,native_head,dflash2}` (source and header), `dflash2_{parity,attn_bench,gemm_bench,small_bench}.cpp`, `dflash2_{bw,launch}_bench.cu`, the headers `{adapt_async_default,batch_rows,batch_telemetry,draft_cap}.hpp` and their tests | A3 | done |
| `src/core/{device_main,load_main,concurrent_main,overlap_main}.cpp` | `src/program/` | A3 | done |
| `include/strata/core/logit_bias.hpp`, `src/core/logit_bias_test.cpp` | `include/strata/common/sample/`, `src/common/sample/` | A3 | done |
| `src/core/{arch_defaults,nv_defaults,weights,layout}.cpp`, `include/strata/core/{coupled_draft,spec_prob,uma_cache,foresight_swap,exchange_storage,hit_hook,remote_expert_opt}.hpp` and the other files left in `src/core/` | decided in A5, B1 or B3 (each is shared by several engines, or mixes model-neutral and Qwen3.8 code) | A5, B1, B3 | planned |
| `sycl/src/{platform,ngram}/` | `sycl/src/base/platform/`, `sycl/src/models/qwen38/ngram/` | A3 | done |
| the SYCL copies of the `src/core/` files above, `sycl/src/core/copy_queue.cpp` | the same new place under `sycl/`; `copy_queue.cpp` to `sycl/src/base/gpu/` | A3 | done |
| `sycl/src/X`, `sycl/include/strata/X` | the same new place as `src/X`, under `sycl/` | A3 | planned |
| `src/dsv41/ds4_os.hpp` | `include/strata/base/platform/os.hpp`, namespace `strata::platform` (`os::x` is `platform::x`); its `wide` is `widen_utf8`, which `direct_file.cpp`, `pinned.cu` and `expert_source.cpp` now use too | A4 | done |
| `src/dsv41/ds4_pool.hpp`, `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp` | one worker pool in `src/base/platform/`, once both engines' speed can be measured ([open issue](REFACTOR_PLAN.md#open-issues)) | A4 | planned |
| `include/strata/dsv41/sampler.hpp` | not a copy: the header of DeepSeek's own sampler kernels (`src/models/deepseekv41/vendor/sampler.cu`); merged with the shared sampler with the other kernels ([open issue](REFACTOR_PLAN.md#open-issues)) | B2 | planned |
| the protocol code in `src/program/generate.cpp`, `src/qwen35/q36_engine.cu`, `src/dsv41/ds4_main.cpp` | `src/common/serve/`. Done for `generate.cpp`: its stdin reader thread (`StdinShared`, `next_line`, `try_next_line`, `stop_req`) is `strata::serve::Input`; the `--serve` loop's handling of BSTOP, BYIELD, LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS is `Loop`, which gets Qwen3.8's part through `EngineHooks` set up just before the loop (`hooks.before_line` is the pipeline drain before a line); the scans of the queue for BSTOP and BYIELD during a prompt read are `Input::take_slot_controls`; the splitting of GEN / GENI / BGEN lines and of their sampling and `img_keys` keys is `parse_request_line`, `read_sampling_key` and `parse_hex_list`. Done for `q36_engine.cu` (`Serve`): its reader thread, queue and STOP flag are `Input` (`poll_control` calls `take_slot_controls`), `run()` is a `Loop` (`is_busy`, `step` = `window()`, `stop_slot`, `list_conversations`), `request()` reads its line with `parse_request_line`, its `Sampling` struct is `serve::Sampling` (its penalty fields are named `penalty_*`; `Sampling::needs_logits()` is the free function `needs_logits`) and its `img_keys` loop is `parse_hex_list`; `Engine::spec_step` takes the stop check as a `std::function` instead of an `std::atomic<bool>*`; `serve(Engine&)` is gone (`Serve(E).run()`) | A5 | `generate.cpp`, `q36_engine.cu` done |
| the session state of `src/qwen35/q36_engine.cu` (`alloc_session`, `copy_session`, the zeroing in `reset`, `sess_fixed_bytes` / `sess_cell_bytes`) | described as a `StateLayout` (`include/strata/common/state/state_layout.hpp`; `Engine::describe_session`) and handled by its `alloc_state`, `copy_state`, `clear_fixed_state` and `bytes_for` | A5 | done |
| DeepSeek's conversation slots (`src/dsv41/ds4_main.cpp`, `ds4_slotmeta.hpp`) | `src/common/state/`: the slot as a `StateLayout` per device (`src/common/state/README.md` says how) | A5 | planned |
| the generic parts of `src/core/{expert_cache,expert_source}.cpp`, `src/dsv41/ds4_store.hpp` | `src/common/experts/` | A5 | planned |
| 7 SYCL copies that are 95%+ the same as `src/` (listed in the A6 commit) | 8 copies deleted: `sycl/include/strata/{base/gpu/graph.hpp,kernels/f16_bits.hpp,kernels/bf16_bits.hpp,kernels/rope.hpp,kernels/rope_scaling.hpp,models/qwen38/session.hpp}`, `sycl/src/{base/platform/direct_file.cpp,models/qwen38/ngram/ple_reader.cpp}`; the SYCL build compiles the shared `include/strata/X` and `src/X`. The CUDA types the two headers name come from `sycl/include/sycl_compat/cuda_runtime.h`; the SYCL build's 64 reading threads in `direct_file.cpp` are `STRATA_DIRECT_FILE_PREAD_THREADS=64` in `sycl/CMakeLists.txt` | A6 | done |
| `src/kernels/`, `include/strata/kernels/` | split: `src/common/kernels/`, `src/models/common/`, `src/models/qwen38/` (one row per file once B1 decides) | B1 | planned |
| `src/prefill/`, `include/strata/prefill/` | split: `src/common/kernels/`, `src/models/qwen38/` | B1 | planned |
| `src/dsv41/{ds4_iq,ds4_gguf,ds4_tq}*` | deleted; the shared quant kernels are used | B2 | planned |
| `src/core/{peer_experts,remote_experts}.cpp` | `src/common/experts/` | B3 | planned |
| the other SYCL copies of `src/` files | the shared `src/` file | C1 | planned |

## Log

| Upstream | Decision | Here | Notes |
|---|---|---|---|

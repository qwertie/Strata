# How the code is organized

Strata's code is arranged in layers, from all-purpose code at the bottom to programs at the top. Code may use
the layers below it, never the ones above it. The layout is being introduced step by step
([REFACTOR_PLAN.md](REFACTOR_PLAN.md)); until it is complete, some code is still in the old folders listed
[at the end](#old-folders).

Folder names are chosen so that, in alphabetical order, shared code comes before the code that uses it.

## Two stacks that meet at run time

- **The engine** is C++ (with CUDA/HIP, SYCL and Metal for the GPU). Sources are in `src/`, headers in
  `include/strata/`, and the two mirror each other: the headers of `src/common/state/` are in
  `include/strata/common/state/`.
- **The programs around it** are Python packages at the top of the repository: the server, the `strata` command
  (the Terminal) and setup. They are at the top because they are import roots, and the release packages ship them
  under these names.
- Neither stack imports the other. The server starts an engine program (built from `src/program/`) and talks to
  it over the engine's standard input and output. What both sides must agree on is in `contracts/`.
- The desktop app (`apps/`) sits on top of both and is not part of these layers.

## The C++ layers

| Layer | Folder | What belongs here | May use |
|---|---|---|---|
| 0 | `src/base/` | All-purpose code with no inference knowledge | `third_party/` only |
| 0 | `src/base/gpu/` | The device runtime interface (memory, copies, streams, events, graphs), one implementation per backend; GPU hardware traits; the HIP compat headers | `src/base/platform/` |
| 0 | `src/base/platform/` | Operating-system services: files read at an offset, mapping, NUMA, thread pinning, free RAM, worker pools | nothing |
| 1 | `src/common/` | Inference code that does not depend on a model | layer 0 |
| 1 | `src/common/artifact/` | GGUF reading, quantization block formats, CPU reference dequantizers | |
| 1 | `src/common/experts/` | The expert store: RAM tier, VRAM slots, prefetch, eviction | |
| 1 | `src/common/kernels/` | Device kernels that work for any model | |
| 1 | `src/common/sample/` | The sampler chain, logit bias | |
| 1 | `src/common/serve/` | The engine protocol, request queue, scheduler, streaming, cancel | |
| 1 | `src/common/spec/` | Speculative decoding: drafting, accept/reject, rollback | |
| 1 | `src/common/state/` | Per-conversation state: the state descriptor, page pool, snapshots, swap, conversation files | |
| 2 | `src/models/common/` | Code shared by several model families: model geometry, tensor layout tables, GDN / gated attention / MoE-routing layers and their kernels | layers 0–1 |
| 3 | `src/models/<family>/` | One model family's engine: `deepseekv41`, `qwen36`, `qwen38` | layers 0–2; never another family |
| 4 | `src/program/` | One thin `main()` per program | layers 0–3 |

Inside `src/common/`, the subfolders may use each other as long as they do not form a cycle.

### Other engine backends

- `metal/` is the Mac engine. Its host code compiles the shared `src/` sources unchanged through
  `metal/include/metal_compat`; its kernels are `.metal` files.
- `sycl/` is the Intel engine. It is mostly translated copies of `src/` files. It takes the same layout as
  `src/`, and its copies become shared sources only where that can be tested ([REFACTOR_PLAN.md](REFACTOR_PLAN.md),
  steps A6 and C1).

## The Python layers

| Layer | Package | What belongs here | May import |
|---|---|---|---|
| 0 | `strata_base/` | All-purpose code: processes, files, HTTP, hashing, downloads | nothing of Strata's |
| 1 | `strata_common/` | Strata-wide code: GPU detection, installing and finding engines, the model family catalog (`catalog.py`: the families, sizes, contexts and file pins), the GGUF reader, the engine protocol client, the client of the server's status API | `strata_base` |
| 2 | `strata_server/` | The API server (OpenAI and Anthropic compatible) and its web page | layers 0–1 |
| 2 | `strata_control/` | The Terminal's library: launch, status, connect, benchmarks | layers 0–1 |
| 3 | `strata_cli/` | The `strata` command's interface | layers 0–2 |
| 4 | `setup.py`, `tools/` | Setup from a source checkout, developer scripts | any |

Packages in the same layer do not import each other: what they share moves down a layer.

## Language-neutral

`contracts/` holds what the C++ and Python sides (and the app) must agree on: the engine protocol and the model
family catalog. It depends on nothing.

## Where does new code go?

1. Would a program that is not an inference engine want it? Then `src/base/` or `strata_base/`.
2. Would every model use it? Then `src/common/` or `strata_common/`.
3. Do two or more model families use it? Then `src/models/common/`.
4. Otherwise, the family's folder in `src/models/`.

## Rules for new code

These replace practices that the old code has, and that new code must not copy:

- **Do not copy a file to make a variant.** Not into `sycl/`, not into another engine's folder. Share it, and
  put the difference behind an interface or a compat header (as `metal/include/metal_compat` does).
- **Code in `src/common/` and `src/models/` does not call the CUDA runtime (`cudaMalloc`, `cudaMemcpyAsync`, …)
  directly.** It uses `src/base/gpu/`, so that it can be built for every backend.
- **No settings in file-wide global variables** (as `g_kv_int8` and the other `qsa_set_*` switches in
  `src/core/layer.cpp`). Pass settings in a struct.
- **One settings reader.** Engines do not each parse their own environment variables.
- **Model-specific code does not go in a folder called `core` or `common`.**
- **`main()` parses its arguments and calls a library.** It is not the engine itself (as the 14,800-line
  `main()` in `src/program/generate.cpp` is).
- **No import or include cycles,** and no reaching up a layer, also not at run time (loading `setup.py` or a
  `tools/` script through `importlib` or `sys.path`). `tests/architecture/test_layering.py` checks the new folders;
  its `KNOWN_UPWARD_LOADS` lists the files that still do it.

## Old folders

These are emptied by the refactor and then deleted. Until then, code in the new folders may still use them. Where
each file goes is listed in [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md#path-map).

| Old folder | Goes to |
|---|---|
| `src/core/` | split: `src/base/gpu/`, `src/common/state/`, `src/common/experts/`, `src/models/common/`, `src/models/qwen38/` |
| `src/kernels/` | split: `src/common/kernels/`, `src/models/common/`, `src/models/qwen38/` |
| `src/prefill/` | split: `src/common/kernels/`, `src/models/qwen38/` |
| `serve/` | `strata_server/`; its all-purpose modules to `strata_base/` |
| parts of `strata_control/`, `setup.py`, `tools/` | `strata_base/`, `strata_common/` |

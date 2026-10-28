# src/common/serve/

The engine side of the line protocol that the Python server speaks (`strata_common/engine_client.py`). Headers:
`include/strata/common/serve/`. Library: `strata_serve` (no GPU).

- `loop.hpp`: `Input` reads the server's lines on a thread of its own (STOP raises a flag that a running request
  checks; the other lines wait in a queue). `Loop` gives the engine the commands that are its own and meanwhile runs
  the engine's busy work (batch slots decoding) and answers BSTOP, BYIELD, LIST_CONVERSATIONS and
  GET_CONTEXT_WINDOWS. An engine says what it does through `EngineHooks`.
- `request.hpp`: a request line (`GEN`, `GENI`, `BGEN`, `BGENI`) split into its parts, and the sampling keys every
  engine reads.

Each engine still handles its own requests, SAVE / RESTORE, VRAM and QUIT, and writes its own `T`, `PP`, `DONE`, `BT`
and `BDONE` lines.

## Which engine uses it

**Qwen3.8** (`src/program/generate.cpp`) and **Qwen3.6** (`Serve` in `src/models/qwen36/q36_engine.cu`) use all of
it. Qwen3.6 keeps two habits of its own: it skips empty lines, and it answers QUIT with `ERR expected: GEN ...` and
ends when stdin does (the server sends `STOP`, `QUIT` and then closes stdin). Its `Engine::spec_step` takes the stop
check as a function. **A5** moves DeepSeek onto it next. What it needs:

**DeepSeek V4.1** (`src/models/deepseekv41/ds4_main.cpp`):

- `Lines` becomes `Input`. `Lines` also takes `STOP <anything>` as STOP, and raises the stop flag at QUIT and when
  stdin ends: `Input` needs an option for that (a constructor flag, off for the Qwen engines, which queue such lines
  and answer them `ERR`), or DeepSeek drops it (the server sends a bare STOP, and STOP before QUIT). Its model's
  `set_cancel` takes `[&in] { return in.is_stop_requested(); }`, as Qwen3.6's `spec_step` does.
- Its loop becomes a `Loop` with `list_conversations` only (no batch slots).
- `parse_gen` becomes `parse_request_line` plus `read_sampling_key` (it sets `top_k = 64` in its `Sampling` before
  reading the keys; it reads `spec`, `dspark_conf`, `pin` and `dspark_cum` itself). DeepSeek takes the first word
  without `=` as the picture file and allows keys after it; `parse_request_line` stops reading keys at the file, so
  it needs a flag that reads `key=value` words after the file too.

Not shared yet: the `DONE` line has 15 fields and named ones (`chunk=`, ...) on Qwen3.8, 15 on Qwen3.6 and 10 on
DeepSeek; a shared writer would have to keep each engine's fields.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

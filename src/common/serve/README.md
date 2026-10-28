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

**Qwen3.8** (`src/program/generate.cpp`) uses all of it. **A5** moves the other two engines onto it. What each needs:

**Qwen3.6** (`Serve` in `src/models/qwen36/q36_engine.cu`):

- Its reader thread and queue become `Input`; `poll_control()` becomes `take_slot_controls`.
- `run()` becomes a `Loop` with `is_busy` = a slot is active, `step` = `window()`, `stop_slot`, and its session list
  as `list_conversations`. It keeps its own skip of empty lines.
- Two differences: it runs one window after every line it reads, where `Loop` runs a step only while no line waits
  (the order of `BT` lines among other answers can change, not their content); and it answers QUIT with `ERR
  expected: GEN ...`, then ends when stdin closes.
- `Engine::spec_step` takes the stop flag as `std::atomic<bool>*`: `Input` needs to hand out its flag, or
  `spec_step` a function.
- `request()` reads the line with `parse_request_line`, its `Sampling` struct becomes `serve::Sampling` (the
  penalty fields are named `penalty_*`; its `needs_logits()` becomes a free function) and its `img_keys` loop
  becomes `parse_hex_list`.

**DeepSeek V4.1** (`src/models/deepseekv41/ds4_main.cpp`):

- `Lines` becomes `Input`. `Lines` also takes `STOP <anything>` as STOP, and raises the stop flag at QUIT and when
  stdin ends: `Input` needs an option for that, or DeepSeek drops it (the server sends a bare STOP, and STOP before
  QUIT).
- Its loop becomes a `Loop` with `list_conversations` only (no batch slots).
- `parse_gen` becomes `parse_request_line` plus `read_sampling_key` (its `top_k` default is 64; it reads `spec`,
  `dspark_conf`, `pin` and `dspark_cum` itself). DeepSeek takes the first word without `=` as the picture file and
  allows keys after it; `parse_request_line` stops reading keys at the file.

Not shared yet: the `DONE` line has 15 fields and named ones (`chunk=`, ...) on Qwen3.8, 15 on Qwen3.6 and 10 on
DeepSeek; a shared writer would have to keep each engine's fields.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.

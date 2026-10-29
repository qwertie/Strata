# The code-sharing refactor

This fork reorganizes Strata so that code is written once and shared: between the three model engines
(Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, DeepSeek V4.1-Flash), between the GPU backends (CUDA, HIP, SYCL, Metal)
and between the Python programs. The goal is less code to read, change and test. Success is measured in lines
of code (below).

- The target layout and its rules: [ARCHITECTURE.md](ARCHITECTURE.md).
- Which old path goes where, and how to carry an upstream change over: [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md).

## Measuring

`python tools/loc_report.py` counts the non-blank lines of Strata's own code at a git revision. It leaves out
copied third-party code (`third_party/`, `*/vendor/`, Intel's `sycl/include/dpct/`), community results
(`bench/results/`) and the desktop app (`apps/`, which this work does not touch).

**Baseline:** `06fcbe98`, the last commit before this work: **493,148 lines** (C++ 249,138, Python 155,555,
CUDA 72,947, other 15,508).

After each step: `python tools/loc_report.py --compare 06fcbe98 HEAD` (add `--detail` for one row per folder).
Record the change in the [results](#results) table.

## Order

Steps are ordered by **risk first, lines saved second**. Moving a file without changing it is safe at any time,
because a broken include fails the compile checks. Changes to what NVIDIA-only code does wait for one batched test
session on an NVIDIA machine. Changes nobody can test come last.

"Est." is a rough estimate of lines saved, from file sizes and duplicate counts. It is not a measurement.

### Phase A: fully testable on the owner's PC (AMD Strix Halo, Windows)

| Step | Work | Est. | Status |
|---|---|---:|---|
| A0 | `tools/loc_report.py`, the baseline, this plan | — | done |
| A1 | Skeleton: the new folders with a README each, [ARCHITECTURE.md](ARCHITECTURE.md), [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md), the layering test | — | done |
| A2 | Python: `strata_base/`, `strata_common/`; `serve/` becomes `strata_server/`; remove the import cycles between `serve/`, `strata_control/` and `strata_cli/`; one copy of the helpers repeated in `setup.py` and `tools/` | 2–5K | |
| A3 | Move every whole file whose new place is already clear, without changing it: the folders `src/platform`, `src/artifact`, `src/spec`, `src/dsv41`, `src/qwen35`, `src/ngram`, `src/plan`; from `src/core/`, the runtime files to `src/base/gpu`, the conversation files to `src/common/state` and the Qwen3.8-only files to `src/models/qwen38`; the matching files in `sycl/`. Files that must be split wait for their step. | 0 | |
| A4 | DeepSeek's own copies of host code: `ds4_os.hpp`, `ds4_pool.hpp`, `include/strata/models/deepseekv41/sampler.hpp` | ~1K | `ds4_os.hpp` done; the pool and the sampler wait (open issues) |
| A5 | The `src/base/gpu` runtime interface (the shared code must not call the CUDA runtime directly), then `src/common/state`, `src/common/serve`, `src/common/spec` and the one-GPU parts of `src/common/experts`. These are also the base for the planned paged-KV engine. | 3–7K | serve: Qwen3.8 and Qwen3.6 done, DeepSeek waits for its model; state: descriptor done, Qwen3.6 and Qwen3.8's running state on it, Qwen3.8's K/V left |
| A6 | SYCL: the 7 files that are 95% or more the same as their `src/` original become shared | ~2.4K | done: 8 files (see [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md)) |

### Phase B: changes NVIDIA-only code; one batched test session on an NVIDIA machine

| Step | Work | Est. | Status |
|---|---|---:|---|
| B1 | Split `src/kernels/` and `src/prefill/` by what they depend on; remove Qwen's `geometry.hpp` from the 12 shared kernels | small | |
| B2 | DeepSeek's own quant kernels (`ds4_iq`, `ds4_gguf`, `ds4_tq`) and its sampler (`vendor/sampler.cu`) replaced by the shared ones | 3–5K | |
| B3 | The multi-GPU parts of `src/common/experts` (`peer_experts`, `remote_experts`) | ~0.5K | |
| B✓ | Test session on an RTX 30-series machine with several cards: parity tests on sm_86, sm_75/80 through `STRATA_EMULATE_CC`, multi-GPU, a DeepSeek speed comparison. Ada (sm_89) and Blackwell (sm_120, NVFP4) stay untested unless such a card is found. | | |

### Phase C: nobody here can test it (optional)

| Step | Work | Est. | Status |
|---|---|---:|---|
| C1 | The other 77 SYCL copies of `src/` files (31 are 75–95% the same, 46 less) become shared | 10–25K | |
| C2 | The SYCL kernels generated from the `.cu` sources instead of kept as translated copies | unknown | |

Do Phase C only if someone with an Intel Arc card can test it.

## Checks for every step

| Check | Where | How |
|---|---|---|
| Python tests | this PC | `.venv/Scripts/python -m pytest -n 16 tests/terminal tests/architecture strata_server --ignore=tests/terminal/test_lifecycle_matrix.py` (that file leaves Monitor windows open when its flaky tests fail; run it on its own), and the tests in `tools/`, `tests/desktop`, `tests/sycl`, `tests/time_skip` and `tests/tuning` one file at a time (`for f in $(git ls-files 'tools/test_*.py' 'tools/**/test_*.py' 'tests/desktop/test_*.py' 'tests/sycl/test_*.py' 'tests/time_skip/test_*.py' 'tests/tuning/test_*.py'); do .venv/Scripts/python -m pytest -q "$f"; done`: pytest-xdist cannot collect `tools/` as a whole). pytest and pytest-xdist are installed in the repo's `.venv`. At the baseline, about 75 tests in `tests/terminal` and 12 in `tools/` already failed on this PC (process handling on Windows, a missing llama.cpp `gguf-py`), and a few time-sensitive ones fail now and then under `-n 16`, so compare the failures by name with a run of the previous commit |
| HIP build and GPU tests | this PC | `build-hip-win` (gfx1151, tests on): build the targets the step touches, run their `*_parity` and `*_test` programs |
| CUDA compile | Docker on this PC | the command below; about 3 minutes after the first build |
| SYCL compile | Docker on this PC | the command below; about 4 minutes |
| Speed | this PC | decode and prompt speed within noise of a same-day run of the previous commit, for any step that touches engine code |

The CUDA image is made once with an inline Dockerfile (`nvidia/cuda:13.0.0-devel-ubuntu24.04` plus
build-essential, git, python3-venv, cmake 4.4.3 and ninja 1.13.2), tagged `strata-cuda-dev`. The SYCL image is
the repo's own: `docker build --pull -t strata-sycl-dev sycl/tools`. Both build into Docker volumes, so nothing is
written to the checkout. From Git Bash:

```sh
# CUDA: every target, both NVIDIA code families (sm_75, sm_120), the DeepSeek engine
MSYS_NO_PATHCONV=1 docker run --rm -v "C:/AI-models/Strata:/work/Strata" -v strata-cuda-build:/build strata-cuda-dev bash -c '
  cd /work/Strata && ( [ -f /build/build.ninja ] || cmake -G Ninja -S . -B /build -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
    -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON "-DCMAKE_CUDA_ARCHITECTURES=75-real;120-real;120-virtual" \
    -DSTRATA_GGML_DIR=/work/Strata/third_party/llama.cpp ) && cmake --build /build -j 32 && cmake --build /build --target strata_kq'

# SYCL: the engine, JIT (no Intel card needed to compile)
MSYS_NO_PATHCONV=1 docker run --rm -v "C:/AI-models/Strata:/work/Strata" -v strata-sycl-build:/build \
  -e REPO=/work/Strata -e BUILD_DIR=/build strata-sycl-dev "cd /work/Strata && bash sycl/tools/build.sh strata"
```

A move that changes CMake paths can leave a stale build in a volume: delete it with `docker volume rm
strata-cuda-build` (or `strata-sycl-build`) and build again.

Metal is checked only by the `metal-build` GitHub workflow, because it needs a Mac.

## Rules for each step

- One subsystem per commit, so that a reviewer can read the work commit by commit.
- A commit that fills a new folder also updates that folder's README and the path map in
  [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md).
- The commit message says what the step changed in `loc_report.py --compare`.
- Upstream's [CONTRIBUTING.md](../CONTRIBUTING.md) rules about byte-identical default paths and agreeing a
  refactor in an issue first do not apply to this fork; the scope will be raised in an upstream issue.

## Open issues

Found during the work; each needs a decision or a later step.

- **Two rules for the model folder.** `setup.py` (started by `START-HERE.bat`) picks the data folder with
  `data_folder` (default `Strata-data` beside the checkout's parent) and ignores the app's saved choice and
  `Documents/Strata`; `strata_control/discovery.py` (`data_dir`) reads both. A source install and the app can therefore
  use different model folders. `strata launch` is not affected (it always passes `--data-dir`). The owner wants one
  rule; not yet scheduled.
- **Ready-made engines in a source checkout** (deferred to phase B, needs NVIDIA testing and an owner decision):
  `setup.py` still downloads GitHub zips into `engine/` and keeps `engine/.previous`, while the Terminal and the app use
  the signed feed and the per-user components folder. Unifying them would remove about 600–850 lines, but drops
  `setup.py --prebuilt <mirror or dist folder>` and may affect the CUDA 12 engine.
- **A test that downloads for real.** `tools/test_setup_dsv41.py::*::test_no_engine_stops_before_the_download` (4
  tests) reaches the live signed feed; since a DeepSeek engine 0.1.52 package for Windows AMD appeared there, each run
  downloads it (137 MB) and the tests fail, at the baseline commit too. The test needs a local fake feed.
- **A third AMD reader.** `strata_common/telemetry.py` has its own Windows display-adapter reader, which works
  differently from `engine_component`'s (it keeps cards with 2 GB or more and reads the vendor from the provider name).
- **SYCL's upstream merge script.** `sycl/tools/merge_upstream.py` maps upstream paths straight onto `sycl/`; after the
  A3 moves the paths no longer line up, so a merge needs the path map in [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md).
- **Two CPU worker pools, not merged in A4.** They share an idea (one 64-bit claim word tagged with the batch, so a
  late worker cannot take a job of the next batch; spin, then sleep), but differ where the speed is decided:

  | | DeepSeek `ds4::Pool` (`ds4_pool.hpp`) | Qwen `ExpertPool` (`src/kernels/cpu/pool.cpp`) |
  |---|---|---|
  | Jobs | any `fn(i)` through `std::function`; `start`/`help`/`wait` run two pools (two NUMA nodes) at once | expert jobs in 7 modes (whole experts, row parts, native rows), one scratch per worker |
  | Claim | `generation << 32 \| index`, the job count in its own atomic | `epoch \| njobs \| index`; per NUMA slice with `STRATA_NUMA_SPLIT` |
  | End of a batch | the caller waits for `done` | the host waits for `done` and for every worker to park |
  | Idle workers | spin 200 µs, then sleep | spin 20 ms (`STRATA_POOL_SPIN_US`), then sleep |
  | Placement | the caller's CPU list, hard affinity, Windows processor group 0 | its own: topology, hybrid cores, `--pool-cores`, `--host-core`, processor groups |
  | Stalls | none | 60 s without progress stops the engine through the watchdog hooks of `include/strata/common/state/progress.hpp` |

  A merge design: a `WorkerPool` in `src/base/platform/` with the threads, the pinning, the tagged claim, the spin
  time as a setting, an optional park barrier and a stall callback the owner installs (so the watchdog hooks stay in
  layer 1: `ExpertPool` installs one that prints `diag` and calls `release_gpu_waits`). `ExpertPool` keeps its modes,
  scratch and NUMA slices on top of it, and `ds4::Pool` becomes a thin `run(n, fn)` with a 200 µs spin and no
  barrier. It needs a DeepSeek and a Qwen3.8 decode and prompt speed run before and after, which this PC could not
  give for A4 (no DeepSeek model here, and the Qwen server owned the GPU).
- **DeepSeek's sampler is its own.** `include/strata/models/deepseekv41/sampler.hpp` is not a copy of
  `include/strata/kernels/sampler.hpp`: it declares `src/models/deepseekv41/vendor/sampler.cu` (581 lines), whose
  `sample_tokens` is engine 0.1.20's one-block kernel (kept in the shared `sampler.cu` behind
  `STRATA_OLD_SAMPLER=1`), plus speculative sampling for the NextN drafts (`SpecList`, `sample_draft`,
  `sample_verify`, `DS4_DSPARK_SAMPLE=1`), which the shared sampler lacks. The shared one has `gumbel`, the cluster
  argmax and the coupled-draft functions instead. Its `host_history` field is unused. Merging means porting the
  draft and verify modes into the shared `sampler.cu`: kernel work, tested on NVIDIA with the rest of B2.
- **Qwen3.6 drafts nothing at temperature 0.** The server adds the model card's `presence_penalty: 1.5` for
  `qwen35moe` (`MODEL_CARD_SAMPLING` in `strata_server/server.py`); any penalty sends the Qwen3.6 engine down its
  sampling path, which skips MTP drafting (`src/models/qwen36/q36_engine.cu`). Measured 2026-10-28 on the Strix Halo,
  UD-Q4_K_XL, 2,000-token prompt: 47 tok/s decode with the penalty, 65 tok/s with `presence_penalty: 0`. Older than the
  refactor; may be intended.
- **A Qwen3.6 prompt read that never ended.** On 2026-10-28, in one of four server runs of the build before
  `9f3ac8a3` (UD-Q4_K_XL, Strix Halo), a greedy 2,031-token prompt read gave no output for 341 s, after the same
  request had taken 2.5 s in the run before; the server ended the engine (`engine_silence_s`).
  Not seen again; cause unknown.

## Running engines and GPU tests on this PC

Windows loads the driver's `System32mdhip64_7.dll` before anything on `PATH`, so an engine or GPU test run from
`build-hip-win` gets the wrong HIP runtime (the engine exits with 0xC0000139; tests hang or crash). Copy the program
into a folder together with `engine\*.dll` (as `engine\strata.exe` is installed) and run it there. End-to-end checks
use a copy of the run config with `"exe"` pointing at the engine under test and the server on another port (8091), with
alternating runs of the two builds. Models on this PC: Qwen3.8 IQ3_S (`strata-iq3_s.json`) and Qwen3.6 UD-Q4_K_XL
(`C:\AI-models\Strata-data\models\qwen36-UD-Q4_K_XL`); DeepSeek V4.1 is being downloaded (2026-10-28).

## Results

| Step | Commit | Lines after | Change from baseline | Notes |
|---|---|---:|---:|---|
| baseline | `06fcbe98` | 493,148 | — | |
| A0 | `e56c4aed` | 493,264 | +116 | the line counter itself |
| A1 | `d757595f` | 493,367 | +219 | the layering test |
| A2 | `f743b698` | 493,434 | +286 | moves only: imports split where some names moved |
| A2 | `6333ec2a` | 493,526 | +378 | the engine client moved out of `serve/server.py` (8,901 → about 6,100 lines) |
| A2 | `27d2e2eb` | 493,565 | +417 | last upward Python imports removed (launch plan, bench, status client, gguf_reader) |
| A2 | `5d272302` | 493,565 | +417 | `serve/` renamed `strata_server/` |
| A2 | `f63a0488` | 493,170 | +22 | one copy of each duplicated Python helper (−395) |
| A2 | `4e991378` | 493,009 | −139 | setup.py uses the shared settings, file, RAM and DeepSeek helpers |
| A2 | `906b00b2` | 492,982 | −166 | model catalog moved to `strata_common/catalog.py` |
| A2 | `87d20a5c` | 492,885 | −263 | setup.py's AMD detection built on `engine_component` |
| A3 | `d1b2c555` | 492,894 | −254 | whole C++ folders moved (moves only) |
| A3 | `0bb91482` | 492,909 | −239 | 98 files moved out of `src/core` (moves only) |
| A4 | `874ed84c` | 492,906 | −242 | DeepSeek's OS helpers moved to `base/platform`; pools not merged (see open issues) |
| A6 | `89a96beb` | 491,154 | −1,994 | 8 SYCL copies replaced by the shared sources (−1,752) |
| A5 | `b17c98fd` | 491,394 | −1,754 | the shared command loop (`src/common/serve`), Qwen3.8 on it |
| A5 | `57a2e1e0` | 491,316 | −1,832 | Qwen3.6 on the shared command loop |
| A5 | `9f3ac8a3` | 491,505 | −1,643 | the state descriptor (`state_layout.hpp`), Qwen3.6 on it |
| A5 | `27083334` | 491,519 | −1,629 | Qwen3.8's running state on the descriptor (its K/V not yet) |

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
| A4 | DeepSeek's own copies of host code: `ds4_os.hpp`, `ds4_pool.hpp`, `include/strata/dsv41/sampler.hpp` | ~1K | |
| A5 | The `src/base/gpu` runtime interface (the shared code must not call the CUDA runtime directly), then `src/common/state`, `src/common/serve`, `src/common/spec` and the one-GPU parts of `src/common/experts`. These are also the base for the planned paged-KV engine. | 3–7K | |
| A6 | SYCL: the 7 files that are 95% or more the same as their `src/` original become shared | ~2.4K | |

### Phase B: changes NVIDIA-only code; one batched test session on an NVIDIA machine

| Step | Work | Est. | Status |
|---|---|---:|---|
| B1 | Split `src/kernels/` and `src/prefill/` by what they depend on; remove Qwen's `geometry.hpp` from the 12 shared kernels | small | |
| B2 | DeepSeek's own quant kernels (`ds4_iq`, `ds4_gguf`, `ds4_tq`) replaced by the shared ones | 3–5K | |
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
| Python tests | this PC | `.venv/Scripts/python -m pytest -n 16 tests/terminal tests/architecture serve` (`serve/` becomes `strata_server/` in A2b), and the tests in `tools/`, `tests/desktop`, `tests/sycl`, `tests/time_skip` and `tests/tuning` one file at a time (`for f in $(git ls-files 'tools/test_*.py' 'tools/**/test_*.py' 'tests/desktop/test_*.py' 'tests/sycl/test_*.py' 'tests/time_skip/test_*.py' 'tests/tuning/test_*.py'); do .venv/Scripts/python -m pytest -q "$f"; done`: pytest-xdist cannot collect `tools/` as a whole). pytest and pytest-xdist are installed in the repo's `.venv`. At the baseline, about 75 tests in `tests/terminal` and 12 in `tools/` already failed on this PC (process handling on Windows, a missing llama.cpp `gguf-py`), and a few time-sensitive ones fail now and then under `-n 16`, so compare the failures by name with a run of the previous commit |
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

## Results

| Step | Commit | Lines after | Change from baseline | Notes |
|---|---|---:|---:|---|
| baseline | `06fcbe98` | 493,148 | — | |
| A0 | `e56c4aed` | 493,264 | +116 | the line counter itself |
| A1 | `d757595f` | 493,367 | +219 | the layering test |
| A2 | `f743b698` | 493,434 | +286 | moves only: imports split where some names moved |
| A2 | `6333ec2a` | 493,526 | +378 | the engine client moved out of `serve/server.py` (8,901 → about 6,100 lines) |

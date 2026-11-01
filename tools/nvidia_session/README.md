# The NVIDIA test session

The code-sharing refactor ([docs/REFACTOR_PLAN.md](../../docs/REFACTOR_PLAN.md)) changed engine code that was only
tested on an AMD card. `run.ps1` checks it on an NVIDIA PC in one go: it builds the engines as they were before the
refactor (commit `87d20a5c`) and as they are now, runs both the same way, and writes a table of what matches. It was
written for a PC with an RTX 3060 (12 GB), an RTX 3080 (20 GB) and 48 GB of RAM on Windows 11. Nobody has to watch it
after it starts.

## Before the day

Install, in this order:

1. The newest NVIDIA driver (580 or newer: CUDA 13 needs it).
2. [CUDA Toolkit 13](https://developer.nvidia.com/cuda-downloads): 13.0 is what the release engines and the plan's
   Docker check are built with; the first session built with 13.4. It sets `CUDA_PATH`.
3. Visual Studio 2022 Build Tools with the workload "Desktop development with C++". CUDA 13 does not take
   Visual Studio 2026.
4. Git for Windows, and Python 3.12 or newer (64-bit, from python.org; the first session used 3.14).
5. This fork, with its `local` branch, and Python's packages for it (`cmake` 4.4.3 and `ninja` 1.13.2 come from
   `requirements-build.txt`):

   ```powershell
   git clone <the fork> Strata; cd Strata; git checkout local
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt -r requirements-build.txt
   ```

Bring the model files on a fast (NVMe) drive, in the same folders as in your `Strata-data` (about 340 GB):

| Folder in `-DataDir` | Size | Used by |
|---|---:|---|
| `models\IQ3_S\` (both `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000?-of-00002.gguf`) | 78 GB | Qwen3.8 checks |
| `packs\iq3_s\` | 1.5 GB | Qwen3.8 checks |
| `mtp\rt\` | 0.8 GB | the drafter; without it Qwen3.8 runs without drafting |
| `models\qwen36-UD-Q4_K_XL\` | 22 GB | Qwen3.6 checks |
| `packs\deepseek-v4.1-flash-ist-das-2.00bpw\` | 9 MB | DeepSeek checks |
| `-DeepSeekDir`: the DeepSeek folder (5 IQ2_XXS shards, the Engram, the DSpark drafter) | 240 GB | DeepSeek checks |

A check whose files are missing is skipped, not failed.

**What fits.** Qwen3.8 IQ3_S's experts are 50.3 GB. By setup's own rule (`low_ram_needed` / `low_ram_fits` /
`low_ram_resident` in `setup.py`) 48 GB of RAM puts it in the low-RAM mode: with the 20 GB card about 14 GB of experts
sit in graphics memory and the other ~36 GB are copied into RAM; with the 12 GB card alone it does not fit. So the
one-card checks run on the card with the most memory, and the multi-GPU checks on both. Qwen3.6 UD-Q4_K_XL (22 GB)
keeps the experts that do not fit the card on the CPU. DeepSeek's guide asks for 40 GB of RAM; it runs on the bigger
card.

## Running it

From the checkout, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File tools\nvidia_session\run.ps1 -DataDir E:\Strata-data -DeepSeekDir E:\DeepSeek-V4.1-Flash
```

- `-Quick`: one run a build, the main variants only, sm_75 only for the emulated builds: about half the time.
- `-Only multi,speed`: only these phases. `-Results <folder>`: continue in an earlier results folder; runs that
  finished are kept, so a `-Quick` session can be filled in later by a full one into the same folder.
- `-DryRun`: prints every command and runs none.
- Builds go to `C:\sn` (`-Work`): two git worktrees and up to six build folders (a CUDA build of every target
  took 2.2 GB on Linux). Before the first build, `run.ps1` checks each worktree out again at its commit, applies
  `portability.patch` (test sources Visual Studio does not compile at the older commit) and copies llama.cpp's
  `gguf-py` from this checkout into it. A target that does not compile no longer stops the others; the build's row
  names it.

| Phase | What | Default | `-Quick` |
|---|---|---:|---:|
| preflight | Windows, cards, driver, PCIe links, peer-to-peer table, tools, RAM, page file, disks | 1 min | 1 min |
| build | both versions, every target, `86-real`, `STRATA_BUILD_DSV41`, `STRATA_NATIVE_EXPERTS`, tests | 1-3 h | 1-3 h |
| unit | `ctest` in both builds, test by test | 20-40 min | 20-40 min |
| smoke | the engines' protocol: Qwen3.8 (commands, batch slots, SAVE / RESTORE across the builds), Qwen3.6 | 45 min | 25 min |
| multi | the B3 list: layer split (auto, 12) in both card orders, peer tier, helper caches, `--vram-elastic`, `--kv-grow`, page-locked RAM copy | 3-4 h | 1 h |
| deepseek | the B2 list: greedy and seeded answers without the expert cache, DSpark on, off and sampled; decode speed | 2 h | 45 min |
| speed | the server on port 8091 with Qwen3.8: answers and prompt / decode tokens/s, 3 runs a build | 30 min | 20 min |
| emulate | builds of PTX for sm_75 and sm_80 run with `STRATA_EMULATE_CC`: the GPU tests and the protocol checks | 3-5 h | 2 h |
| memory | not in the default list: [the memory manager's checks](#the-memory-managers-checks) | 2-3 h | 2-3 h |

The times are estimates (not measured: they depend on the CPU and the drive); the summary records the real ones.
The emulated builds run last because they are the longest and the least likely to find something. Under
`STRATA_EMULATE_CC=80` the host asks for the A100's 163 KB of shared memory, which these cards do not have, so a test
may fail there in both builds; `decode_cluster_parity` passes without comparing anything below sm_90.

## The follow-up session

The first session (30 October) left a few checks open: answers that changed in one run of the build after, and
decode speeds measured once. `-Followup` runs only those, 4 times a build (`-Runs N` for another count): both builds
again (now with every test), the unit tests, `multi-helper` and `multi-peer` in both card orders,
`multi-split-auto-res` and `multi-split-12-res`, the server speed check (5 runs a build), and `smoke-q38state` on the sm_75 build.

```powershell
powershell -ExecutionPolicy Bypass -File tools\nvidia_session\run.ps1 -Followup -DataDir G:\AI-stuff\Strata-data -Work "Z:\C++\strata-builds"
```

| Part | Estimate |
|---|---:|
| builds at sm_86 (the first session's worktrees and build folders are reused) | 1-2 h |
| unit tests, both builds | 20-40 min |
| `multi-helper` and `multi-peer` (BS, SB), `multi-split-auto-res`, `multi-split-12-res` (BS): 48 runs of about 3.7 min | 3 h |
| server speed: 10 runs of about 3.4 min | 35 min |
| sm_75 builds, then `smoke-q38state-75v`: 8 runs of about 2.9 min | 1-1.5 h |

About 6-7 hours in all. The run times are the first session's; the build times are guesses. Send back the zip it
names at the end.

The follow-up's multi-GPU and speed runs, in both builds, also print where a decode's time went (`STRATA_TRACE=1`,
`STRATA_SPLIT_TIMING=1`, `STRATA_DECODE_TIMING=1`: a few lines a request, none a token). Under each of those checks'
row, a line per build gives the medians of its runs: the ms of a decode window and how much of it waited for the
GPUs, each card's wait for its GPU and its CPU pool time per window, the shares of the routed experts that CUDA0's
cache, another GPU or PCIe, and the CPU computed, and how many experts each card holds. A slower build then shows
where it lost the time. The lines stay in the raw logs and are left out of the comparison.

## The memory manager's checks

On the `memory-manager` branch, the `memory` phase checks the new Qwen3.8 engine, `strata-q38`
([docs/MEMORY_MANAGER_PLAN.md](../../docs/MEMORY_MANAGER_PLAN.md), steps M7 to M9). It compares against the build of
M7 (`4fa1d1cf`), so it takes `-Before 4fa1d1cf`, and needs both builds:

```powershell
git checkout memory-manager
powershell -ExecutionPolicy Bypass -File tools\nvidia_session\run.ps1 -Only preflight, build, memory -Before 4fa1d1cf -DataDir E:\Strata-data
```

`strata-q38` gets `--ram-mib 20480`: its RAM pool is page-locked and by default takes what RAM has free, which on 48 GB
leaves Windows' commit limit too little. Each check runs 2 times a build (`-Runs N` for another count).

| Check | What it compares | It confirms |
|---|---|---|
| `memory-q38-1card` | `strata-q38` on the bigger card, M7's build against this one | its answers and speed did not change (it now measures the card's link at start and paces the experts it puts back by it) |
| `multi-split-auto-SB`, `-BS` | the old engine's `--layer-split auto`, both card orders, M7's build against this one | the old engine is unchanged now that its split planner is shared code (`src/common/experts/layer_split.cpp`): the same answers, and the same `layer split auto: K=...` line (the startup lines are compared too) |
| `memory-speed-split-q38` | the old engine's `--layer-split auto` on both cards ("before") against `strata-q38` on the bigger card ("after") | today's split's speed against the new engine's; DIFF here only means `strata-q38` decoded more than 5% slower |
| `memory-q38-2cards-SB`, `-BS` | `strata-q38` on the bigger card against `strata-q38 --layer-split auto` on both cards | the same tokens on two cards as on one (step M8) |
| `memory-cut-move-SB`, `-BS` | `strata-q38` on both cards, a 96-token story, against the same with the first card's memory halved after 24 tokens and set back after 48 (`CHANGE_SETTINGS`; the answers to the changes go to `*.out.settings`) | a cut moved while decoding gives the same tokens (step M9) |
| `memory-rows-card`, `-ram`, `-ssd` | `tools\measure_rows.py` on the bigger card: as many experts on the card as fit, most of them in RAM, about half on the SSD | M7's table of conversations at once and rows per step on NVIDIA (docs/DETAILS.md has the Strix Halo's numbers). Needs `strata-iq3_s.json`, the run config setup writes; skipped without it |

Not done yet: `strata-q38` still runs on one card, so the two-card checks are recorded as SKIP ("strata-q38
refuses a layer split") until step M8's engine part is built; the kit runs them as soon as it is. The old engine's
checks and `memory-q38-1card` run now.

**Measured on this PC by hand.** Under the table, `memory-q38-1card`'s stderr (`*.out.stderr`) has a line
`strata-q38: the GPU's link to RAM: X GB/s to the GPU, Y GB/s back, Z ms a round trip`, and the old engine's split
runs print `layer split auto: CUDA<n> link X GB/s`. Note both for each card (and with `nvidia-smi -q -d PCIE` whether
the 3060 sits on x4): the planner and the rules for experts on another card are priced from these numbers.

## What the table says

Each run's output is compared request by request (`report.py`), with times and memory-dependent numbers masked,
two batch slots' lines compared slot by slot, and a continued conversation compared only between runs that sent it
the same prompt. **PASS**: every run printed the same. **VARIES**: some runs printed something else, but a run of
the other build printed it too, or both builds vary: chance, not a change. **DIFF**: a build printed something no run
of the other build printed, while at least one build repeated itself; or every run after decoded more than 5% slower
than every run before (at least two runs a build). The note names each run that differs, from which answer token,
and "(new)" where no run of the other build printed it. **FAIL**: a run did not finish, an engine was not built, or a
test failed or was not built. **SKIP**: files or a second card missing.

Under the table, a line per run says what the cards and the PC did while it ran (`drive.py` samples `nvidia-smi`, the
CPU, RAM, page file and disk once a second): SM clock, temperature, load, PCIe link, throttling. A DeepSeek check
that decodes below 3 tokens/s also says how many experts a token its CPU lane read from the model's files.

## Afterwards

Send back `tools\nvidia_session\results\<time>.zip` (the summary is `summary.txt` inside; each run's samples are its
`.gpu.csv` and `.sys.csv` files). Under the table it lists the commits that changed engine code, to bisect a DIFF by
building them one at a time.

To clean up: `git worktree remove --force C:\sn\before`, the same for `C:\sn\after`, then delete `C:\sn`.

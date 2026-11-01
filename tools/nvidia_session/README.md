# The NVIDIA test session

The code-sharing refactor ([docs/REFACTOR_PLAN.md](../../docs/REFACTOR_PLAN.md)) changed engine code that was only
tested on an AMD card. `run.ps1` checks it on an NVIDIA PC in one go: it builds the engines as they were before the
refactor (commit `87d20a5c`) and as they are now, runs both the same way, and writes a table of what matches. It was
written for a PC with an RTX 3060 (12 GB), an RTX 3080 (20 GB) and 48 GB of RAM on Windows 11. Nobody has to watch it
after it starts.

## Before the day

Install, in this order:

1. The newest NVIDIA driver (580 or newer: CUDA 13.0 needs it).
2. [CUDA Toolkit 13.0](https://developer.nvidia.com/cuda-13-0-0-download-archive) (13.0.x, the version the release
   engines and the plan's Docker check are built with). It sets `CUDA_PATH`.
3. Visual Studio 2022 Build Tools with the workload "Desktop development with C++". CUDA 13.0 does not take
   Visual Studio 2026.
4. Git for Windows, and Python 3.12 (64-bit, from python.org).
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
  took 2.2 GB on Linux).

| Phase | What | Default | `-Quick` |
|---|---|---:|---:|
| preflight | cards, driver, peer-to-peer table, tools, RAM, disk | 1 min | 1 min |
| build | both versions, every target, `86-real`, `STRATA_BUILD_DSV41`, `STRATA_NATIVE_EXPERTS`, tests | 1-3 h | 1-3 h |
| unit | `ctest` in both builds, test by test | 20-40 min | 20-40 min |
| smoke | the engines' protocol: Qwen3.8 (commands, batch slots, SAVE / RESTORE across the builds), Qwen3.6 | 45 min | 25 min |
| multi | the B3 list: layer split (auto, 12) in both card orders, peer tier, helper caches, `--vram-elastic`, `--kv-grow`, page-locked RAM copy | 3-4 h | 1 h |
| deepseek | the B2 list: greedy and seeded answers without the expert cache, DSpark on, off and sampled; decode speed | 2 h | 45 min |
| speed | the server on port 8091 with Qwen3.8: answers and prompt / decode tokens/s, 3 runs a build | 30 min | 20 min |
| emulate | builds of PTX for sm_75 and sm_80 run with `STRATA_EMULATE_CC`: the GPU tests and the protocol checks | 3-5 h | 2 h |

The times are estimates (not measured: they depend on the CPU and the drive); the summary records the real ones.
The emulated builds run last because they are the longest and the least likely to find something. Under
`STRATA_EMULATE_CC=80` the host asks for the A100's 163 KB of shared memory, which these cards do not have, so a test
may fail there in both builds; `decode_cluster_parity` passes without comparing anything below sm_90.

## What the table says

Each run's output is compared request by request, with times and memory-dependent numbers masked
(`report.py`). **PASS**: the build after printed the same as the build before. **DIFF**: something else (the note
names the first requests). A request is compared only where two runs of the build before agree, because some answers
change between runs of one build (DeepSeek's, see the plan's open issues); the note counts those. **FAIL**: a run did
not finish, or a test failed in both builds. **SKIP**: files or a second card missing. Speed checks are a DIFF when
decode is more than 5% slower.

## Afterwards

Send back `tools\nvidia_session\results\<time>.zip` (the summary is `summary.txt` inside). Under the table it lists
the commits that changed engine code, to bisect a DIFF by building them one at a time.

To clean up: `git worktree remove --force C:\sn\before`, the same for `C:\sn\after`, then delete `C:\sn`.

## Rows per step (the memory manager's M7)

On the `memory-manager` branch, `tools\measure_rows.py` measures strata-q38's decode speed by conversations at once and
rows per step, with story, code and chat prompts, the drafter's guesses kept, and the experts each step routes to
(docs/DETAILS.md, "Rows per step, lookahead and conversations at once", has the Strix Halo's numbers). It is not part
of `run.ps1`. To repeat it on this PC, build `strata-q38` in a CUDA build folder (`cmake --build <folder> --target
strata-q38`), then run it three times, once for each place the experts live, on the RTX 3080 (`--gpu` takes its
number in nvidia-smi's order). `strata-iq3_s.json` is the run config setup wrote:

```powershell
$q38 = "<folder>\strata-q38.exe"
.venv\Scripts\python tools\measure_rows.py --config strata-iq3_s.json --exe $q38 --case card --out results\rows-card -- --gpu 1
.venv\Scripts\python tools\measure_rows.py --config strata-iq3_s.json --exe $q38 --case ram --out results\rows-ram -- --gpu 1 --gpu-mib 8192
.venv\Scripts\python tools\measure_rows.py --config strata-iq3_s.json --exe $q38 --case ssd --out results\rows-ssd -- --gpu 1 --gpu-mib 8192 --ram-mib 4096
.venv\Scripts\python tools\measure_rows.py --report results\rows-card.json results\rows-ram.json results\rows-ssd.json
```

`card` keeps the engine's own sizes (as many experts on the card as fit; the rest in RAM), `ram` puts most of them
in RAM, and `ssd` reads most of them from the SSD. Each takes one engine start, a warm-up and three rounds: about 15
minutes, and more for `ssd`. Send back the `results\rows-*` files (the JSON, the engine's `.out` and `.out.stderr`).

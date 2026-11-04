# DeepSeek shared pipeline groups, 4 November 2026

These are local development checks, not a two-card speed claim.

## Setup

- Windows 11; Ryzen AI MAX+ 395, Radeon 8060S (one physical GPU), 64 GiB physical RAM.
- HIP gfx1151 Release, engine 0.1.51, branch `memory-manager`, `30e1e7df` plus this task's changes.
- Private copies of `Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw`, pointed at
  `C:/AI-models/DeepSeek-V4.1-Flash`. The shipped 2.00 profile was reset at every engine start.
  Model files and the source pack were not written.
- stdin/stdout only. One engine at a time. Docker Desktop/backend and vmmem were absent.
  `DS4_COMMIT_CAP_GB=30`; explicit GPU and RAM pools; PID-scoped watchdog below 4 GiB free physical RAM.
- Common flags: `--gpus 0 --split-device 0 --layer-split 20 --gpu-mib 4096,4096 --ram-mib 4096
  --max-context 1024 --tokens-min 128 --expert-cache 240 --no-expert-cache --remote-experts off
  --dspark none --batch 4`. Requests use `temperature=0`.

## Identity and live controls

`valid-matrix-{1,2,auto}.out` contains two rounds at each of two, three and four conversations.
The driver uses the NVIDIA kit's four prompts without the repeated DeepSeek prefix, and eight output tokens
per request. `CONTROL_REPLY` records successful live changes to concurrency 2, 3 and 4. All 18 rounds match
standalone; the six `GROUP_TOKENS` rows also match byte for byte across all three group settings.
The stage log records two occupied stages. Three live conversations use one group on two stages.

Mean aggregate decode tokens/s, two rounds; these very short timings contain substantial restart/system noise:

| Conversations | Groups 1 | Groups 2 | Groups auto |
| --- | ---: | ---: | ---: |
| 2 | 2.2044 | 2.1862 | 2.2545 |
| 3 | 1.7830 | 1.9477 | 1.9812 |
| 4 | 1.8317 | 1.8356 | 1.8135 |

`summary.json` retains both samples. Decode timing begins after the last admission; earlier emitted tokens and
prompt reads are excluded. No overlap between physical cards is possible in this setup.

`cut-live-auto-valid.out` uses the kit's repeated prefix (prompt lengths 124, 125, 123 and 120) and eight output
tokens, crossing the first 128-token compressor boundary. It starts at cut 10. While four requests decode, the
driver sends GPU0 pool changes 4096 -> 2048 -> 4096 MiB and concurrency changes 4 -> 2 -> 4, in both rounds.
All commands succeed. Stderr records cut 10 -> 20 with four conversations paused, RAM and disk pauses/restores,
and pool resizing. Both rounds match standalone. Aggregate rates including the control work: 1.8555 and 1.9053.

`dspark-live-auto-full.out` uses GPU pools 4096,7680 MiB, expert cap 2000, `--dspark on --dspark-conf 0`, and
concurrency 1 -> 2 -> 1. The lower confidence threshold forces verification of draft guesses for this test.
All six rounds match standalone. Stderr records accepted guesses before and after the grouped run; the shared
step's unit test separately requires zero proposals while two singleton groups are active. DSpark's normal
confidence threshold remains unchanged in the product and the NVIDIA kit.

Earlier pilot runs used a 1K context with the default 8K token minimum; their settings commands were rejected.
They are excluded. The corrected runs above set `--tokens-min 128` and check command success.
The initial 4/6 GiB DSpark-pool pilots did not fit all drafter experts and are excluded from drafting evidence.

## One-group comparison with the old DeepSeek components

The original `ds4_model.cu` and `paged_model.cpp` were exported from `30e1e7df` and rebuilt separately with
this build's compiler flags. Their objects were linked with the same other libraries. The shared step's change
only affects grouped drafting and is not executed by this one-group comparison. Working sources, branches,
model files and the current executable were not replaced. Binary hashes and the baseline build log are saved.

Both sides use the common flags above, four conversations, short prompts, groups 1, and two rounds.
Every answer matches standalone and the two `GROUP_TOKENS` rows match between old and current components.

| Output tokens per request | Order | Old mean tok/s | Current mean tok/s |
| --- | --- | ---: | ---: |
| 8 | old then current | 1.8302 | 1.7767 |
| 16 | current then old | 1.6525 | 1.9427 |

Samples are in `one-group-summary.json` and `perf*-one-*.out`. This shows no consistent one-group regression,
not a speed improvement claim. Startup order, system activity and the short workloads limit the timing evidence.

## Host validation

- HIP `strata-ds41` build passed; `step_test` passed, including the whole-wave drafting limit and resumption test.
- Python checks: 64 tests and 9 subtests passed for the NVIDIA kit driver/report, paged launcher and architecture.
- PowerShell kit syntax and `git diff --check` passed.
- Not run: CUDA/SYCL builds, real two-card NVIDIA inference/speed, or expert hops (remote experts stay off on
  this same-GPU split). No forbidden GPU tests or lifecycle-matrix tests were run.

The friend's full command is in `tools/nvidia_session/README.md`. The kit keeps 96 output tokens per request and
checks groups 1 versus auto with 1, 2, 3 and 4 conversations, DSpark on/off, and both physical card orders.

## Temporary validation files

Automatic approval review blocked both recursive and file-only cleanup, stating only "blocked by policy".
The `.task_ds_*` scripts/logs and `dspack` / `baseline-build` directories remain. No test engines or background
harnesses remain running. Pilot outputs are excluded from the reported results, as described above.
The local commands were `.venv/Scripts/python.exe .task_ds_matrix.py`, `.task_ds_cut.py`, `.task_ds_dspark.py`
and `.task_ds_perf16.py`; each wrapper invokes the same kit driver with the test settings described here.

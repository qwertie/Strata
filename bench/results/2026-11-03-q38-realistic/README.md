# Qwen3.8 IQ3_S with all experts on the GPU

## Follow-up: the prompt path and decode kernels

The follow-up uses `memory-manager` at `0c3006ca` with uncommitted changes. The initial measurements below
are retained as the before comparison. The same hardware, model, saved environment and memory watchdog apply.

The story's token-2 difference comes from prompt handling. The old server's `--short-read 64` default reads
this prompt through verify windows and splits it at the final chat-turn boundary
(`src/program/generate.cpp:6642`, `6771`, `7494`). The paged adapter always used batched prefill.
Changing owned versus borrowed scratch did not change that decision. Batched prefill feeds FP16/BF16
projection inputs (`src/models/qwen38/prefill/prefill.cpp:3272`); the verifier uses its Q8_1 projection path.
These paths round differently before decode starts. This is a missing short-read policy in the adapter,
not evidence of an incorrect expert layout, position, K/V format or HC weight setting.

The layer trace compares the same story input at positions 33 and 34. Before the fix, layer 0's maximum
absolute difference at position 34 was 0.000157665; the logits differed by up to 0.8460803.
The old logits preferred token 13071, ` storm`, at 18.415834 over token 9662, ` wind`, at 18.233873.
The new logits reversed them: 18.144032 and 18.167442. The adapter now honors short reads using the old
verify width and final turn boundary, and feeds the drafter from those verified residuals with its own
prefill pass. All 41 captured prompt/decode rows then matched exactly: 47 tapped layers of 2,560 floats
and all 248,320 logits per row. Reusing the GDN decode fusions kept this exact match.
[divergence.json](divergence.json) retains every layer's error, the top logits, equality checks and dump hashes.
Raw `.tap` and `.logits` files remain in `%TEMP%/q38-realistic`; logits were exported temporarily through
the existing `Verifier::copy_logits`, and that temporary exporter is absent from the final engine.
The existing layer-tap exporter now also supports batch windows and their actual row positions.

The decode changes reuse existing kernels. A single conversation's rows can use the ordinary QSA attention
and grouped K/V and indexer appends over its shared pools. Per-slot GDN recurrence writes its Q8_1 output
directly; one chain can use the combined
convolution/gate kernel. The shared QSA commit helper restores and appends all indexer layers together,
including per-conversation rotary tables. Existing overrides and per-layer fallbacks remain available.
The saved `STRATA_GDN_SPLIT=1` hardware default prevents the combined GDN commit helper from running;
that helper is not credited with a speed improvement here.

The remaining large decode cost was the shared expert's side stream. In two warm and two measured
256-token seasons controls, `STRATA_SH_STREAM=0` raised no-draft speed to 32.62 tok/s and MTP to
52.04 tok/s, with identical answer tokens. The preceding default-path pair gave 24.83 and 44.34 tok/s.
[shared-inline-controls.json](shared-inline-controls.json) retains both controls. The fully resident
paged path was slower when shared projections overlapped the resident expert work; GPU hardware
counters were not collected, so the specific resource contention is not established.

The new default therefore keeps shared projections on the main stream only for batch graphs with
all experts resident, one gfx1151 GPU, and the GGUF IQ3_S tier (`general.file_type=26`, which includes
mixed expert formats). It reads the model's metadata, rather than matching its filename. Explicit
`STRATA_SH_STREAM=0|1` settings take precedence. Other devices, quantization tiers, partial-residency
graphs, split runs and legacy callers keep their existing policy. The shared fork/join condition is
computed once per captured window. A separate-commit alternative was tested and rejected: it was
slower, with large timing spikes; its answers matched ([rejected control](rejected-commit-control.json)).

Single-GPU MTP can overlap its state commit with the next draft. The shared engine drains pending state
writes before commands, page growth, interleaved prompt chunks, completed requests and destruction.
Reservations already covered by published physical units only advance host bookkeeping, so they do not
force a wait before drafting. The verifier owns a small row-coordinate table, leaving the drafter's
full cell table intact between steps. It copies a full table only when the conversation or coordinates
change, ordered on its own stream. Short prompt
reads preserve residual rows still owed to other conversations' drafters. Snapshot copies use that same
stream, and prompt admission waits for them before reusing verifier scratch. Legacy batch callers remain
synchronous; `STRATA_COMMIT_SYNC=1` still forces synchronous commits.

The guarded [state control](state-control.json) passes with a saved image embedding, three live requests,
a change from 8,192 to 16,384 minimum tokens, cancellation of all three requests, and a return to 8,192.
The standalone 256-token story and picture outputs are identical before and after. The first control
failed when snapshot copies were not ordered with the drafter; that failed run is recorded separately.
The corrected run exited normally without a watchdog kill. It used the existing embedding file; no
image rendering or encoding ran alongside inference.

## Final paired measurement: 4 November 2026

All eight cases were rerun after the final HIP build, on the hardware and with the flags below.
The owner-approved final MTP/suffix new-engine retry trimmed the GPU budget from 60 to 58 GiB
to fit ordinary application headroom. All 24,576 experts remained resident; the other cases used 60 GiB.
Each case has two full warm rounds, two measured 256-token rounds, and, for the new engine,
one warm plus two measured three-conversation rounds. Rates are pooled over both measured rounds.
No compilation, preview, rendering or encoding ran during inference. Desktop applications remained open;
GPU clocks were not fixed or recorded. The owner reported vitest activity in another project during
earlier attempts; the last pair ran after it finished. Background CPU load was not isolated.
These measurements establish this configuration's result.

| MTP | Suffix | Old seasons | New seasons | Old repeat | New repeat | Old story | New story | New three conversations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Off | Off | 26.50 | 31.85 | 25.83 | 31.09 | 27.04 | 31.15 | 60.09 |
| Off | On | 24.01 | 31.48 | 73.21 | 91.17 | 23.56 | 31.01 | 60.66 |
| On | Off | 44.34 | 51.14 | 64.73 | 72.06 | 36.46 | 42.37 | 65.65 |
| On | On | 40.29 | 48.34 | 61.89 | 86.75 | 34.47 | 40.81 | 64.66 |

New was faster in all 12 standalone prompt/mode comparisons, by 11.3% to 40.2%. The previous agent's initial matrix remains below as the before
comparison. [Interactive comparison](comparison.html). [final.json](final.json) retains every warm and measured answer, command, decode time,
draft count, concurrent round, memory floor and timing report. Both executables' hashes are included.

The new no-draft one-row median was verify 31.143 ms, commit 0.006 ms,
and total 31.269 ms. These per-step medians include warm, standalone and concurrent work;
they are not substituted for the paired request rates. MTP's four-row median commit callback was
0.161 ms, compared with about 1.8-2.1 ms in the earlier synchronous diagnostic.
Old `STRATA_DECODE_TIMING=1`, new `--step-profile 1 --stats`, and MTP draft graph timings are retained
in `final.json`. Earlier layer-stage profiles are in [profiling.json](profiling.json), labelled by their
code stage. The event profiler disables some stream and convolution fusions, so its stage totals
are diagnostic numbers rather than final throughput.

Every memory snapshot retained 24,576 experts / 50,292,326,400 expert bytes on the GPU.
The final matrix exited normally in all eight cases, with minimum free commit 3.849 GiB and
minimum free physical RAM 18.185 GiB. New-engine decode file counters stayed flat and CPU misses
were zero; the old engine reported zero decode file blobs. The 60 GiB maximum GPU budget and PID-scoped
3 GiB commit / 4 GiB physical watchdog remained in force. After vitest finished, the owner directed
returning to about 68 GiB preflight headroom, with a small pool trim if needed. The last retry used
66.5 GiB minimum free commit with the 58 GiB pool; earlier successful cases used the original thresholds.

### Answer checks and limits

Prompt order is seasons, repeat, story. `match` means all 256 output tokens matched. Concurrent
comparisons retain seeds; their throughput excludes tokens before the last admission.

- MTP off, suffix off: standalone [match, match, match, match, match, match];
  concurrent [[68, match, match], [68, match, match], [68, match, match]].
- MTP off, suffix on: standalone [match, match, match, match, match, match];
  concurrent [[21, match, 19], [21, match, 19], [21, match, 19]].
- MTP on, suffix off: standalone [match, match, match, match, match, match];
  concurrent [[68, match, 65], [68, match, 65], [68, match, 65]].
- MTP on, suffix on: standalone [match, match, match, match, match, match];
  concurrent [[match, match, 10], [match, match, 10], [match, match, 10]].

The final short-read trace also matches the old file byte for byte across all 41 rows and 47 tapped
layers. Its earlier full-logit comparison matched all 248,320 logits per row. The final text/image
state control passes after live memory changes and cancellation. Those checks do not establish
universal identity across suffix histories, concurrent row layouts or long prompts.

### Validation

Passed: HIP builds of `strata`, `strata-q38` and `strata-ds41`; all five architecture tests;
`step_test`, `batch_rows_test`, `batch_read_test`; the final layer replay, text/image state control
eight-case matrix, and explicit `STRATA_SH_STREAM=1` replay (all 41 tapped rows identical). The second pass was performed by the implementation agent, as requested.

Failed/rejected: the earlier text/image control exposed unordered residual snapshots and passed after
the stream-order fix. Three attempts hit the commit watchdog (diagnostic startup, no-draft concurrency, MTP/suffix startup).
Their complete retries passed; no final rate mixes in a stopped attempt. [followup-safety.json](followup-safety.json) records those stops separately
from the successful matrix. Disabling inline one-row commits was slower and was rejected; its answer
check passed. No speed claim uses that control's timings.

Not run: CUDA, SYCL or Metal builds/inference; DeepSeek inference; multi-card inference or long-context
speed/quality. The owner-excluded `test_lifecycle_matrix`, `vmm_test`, `hip_prefill_gemm` and
`hip_prefill_mmq_parity` were not run. The user's server and port 8080 were not touched.

## Initial full-residency comparison

Measured on 3 November 2026, Windows HIP, Ryzen AI Max+ 395 / Radeon 8060S (gfx1151),
128 GB unified memory. The Windows commit limit was 127.295 GiB throughout the recorded runs.
This is a targeted engine comparison, not the standard benchmark protocol. Both binaries were built
locally on the same day from `memory-manager`, based on `d87c5d39`, with the uncommitted changes described below.

The earlier timings attached to `4c047a0f`, `dce39bcf` and `0b8b86ab` used small expert caches.
They remain useful restricted-memory checks, but do not describe this owner's normal speed.
The current changes do not establish universal greedy identity or a speed win on every case.

## Settings and method

The saved source-setup config is `strata-iq3_s.json`: native IQ3_S GGUF shards under
`C:/AI-models/Strata-data/models/IQ3_S`, pack `C:/AI-models/Strata-data/packs/iq3_s`,
MTP `C:/AI-models/Strata-data/mtp/rt`, `parallel: 2`. Its engine options include:

```text
--pack C:/AI-models/Strata-data/packs/iq3_s
--native C:/AI-models/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
--ple-gguf C:/AI-models/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
--expert-profile <a private copy of data/expert-profile.bin>
--expert-cache auto --resident-experts --prefill auto
--spec 4 --spec-min-p 0.5 --max-context 204800 --kv int8 --vision --vram-reserve-mib 700
```

Each measured old-engine case adds `--resident-budget-gib 4 --suffix-draft 0|3`.
Each final new-engine case replaces the expert count and reserve and adds:

```text
--expert-cache 24576 --vram-reserve-mib 4300
--resident-budget-gib 0.5 --gpu-mib 61440 --ram-mib 512
--rows 8 --mtp-max-t 4 --batch 3 --pause-dir <a private temporary directory>
--step-profile 1 --stats --suffix-draft 0|3
```

MTP-on cases add `--mtp C:/AI-models/Strata-data/mtp/rt`. New-engine lookahead is `0` when both
draft sources are off and `5` otherwise. The old suffix rule expands `--spec 4` to six verify rows,
retaining a four-row model cap. The new default now uses that shared rule too.
New prompt buffers are owned, with auto prefill capped at 2,048 rows; its paged expert cache cannot
lend the contiguous arena the old prompt planner uses. Explicit numeric prefill sizes still apply.

The saved HIP environment is retained: `STRATA_PF_FUSED=1`, `STRATA_PF_GEMM=1`,
`STRATA_HC_UPMIX=1`, `STRATA_PA_FAST=1`, `STRATA_HIP_WMMA=1`, `STRATA_SELECT_WMMA=1`,
`STRATA_PF_SWITCH_MIN_T=4096`, `STRATA_PREFILL_STREAM_MIN=128`, and the saved gfx1151 hipBLASLt
tuning file. Hardware defaults, including HC Q8, remain enabled. `STRATA_SPEC_COST` is unset in the main matrix.
Each start sets `DS4_COMMIT_CAP_GB` to the measured free commit minus 3 GiB, in decimal GB;
this variable controls DeepSeek, not Qwen. Qwen's new pool reserve is explicitly
`STRATA_COMMIT_RESERVE_MIB=3072`; the normal automatic reserve is unchanged.
[measurements.json](measurements.json) retains every matrix command and environment value.

All engines use private stdin/stdout protocol commands, without an HTTP server or listener.
The user's port 8080 and server were not touched. One engine ran at a time. Every start checked both
free commit and physical RAM and refused Docker Desktop, its backend or WSL's VM processes.
A transient Docker CLI was ignored. A PID-scoped watchdog sampled every 250 ms and killed only the
test engine below 3 GiB free commit or 4 GiB free physical RAM, as the owner explicitly requested.
No browser preview, rendering, encoding, Docker VM or container ran alongside inference.

Every main answer is greedy and 256 tokens long, using the saved no-thinking chat template. Prompts are:

1. “Explain how the seasons on Earth work, including the axial tilt, solstices, equinoxes, and differences
   between hemispheres. Write a detailed explanation of at least 600 words.”
2. “Repeat exactly this list 100 times, without introduction or commentary: alpha beta gamma delta epsilon.”
   followed by two more copies of the five-word list on separate lines.
3. “Write a detailed story of at least 600 words about a lighthouse keeper finding a mysterious letter.”

Before timing, GPU expert bytes were unchanged at three idle samples, then unchanged through two complete
three-prompt warm-up rounds. The new engine already held all experts before `READY`; the additional checks
confirm that requests did not displace them. Two measured rounds follow. Standalone rates pool the two
engine-reported decode times, excluding prefill, using the engine's 256-output-token counting convention.
For three conversations, the first complete round is warm-up. Two measured rounds count only tokens emitted
after the last admission: 762 each without MTP, 760 each with it, excluding seeds and earlier tokens.

## Throughput

All eight main cases held **24,576 / 24,576 GPU experts, 50,292,326,400 bytes (46.84 GiB)**.
No decode expert file loads or CPU misses occurred in the measured new-engine rounds. Its file-read counters
stayed flat; the old engine also reported zero decode file blob reads. Old RAM expert copies used 2.69 GiB,
including the prompt-loan complement. The new RAM pool held about 0.247 GiB initially and about 0.136 GiB
after warm-up; it also parked conversations on disk. This small RAM pool did not starve decode: all experts
fit on the GPU.

| MTP | Suffix | Old seasons | New seasons | Old repeat | New repeat | New three conversations, aggregate |
|---|---|---:|---:|---:|---:|---:|
| Off | Off | 25.20 | 23.98 | 25.05 | 24.28 | 48.38 |
| Off | On | 25.29 | 24.56 | 75.91 | 72.49 | 49.47 |
| On | Off | 43.21 | 37.08 | 60.97 | 58.00 | 54.48 |
| On | On | 39.47 | 37.10 | 60.29 | 67.37 | 56.05 |

Rates are tokens/s. The story control was old/new 24.99/24.31, 25.44/24.73,
34.93/31.91 and 33.46/31.54 in the same row order. Exact round times are in the JSON.
The final matrix retains slower results rather than selecting the best replay: an earlier full-cache,
same-day no-draft replay gave new 25.43/25.62 seasons/repeat and 49.30 aggregate.
Ordinary applications remained open; GPU clocks and package power were not recorded.

The normal-prompt MTP gap is largely an acceptance difference: the old engine accepted 158/239 drafts in
each measured round, versus new 144/243. That is 97 versus 111 decode windows for the 255 tokens after the seed.
Mean round time per window is about 61.1 versus 62.2 ms. The different answers prevent treating this as an
equal-token engine-overhead comparison. The repetitive prompt is identical across engines in the main matrix.

## Picture and legacy-batch controls

Using the same full-residency settings, no drafting and an existing 16-row image embedding file,
a picture/counting request gave old **25.06** and new **24.76 tok/s** (two measured 256-token rounds
following a picture warm-up). Image encoding is excluded. The new picture/seasons/repetition mix gave
**49.91 aggregate tok/s**, two measured rounds after a batch warm-up, 762 tokens counted after final
admission in each. Picture tokens match old/new and concurrent/standalone in both rounds; seasons
changes at token 56, repetition matches. Old/new minimum free commit was 8.068/5.785 GiB and minimum
free physical RAM 23.723/26.262 GiB. Observed system commit increases were 60.844/62.997 GiB.

The old engine's optional three-slot startup at 204,800 context crossed the commit floor (2.906 GiB),
so its concurrency identity control was repeated at **8,192 context**, `--batch 3`, otherwise the old
full-residency no-draft settings. It completed. Seasons and repetition match standalone; story first
differs at token **186** in both 256-token batch rounds. This proves batch dependence in the old engine,
without proving that it explains the new engine's token-2 divergence. The exact earlier sparse-slot,
live-group picture failures from DETAILS.md were not rerun in the old engine. Same-GPU split batching
is refused by the old engine; there is only one physical GPU here.

## Memory and startup

The requested new GPU budget is 60 GiB, not the old 14 GiB test budget. Actual pools were capped by
the integrated GPU's dedicated-share and commit checks, weights, scratch and the explicit outside-pool reserve.
The pool includes conversation state; its size is not the GPU expert size alone.

| Engine / MTP / suffix | Actual GPU pool GiB | Free commit before GiB | Minimum free commit GiB | Minimum free physical GiB | Observed system commit increase GiB |
|---|---:|---:|---:|---:|---:|
| Old / off / off | Expert cache 46.84 | 69.024 | 7.991 | 18.356 | 61.033 |
| Old / off / on | Expert cache 46.84 | 70.286 | 13.629 | 22.963 | 56.657 |
| Old / on / off | Expert cache 46.84 | 74.618 | 12.147 | 23.361 | 62.471 |
| Old / on / on | Expert cache 46.84 | 75.119 | 14.265 | 23.944 | 60.854 |
| New / off / off | 53.75 | 68.168 | 4.334 | 24.666 | 63.834 |
| New / off / on | 53.75 | 67.706 | 3.538 | 24.768 | 64.168 |
| New / on / off | 52.50 | 66.966 | 3.762 | 24.241 | 63.204 |
| New / on / on | 52.75 | 68.799 | 3.945 | 24.265 | 64.854 |

The last column is the measured change in **system** charge from preflight to the run's maximum.
It includes changes in other applications, so it is not an exact attribution to the engine.
GPU allocations are not consistently represented by process private bytes on this driver: those peak at
53.08–53.20 GiB for the old engine and 2.93–4.52 GiB for the new one. Do not add GPU usage to private bytes
or to system charge: this is unified memory. The page file did not grow during these recorded runs.
A read-only Windows query reported `AutomaticManagedPagefile=true`, `C:/pagefile.sys` allocated
65,174 MiB, and 68,340,748,288 bytes of Windows-visible physical memory. A read-only DXGI query reported
68,535,664,640 bytes (63.83 GiB) of dedicated GPU share. Increasing the page file can increase commit
headroom, but does not increase that share. The allocator caps against both
(`src/common/serve/paged_engine.cpp:428`).
The main matrix passed the owner's 3/4 GiB floors, but leaves little commit headroom for more applications.

Old full-residency `READY` took 39.41–40.28 seconds with the bounded, unbuffered load path.
New full-residency `READY` took 39.29–45.65 seconds; 16.55–19.42 seconds of that is the final shared pool fill.
Earlier page-pool pilots reached `READY` with about 20 GiB of experts and filled toward 41 GiB only later.
The new shared startup fill removes that partially resident first-request state; it does not bypass pool limits.

The saved old mode is `--resident-experts`, which selects mmap plus a pinned RAM complement
(`include/strata/models/qwen38/options.hpp:411`). `Model::load` opens the read-only expert mappings, then
copies ranked experts into the GPU cache (`src/models/qwen38/model.cpp:3273`). On HIP,
`ExpertCache::fill_slot_blocking` stages **one expert** in pinned host memory and copies it to the GPU
(`src/core/expert_cache.cpp:545`). After the GPU fill, `src/program/generate.cpp:2328` calls
`Model::pin_resident_experts`. `FileExpertSource::pin_cache_complement` first checks that fill is complete,
then trims mapped working-set pages to the standby list and allocates the pinned complement
(`src/core/expert_source.cpp:3041`, `3089`). The mappings are `PAGE_READONLY` / `FILE_MAP_READ`
(`src/core/expert_source.cpp:1285`): these file-backed pages are not a full private committed expert arena.
The alternative arena mode does allocate CPU expert storage, but the saved resident/mmap flags do not select it.

A separate startup-only run without `--resident-budget-gib` tested the owner's saved mapped load path.
Its working set grew to 22.14 GiB while private bytes stayed at 48.61 GiB, consistent with mapped file pages
growing over an already allocated GPU cache. At 91.93 seconds the watchdog stopped it at 3.960 GiB free
physical RAM; free commit was still 11.851 GiB. It did not reach the trim, so the owner's complete
25–30 GB RAM drop was not reproduced under the required physical-RAM floor.
The bounded old runs did reach the trim, but unbuffered reads left little mapped working set to release.

New `strata-q38` reads experts from the file/RAM source into GPU pool pages; it still uses host staging,
not direct SSD-to-GPU I/O. Its initial `Model::load` fills part of the pool. The shared
`PagedEngine::serve` now calls the existing `fill_cards` before `READY`
(`src/common/serve/paged_engine.cpp:534`, `1839`). Both model families use that shared path.
The Qwen adapter selects uninitialized chunk allocation and initializes experts and conversation pages when
they are used, avoiding writes over unused GPU runs (`src/models/qwen38/paged_model.cpp:1104`,
`src/models/qwen38/conversation_pages.cpp:108`). Once both engines are fully resident, file-cache versus
unbuffered startup cannot explain the remaining measured steady-state gap: neither is loading experts then.

## Changes and validation

The shared engine fills GPU expert pools before `READY`, reports optional per-step timings, and allows an
explicit commit reserve for controlled runs. Shared page-pool routing reuses ordered-index nodes when an expert
is touched, preserving eviction order without allocating replacement nodes. The shared window-width helper
keeps the old suffix capacity rule in both Qwen entry points.

Qwen can switch to the resident graph after the pool becomes full, owns its prompt scratch, avoids zeroing
unused GPU chunks, and uses the existing one-row state-commit path by default in the paged adapter.
An explicit `STRATA_BATCH_SELF_COMMIT=0|1` still overrides it; the old engine's default is unchanged.
Picture coordinates are cached per conversation and copied on the GPU for the drafter, rather than rebuilt
and uploaded from the CPU for every proposal.

Changed implementation files:

- `src/common/serve/paged_engine.cpp`, `paged_start.cpp` and their header: startup fill, pool allocation policy,
  the explicit reserve and optional phase timing.
- `src/common/state/page_pool.cpp` and its test: reuse eviction-index nodes when routing touches an expert.
- `include/strata/common/spec/window_planner.hpp`, `src/common/spec/step_test.cpp` and
  `src/models/qwen38/model.cpp`: share and check verify-width sizing.
- `src/models/qwen38/verify.cpp` and its header: permit the resident graph after refill, enable the paged
  adapter's existing one-row commit path and include resident synchronization in wait telemetry.
- `src/models/qwen38/paged_model.cpp`: owned prompt scratch, page initialization on use and cached coordinates.
- `src/base/gpu/runtime.cu` and its header: uninitialized allocation and an explicit stream wait for cached copies.

The no-draft replay's median commit callback fell from about 1.8 ms to 0.005 ms. This is a phase-cost
improvement, not a controlled claim that the combined patch increased throughput by a particular percentage.
The full-cache no-draft before/after check kept all six 256-token standalone answers identical.
In the final matrix, median step times are about 40.9 ms without drafting and 65–66 ms with MTP
(the latter mixes standalone and concurrent row layouts). Verification dominates. Typical MTP draft time is
7.3–7.5 ms, commit 2.1 ms and page/pool work 0.013–0.016 ms. The resident verifier's synchronization time
is included in verification and separately accumulated in its wait counter; CPU misses and file loads remain zero.
These are host elapsed timings, not a complete GPU kernel timeline.

**Passed:** all eight full-residency matrix cases, two measured 256-token rounds; all new three-conversation
cases, two measured rounds after warm-up; HIP builds of `strata`, `strata-q38`, `strata-ds41`;
`step_test`; `page_pool_test` (1,121,445 checks); explicit architecture and paged-launch tests
(16 tests and two subtests). The main matrix had no watchdog trip.

**Failed:** universal greedy identity. Old/new seasons first differ at output token 56 without MTP,
68 with MTP only, and 117 with MTP plus suffix; story differs at token 2. Repetition matches throughout.
New concurrent/standalone seasons differ at token 56 without MTP; story differs at token 2 in all four cases.
The old engine's own MTP-on/off answers also differ (seasons token 68, story token 65).
These observations do not by themselves prove the cause is numerical rounding.
An old-engine owned-prefill control retained the same old/new mismatches, ruling out prompt borrowing as
their sole cause. Matching or beating the old engine on every case and preserving every greedy answer
remain unmet requirements; no universal performance or identity claim is made.

The existing cost-aware drafting policy was also tested (`STRATA_SPEC_COST=1`), without changing its default.
MTP-only seasons/repeat/story were 37.77/58.24/33.21 tok/s; MTP plus suffix gave 39.36/68.83/34.86
in its two completed standalone rounds. It changed some answers relative to the main matrix (seasons at
token 117, story at 109 or 116), and the suffix run's final concurrent round hit the watchdog, so it is
not promoted as an answer-preserving fix or a completed concurrent comparison.

The retained logs contain six 3 GiB commit-watchdog stops, separate from the successful main matrix:

| Exploratory run | Minimum free commit GiB | Minimum free physical GiB |
|---|---:|---:|
| First 60 GiB startup, smaller outside-pool reserve | 2.999 | 17.399 |
| Earlier MTP/suffix concurrent replay | 2.956 | 20.368 |
| Old three-slot, 204,800 context startup | 2.906 | 22.857 |
| Cost-aware MTP/suffix final batch | 2.868 | 21.926 |
| Old HC-Q8-off diagnostic | 2.982 | 18.275 |
| Old HC-Q8-off startup retry, owned prompt scratch | 0.004 | 12.818 |

The watchdog samples every 250 ms; a large allocation or another application's growth can jump below its floor
between samples. Only the test PID was killed. The last retry started with 56.590 GiB free commit;
an unrelated Node process grew to about 12 GiB private memory. That preflight estimate was insufficient for
the engine's outside-pool allocations and background growth. The harness now requires the larger of the pool
sum and the observed total allocation, plus the 3 GiB reserve: 68 GiB for new, 66 GiB for old.
The saved old unbounded startup was stopped at the physical floor (minimum 3.960 GiB), as described above.
Earlier 16 GiB-floor pilots are separate from the owner-approved 3 GiB-floor matrix.
[memory-runs.csv](memory-runs.csv) records pre-start commit, minimums and observed charge changes for all
34 retained run logs, including archived attempts; [controls.json](controls.json) retains picture, batch and
cost-policy answers and commands.

**Blocked at the end of the initial pass:** further full-residency diagnostics and optimization.
That pass's final preflight refused to start with
64.479 GiB free commit and 23.879 GiB free physical RAM, below the revised 68 GiB requirement
(`%TEMP%/q38-realistic/run.py:105`). No engine is left running. The HC-Q8-off comparison did not complete,
and the planned isolated old/new MTP graph-timing control did not start. More commit headroom is needed
before those GPU checks can continue. The measured MTP speed gap and identity failures remain unresolved.

**Not run:** CUDA, SYCL or Metal builds/inference; DeepSeek inference after these shared changes;
real multi-card batch groups; long-context speed or quality; the excluded lifecycle/GPU tests.
The one-GPU three-conversation numbers do not replace a two-card group-overlap measurement.

Full commands, protocol streams, stderr, 250 ms watchdog samples and the guarded Python harness remain in
`%TEMP%/q38-realistic`. The compact JSON here retains the final matrix and exact answer tokens.
No commit or push was made.

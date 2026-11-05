# DeepSeek realistic pools, 4 November 2026

Windows 11 Pro 10.0.26300; Ryzen AI MAX+ 395, Radeon 8060S, HIP gfx1151, engine 0.1.51, branch `memory-manager`,
`73edf255`. The requested new pools are **60 GiB GPU + 20 GiB RAM**. The allocator capped the GPU pool to **51.49 GiB
without DSpark and 49.99 GiB with it**. Both available pools were filled before READY. The requested 60 GiB GPU pool
was not allocated. The pack contains 15,360 main-model experts; filling the GPU cache does not put them all on the GPU.

## Settings and method

The owner's `strata-deepseek-v4.1-flash-ist-das-2.00bpw.json` supplies the model and context. Read-only GGUF files are
under `C:/AI-models/DeepSeek-V4.1-Flash`. Each start uses a private metadata pack and resets its expert profile to the
same seed. No original model or pack was written. The legacy target is `strata-dsv41` from this checkout, not the saved
installed 0.1.52 executable. Both diagnostic executables reuse the existing HIP kernel libraries; host-only copies
print counters. No production engine source or executable was changed. Diagnostic logging overhead was not isolated.

Common engine flags:

```text
--serve --pack <private metadata pack> --max-context 524288 --eos 1
--dspark none|on --dspark-conf 0.65
```

`none` is the off case: the drafter is not loaded. The saved config's `auto` is replaced by the explicit matrix mode.
The default five routed experts per token (faster mode), KV format, prefill policy and CPU thread selection remain the
owner's defaults. New-engine additions:

```text
--gpu-mib 61440 --ram-mib 20480 --expert-cache auto
--batch 4 --concurrent 4 --remote-experts off --step-profile 1
--pause-dir <private temporary directory>
DS4_COMMIT_CAP_GB=115
STRATA_COMMIT_RESERVE_MIB=0
```

Legacy additions:

```text
--vram-experts 1 --ssd-cache 21.47483648
DS4_COMMIT_CAP_GB=62.95 (off), 61.5 (on)
DS4_UMA_GPU_RESERVE_GB=4
```

The legacy caps keep GPU expert allocations within 1% of the new engine's measured allocation. Its outside-cache UMA
reserve is explicit because the automatic reserve scales with the configured maximum context. `--ssd-cache` uses
decimal GB; 21.47483648 GB is 20 GiB. [measurements.json](measurements.json) keeps exact commands, environments,
profile seed hash, answer token IDs, timings, residency reports and per-round counters.

All requests use the pack's no-thinking chat template, `temperature=0`, and 256 output tokens. Prompt IDs are 0:
detailed explanation of seasons; 1: repeat `alpha beta gamma delta epsilon`; 2: a lighthouse keeper finding a
mysterious letter. Exact texts and rendered prompts are in the JSON. Before timing, GPU expert bytes were unchanged at
three idle samples and through two complete three-prompt warm-up rounds. Two measured rounds follow. No fixed cooling
gaps were added. Standalone rates pool native engine decode times and its 256-token counting convention, excluding
prefill. Concurrent rates count only tokens after the final admission and end at the last BDONE. Two-chat samples cover
all three prompt pairs twice; three chats use all prompts; four repeat seasons. Each pair/width has a preceding batch
warm-up. Per-slot rates are retained in the JSON.

The no-drafter new matrix was interrupted during an unfinished pair and continued in a second engine process. No
completed timed round was repeated. Unfinished cells were completed after fresh startup and warm-up. The two segment
commands and exit statuses are retained. Both segments held the same GPU expert bytes. This restart is a limit on
interpreting pooled timing differences.

**Thermal caveat:** the owner reports 96–99 °C GPU temperature under load with a quiet fan. Clocks, package power and
temperature were not logged per timed round. Thermal throttling may contribute to round-to-round variation; this study
does not isolate its effect. Runs remain sequential, not A/B/A/B. No fan or power settings were changed. Earlier faster
samples are retained as controls, not substituted for the final matrix. Those controls also use different cache or
drafter-allocation flags; they do not isolate thermal effects.

## Throughput and residency

Rates are tokens/s. One-chat totals pool the three prompts equally by output-token count. “Mean each” is aggregate
divided by chat count. GPU counts are trunk experts, plus the separate 384 drafter experts when enabled. RAM counts are
pool members for new and allocated store slots for old; their payload sizes and allocation strategies differ.

| Engine | DSpark | Chats | Total tok/s | Mean each | Kept/offered | GPU experts | RAM experts/slots |
|---|---|---:|---:|---:|---:|---|---|
| Old | Off | 1 | 5.24 | 5.24 | 0/0 | 6266 | 2090 |
| Old | On | 1 | 7.25 | 7.25 | 710/761 | 5259 + 384 draft | 2090 |
| New | Off | 1 | 6.82 | 6.82 | 0/0 | 6308 | 2431 |
| New | Off | 2 | 9.19 | 4.60 | 0/0 | 6308 | 2431 |
| New | Off | 3 | 9.96 | 3.32 | 0/0 | 6308 | 2431 |
| New | Off | 4 | 9.44 | 2.36 | 0/0 | 6308 | 2431 |
| New | On | 1 | 11.16 | 11.16 | 691/746 | 5277 + 384 draft | 2417–2425 |
| New | On | 2 | 7.52 | 3.76 | 0/0 | 5277 + 384 draft | 2395–2414 |
| New | On | 3 | 7.65 | 2.55 | 0/0 | 5277 + 384 draft | 2381–2386 |
| New | On | 4 | 8.96 | 2.24 | 0/0 | 5277 + 384 draft | 2371–2372 |

Standalone prompt rates:

| Engine | DSpark | Seasons | Repetition | Story |
|---|---|---:|---:|---:|
| Old | Off | 4.51 | 5.91 | 5.52 |
| Old | On | 6.49 | 13.32 | 5.41 |
| New | Off | 6.68 | 7.25 | 6.57 |
| New | On | 9.85 | 21.22 | 8.32 |

New DSpark-on drafting steps / measured step calls: 1 chat(s): 568/845; 2 chat(s): 0/1530; 3 chat(s): 0/510; 4 chat(s):
0/510. The drafter's real work was separated from no-op calls by host duration: actual calls were at least 12.35 ms in
the retained on case; no-op calls were at most 0.002 ms. A 1 ms threshold counts attempts, including guesses rejected
by confidence. No actual drafting occurred in the measured concurrent rounds. The on case keeps about 6.72 GiB of
drafter experts on the GPU even while drafting is paused.

## Expert misses and file reads

The following are totals over the samples, not normalized rates. One-chat rows contain six answers, two-chat rows six
paired rounds, and three-/four-chat rows two rounds each. CPU dispatches use the shared model's counter for expert rows
sent outside the GPU cache. New file reads count expert blobs and payload bytes requested from its unbuffered file
source, including adaptive cache copies. Old file reads use RAM-store misses and its bytes-read counter, including
lookahead reads. Concurrent counter boundaries exclude the final admission's seed, as the speed timer does. These are
expert payload counters, not device-level SSD traffic: sector padding, Engram reads, OS/driver paging and buffered
fallback behavior are not measured by them.

| Engine | DSpark | Chats | CPU expert dispatches | File expert reads | Payload GiB |
|---|---|---:|---:|---:|---:|
| Old | Off | 1 | 12,443 | 5,005 | 40.19 |
| Old | On | 1 | 20,571 | 7,291 | 58.55 |
| New | Off | 1 | 15,114 | 4,968 | 40.15 |
| New | Off | 2 | 22,824 | 5,266 | 42.59 |
| New | Off | 3 | 12,414 | 4,239 | 34.26 |
| New | Off | 4 | 16,484 | 5,515 | 44.51 |
| New | On | 1 | 29,512 | 7,826 | 63.08 |
| New | On | 2 | 32,837 | 9,817 | 79.20 |
| New | On | 3 | 19,585 | 6,974 | 56.32 |
| New | On | 4 | 25,965 | 8,603 | 69.16 |

## Host phase profile

Mean host elapsed milliseconds per measured step, not per output token. Verification includes model execution and
expert I/O waits. Speculative and independent rows have different work per step.

| DSpark | Chats | Draft ms | Verify ms | Commit ms | Pages ms |
|---|---:|---:|---:|---:|---:|
| Off | 1 | 0.000 | 144.859 | 0.001 | 1.607 |
| Off | 2 | 0.000 | 214.408 | 0.001 | 2.663 |
| Off | 3 | 0.000 | 295.966 | 0.001 | 3.882 |
| Off | 4 | 0.000 | 416.491 | 0.001 | 4.735 |
| On | 1 | 19.729 | 140.070 | 0.022 | 2.808 |
| On | 2 | 0.001 | 262.767 | 0.001 | 2.681 |
| On | 3 | 0.001 | 386.703 | 0.001 | 3.906 |
| On | 4 | 0.001 | 438.934 | 0.001 | 4.646 |

## Identity

Positions below are one-based output-token positions; “match” means all 256 IDs match.

| DSpark | Prompt | Old/new first difference, rounds 1 / 2 |
|---|---|---|
| Off | Seasons | 1 / 9 |
| Off | Repetition | match / match |
| Off | Story | 33 / 39 |
| On | Seasons | 1 / 9 |
| On | Repetition | match / match |
| On | Story | 2 / 2 |

New concurrent/standalone matches, compared with the first measured standalone reference: Off: seasons 0/10, repetition
8/8, story 0/8; On: seasons 0/10, repetition 8/8, story 0/8. Exact first differences for every slot are in the JSON.
Repeated standalone rounds also change some seasons/story answers within each engine. The caches remain adaptive:
stable byte/count residency does not freeze which expert IDs occupy each tier. Temperature zero therefore did not
establish universal identity in these adaptive-cache runs. These observations do not establish a numerical cause.
Repetition provides an identical-answer control; other answers must not be used as equal-token engine-overhead
comparisons.

## Memory, validation and limits

Windows exposes 68,340,748,288 bytes of physical RAM. A read-only DXGI query reports a 63.83 GiB dedicated GPU share.
Windows' system-managed page file was left alone. Every start checked free physical RAM and refused Docker Desktop, its
backend, vmmem/WSL or an existing Strata engine. A PID-scoped watchdog sampled every 250 ms and stopped only the test
engine below 4 GiB free physical RAM. There was no harness free-commit floor. Engine allocation caps are recorded
separately. stdin/stdout only: no HTTP listener, no port 8080 operation, and no user server restart. No browser
preview, rendering, encoding or unrelated GPU workload was started.

| Engine | DSpark | GPU expert GiB | Min free physical GiB | Min free commit GiB | READY s |
|---|---|---:|---:|---:|---:|
| Old | Off | 50.62 | 6.340 | 0.014 | 30.90 |
| Old | On | 49.27 | 5.485 | 0.005 | 31.14 |
| New | Off | 50.51 | 10.200 | 0.099 | 53.80 |
| New | On | 48.98 | 10.078 | 0.009 | 54.24 |

The table includes total GPU expert bytes (trunk plus drafter). Minimum counters cover each process's whole lifetime;
the restarted new-off case combines both segments. READY shows its first segment. System counters include other apps
and are not an exact attribution of memory to the engine. GPU pool placement is not a measurement of OS/driver physical
residency. Actual RAM allocation and expert counts are retained for every round.

**Passed:** all requested matrix cells have at least two completed measured rounds, with 256 tokens per answer;
diagnostic host compilation/relink of both HIP engines; `build-hip-win/step_test.exe`; `python -m pytest
tests/architecture -q` (6 passed); [result consistency checks](validation.json) and `git diff --check`.

**Failed:** universal greedy identity, as above. An earlier legacy pilot with a larger GPU cache and a 20 GiB RAM
budget crossed the watchdog floor at 3.994 GiB free physical RAM during warm-up; none of its timings are in the main
table. A 16 GiB RAM-budget control completed. The final matched legacy cases use the requested 20 GiB budget.
[controls.json](controls.json) retains these controls and the earlier complete drafting-off (`--dspark off`) control,
distinct from main `none` mode. The deliberate interrupted new-off process is recorded separately from that watchdog
stop.

**Optimization:** the new engine was faster in all six matched single-conversation prompt/mode comparisons, so the
requested optimization condition was not triggered. No production changes or optimization gain from this task are
claimed. The phase logs profile verification, drafting, commit and page/pool work; timings are host elapsed times, not
a GPU kernel timeline. Comparisons use the measured allocations and retain all slower samples. Thermal variation and
differing answers limit causal performance conclusions.

**Not run:** CUDA, SYCL or Metal builds/inference; multi-card inference; long-context speed/quality; GPU-driver paging
telemetry; the excluded lifecycle matrix, vmm_test, hip_prefill_gemm and hip_prefill_mmq_parity tests. The configured
524,288 maximum context is not a long-prompt benchmark.

Full protocol streams, stderr, watchdog samples, diagnostic source copies and harnesses remain in
`%TEMP%/ds41-realistic`. Result folders contain no binaries, model files or pack copies. No commit or push was made.
All test engines and harnesses were stopped before handback.

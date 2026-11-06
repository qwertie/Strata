# Labelled memory reports

The shared protocol is defined in
[`memory_report.hpp`](../include/strata/common/serve/memory_report.hpp).
`GET_MEMORY_USE` and `CHANGE_SETTINGS` return the same report.

Each `MEMORY` line identifies a GPU, RAM or SSD and its sizes and settings.
For an adjustable pool, `pool_offset` gives the bytes before it on the bar,
so category names do not determine the divider position.
It is followed by any number of lines of this form:

```text
MEMORY_BLOCK gpu0 1048576 other Device%20allocations Drafter
```

The fields are memory ID, bytes, category, pool name and block label.
The last three fields are percent-encoded UTF-8; `~` represents an empty string.
Categories are engine-provided strings. There is no category allowlist in the
shared engine, Python parser or web panel. Older engines may still send category
totals as numeric fields on `MEMORY`, without block details.

Blocks are disjoint totals, grouped by matching labels. Backing allocations for a
page pool are excluded when its blocks are reported. Page padding belongs to the
block, and unused pool space is reported separately. Experts retain their layer
and size class; conversation blocks retain their conversation and state labels.
When a pool starts inside its backing allocation for alignment, prefix and suffix
bytes are reported separately as `Alignment padding`.

The GPU and pinned-host allocators retain allocation lifetimes through
`base/gpu/managed_memory.hpp` and the shared `MemoryManager`. Page pools use the
same label type. Engines name significant allocations at their allocation sites.
The shared paged engine combines these records with each page pool's block report.
Adding a category or sublabel requires no Python or UI changes.

Python exposes `uses` (category totals) and `blocks` (category, pool, label, bytes).
The web panel and desktop app share the same renderer. Segment and legend tooltips
show all matching block labels and sizes. `other` is Strata's own miscellaneous
allocations. `other_apps` is memory outside the engine's accounted use, and is
drawn after free memory at the right of the bar. Where Windows provides the
engine's dedicated usage, unclassified runtime bytes remain under Strata's Other.

Pool growth is limited by actual allocation capacity and configured pool limits.
There is no automatic Windows commit reserve. The operating system and allocator
can still refuse an allocation, which is reported through the existing refusal
or shortfall response.

Blocks may append an identity group and comma-separated unsigned member IDs:
`MEMORY_BLOCK <memory> <bytes> <category> <pool> <label> <identity-group> <ids>`.
The group is percent-encoded like the labels. Members within one block have
identical allocated sizes (`bytes / member count`). A group and member ID
identify the same logical content across memories, independently of display
labels. Page pools use expert category and layer as the group, and the expert
index as the member ID. Readers that do not need residency can ignore these
optional fields. The Python view preserves them as `identity_group` and `ids`.

The diagram shades RAM experts also held by any GPU in darker teal. Those
copies still count as allocated RAM. SSD experts mean experts held in neither
VRAM nor RAM, rather than the size of the complete model files on disk.

`CONVERSATION` lines can append `last_tokens_at=<Unix seconds>` before the
optional token preview. This is the last time prompt or generated tokens
reached that logical conversation. Pausing, copying and restoring preserve
it. Missing times are unknown, including files written before this field
was recorded. `PAUSE_DIRECTORY <path>` gives the engine's folder for its
automatic SSD pause files, with spaces retained to the end of the line.
The memory view passes it through as `pause_directory`.

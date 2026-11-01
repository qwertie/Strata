// include/strata/common/state/pool_copy.hpp - a conversation's pages copied between two page pools (a GPU's and the
// RAM's) and to and from a file on the SSD: how a conversation is paused and resumed (docs/MEMORY_MANAGER_PLAN.md,
// step M5, "Tiers and pausing").
#pragma once

#include "strata/common/state/page_pool.hpp"

#include <string>

namespace strata::core {

/// Copies conversation `from_id`'s blocks in pool `from` into conversation `to_id`'s in pool `to`: per region, as
/// many of its first blocks as `to` holds, which `from` must hold too.  The pools have the same layout and page size;
/// either may be in a GPU's memory.  The copies are done on return.
bool copy_conversation(const PagePool& from, ConversationId from_id, const PagePool& to, ConversationId to_id,
                       std::string& err);
/// Writes conversation `id`'s blocks in `pool` to a new file at `path`, flushed to the drive before it returns.  When
/// that fails, no file is left.
bool save_conversation(const PagePool& pool, ConversationId id, const std::string& path, std::string& err);
/// Gets the start of the names of this process's files of paused conversations in folder `dir`: "paused-", its
/// process id, the time now and "-", so that engines sharing the folder keep apart and remove_stale_pause_files can
/// tell whose a file is.
std::string get_pause_file_prefix(const std::string& dir);
/// Deletes the files of paused conversations in folder `dir` whose process no longer runs (it crashed or was killed
/// before it could delete them); gets how many.  A file whose process id a new process has taken stays.
int remove_stale_pause_files(const std::string& dir);
/// Reads a file save_conversation wrote into conversation `id`'s blocks in `pool`: per region, as many of the file's
/// first blocks as the conversation holds, which the file must hold too.
bool load_conversation(const std::string& path, const PagePool& pool, ConversationId id, std::string& err);

}  // namespace strata::core

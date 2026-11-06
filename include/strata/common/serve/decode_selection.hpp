#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strata::serve {

/// Keeps at most `limit` requests for PagedEngine's next decode step. Cancellations drain first; otherwise higher
/// priorities win, with older requests winning ties. Conversation supplies `dec.stop`, `dec.priority` and `used`
/// (its request sequence number); advancing a request does not change that sequence number.
template<class Conversation>
void select_decode_conversations(std::vector<Conversation*>& conversations, size_t limit) {
    std::stable_sort(conversations.begin(), conversations.end(), [](const Conversation* a, const Conversation* b) {
        if (a->dec.stop != b->dec.stop)
            return a->dec.stop;
        return a->dec.priority != b->dec.priority ? a->dec.priority > b->dec.priority : a->used < b->used;
    });
    if (conversations.size() > limit)
        conversations.resize(limit);
}

}  // namespace strata::serve

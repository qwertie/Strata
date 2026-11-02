// include/strata/base/gpu/links.hpp - how fast a card's links are: to the host and to another card, each way, and the
// round trip of a small copy, measured at start (docs/MEMORY_MANAGER_PLAN.md, "Links are measured, not assumed").
#pragma once

#include <cstddef>

namespace strata::core {

/// One link's speed as measured.  A card pair without peer access is measured as the runtime copies between them
/// (through the host), which is what a hop between them costs.
struct LinkSpeed {
    double to_gbps = 0;       ///< toward the card (from the host, or from the other card), GB/s
    double from_gbps = 0;     ///< from the card back, GB/s
    double latency_ms = 0;    ///< a small copy there and one back, waited for

    /// Tells whether both directions were measured.
    bool is_known() const { return to_gbps > 0 && from_gbps > 0; }
};

/// Measures the link between card `dev` and the host with copies of `bytes` from and to page-locked memory, after one
/// copy each way that wakes the link (the best of four each way: a disturbance only ever slows a copy).  Copies of
/// 256 MiB measure the link's speed; much smaller ones also measure the time to start a copy.  Unknown (zeros) when
/// the memory could not be had.
LinkSpeed measure_host_link(int dev, size_t bytes);
/// Measures the copies from card `from` to card `to` and back, `bytes` each, as the runtime makes them (peer to peer
/// where the cards can, else through the host).  Unknown (zeros) when the memory could not be had.
LinkSpeed measure_card_link(int from, int to, size_t bytes);
/// Gets card `dev`'s compute speed as the layer split's rules take it (split_rules::get_layer_ms): its
/// multiprocessors times its clock in GHz, at least 1; the clock is taken as 1.8 GHz where the runtime does not say.
double get_card_speed(int dev);

}  // namespace strata::core

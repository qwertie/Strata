// include/strata/models/qwen38/session_identity.hpp - what a saved Qwen3.8 conversation is bound to (a session file of
// `strata --serve` and a pause file of strata-q38): the model files the engine loaded, and the settings that change
// what the saved bytes mean.
#pragma once

#include "strata/core/expert_source.hpp"
#include "strata/models/qwen38/model.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace strata::qwen38 {

/// Gets the fingerprint of every model input `m` loaded (session_model_fingerprint's) into `fingerprint`; false (and
/// `err`) when a file cannot be read.  `experts` is the source its experts were read from, from which the files they
/// were read from come (null: none); `next_file` is called before each file is read, as session_model_fingerprint's
/// `beat` is.
bool get_session_model_fingerprint(const Model& m, const strata::core::ExpertSource* experts, uint64_t& fingerprint,
                                   std::string& err, const std::function<void()>& next_file = {});
/// Gets the fingerprint of `m`'s resolved settings that change what a saved conversation's bytes mean: its rope, the
/// loaded control vectors, the K/V format and the arithmetic switches.  Sampling, seeds, draft tuning and the expert
/// tier are not in it.
uint64_t get_session_config_fingerprint(const Model& m);

}  // namespace strata::qwen38

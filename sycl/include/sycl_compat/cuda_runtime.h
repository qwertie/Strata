// sycl/include/sycl_compat/cuda_runtime.h - the CUDA runtime names that shared headers use, on SYCL.
//
// This directory goes first on the SYCL build's include path (sycl/CMakeLists.txt), so shared headers that name CUDA
// handle types (include/strata/base/gpu/graph.hpp, include/strata/models/qwen38/session.hpp,
// include/strata/core/peer_experts.hpp) compile for SYCL unchanged; the CUDA and HIP builds never see this
// file. Each name is the type dpct translates it to, so the migrated sources that use these headers see the same types
// as before. A shared header that needs another name fails to compile here: add the name, do not copy the header into
// sycl/include.
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

using cudaGraph_t = dpct::experimental::command_graph_ptr;
using cudaGraphExec_t = dpct::experimental::command_graph_exec_ptr;
using cudaEvent_t = dpct::event_ptr;
using cudaStream_t = dpct::queue_ptr;

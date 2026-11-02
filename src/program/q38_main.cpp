// src/program/q38_main.cpp - strata-q38: the Qwen3.8 engine whose conversation lives in a page pool
// (include/strata/models/qwen38/paged_model.hpp).
#include "strata/models/qwen38/paged_model.hpp"

int main(int argc, char** argv) { return strata::qwen38::run_paged_engine(argc, argv); }

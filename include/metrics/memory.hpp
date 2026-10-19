
#pragma once
#include "common/config.hpp"

namespace vocab_opt::metrics {

MemoryEstimate estimate_memory(int vocab_size, const ModelConfig& cfg);

double mib_from_bytes(int64_t bytes);

} // namespace vocab_opt::metrics

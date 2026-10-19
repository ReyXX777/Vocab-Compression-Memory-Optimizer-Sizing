
#pragma once
#include "common/config.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace vocab_opt::reporting {

void write_csv(const std::filesystem::path& path, const std::vector<CandidateResult>& results);

} // namespace vocab_opt::reporting

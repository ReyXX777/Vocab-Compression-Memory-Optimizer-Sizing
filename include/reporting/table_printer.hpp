
#pragma once
#include "common/config.hpp"
#include <vector>

namespace vocab_opt::reporting {

void print_table(const std::vector<CandidateResult>& results, const CandidateResult& best);

void print_knee(const std::vector<CandidateResult>& results);

} // namespace vocab_opt::reporting

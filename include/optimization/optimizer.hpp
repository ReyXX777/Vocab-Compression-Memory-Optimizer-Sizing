
#pragma once
#include "common/config.hpp"
#include <vector>

namespace vocab_opt::optimization {

class VocabOptimizer {
public:
    explicit VocabOptimizer(const ExperimentConfig& cfg) : cfg_(cfg) {}

    // Global optimum via evaluating U(V) = rho(V) - lambda*M_total
    // Proposition 4: finite optimum due to bounded rho + linear penalty
    CandidateResult find_best(const std::vector<CandidateResult>& results) const;

    // First-order neighbor conditions for interior optimum
    bool check_first_order(const std::vector<CandidateResult>& sorted, std::size_t idx) const;

    double compute_utility(const CompressionResult& comp, const MemoryEstimate& mem) const;

private:
    ExperimentConfig cfg_;
};

} // namespace vocab_opt::optimization


#pragma once
#include "common/config.hpp"
#include <vector>

namespace vocab_opt::metrics {

// Identify diminishing returns: compute Delta rho and marginal per MiB
void compute_marginals(std::vector<CandidateResult>& results);

struct KneeResult {
    int knee_vocab = 0;
    std::size_t knee_index = 0;
    double max_curvature = 0.0;
};

KneeResult detect_knee_elbow(const std::vector<CandidateResult>& results);

KneeResult detect_knee_threshold(const std::vector<CandidateResult>& results, double lambda_kappa);

} // namespace vocab_opt::metrics

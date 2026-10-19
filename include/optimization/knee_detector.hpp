
#pragma once
#include "common/config.hpp"
#include <vector>

namespace vocab_opt::optimization {

// Separate knee-point heuristic from optimizer (as per recommendation)
class KneeDetector {
public:
    struct Result {
        int vocab_size = 0;
        std::size_t index = 0;
        double score = 0.0;
        std::string method;
    };

    static Result elbow_distance(const std::vector<CandidateResult>& results);
    static Result max_marginal_drop(const std::vector<CandidateResult>& results);
    static Result threshold_crossing(const std::vector<CandidateResult>& results, double lambda_kappa);
};

} // namespace vocab_opt::optimization

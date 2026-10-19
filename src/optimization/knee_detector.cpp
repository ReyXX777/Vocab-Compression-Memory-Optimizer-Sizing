
#include "optimization/knee_detector.hpp"
#include "metrics/marginal.hpp"
#include <algorithm>
#include <cmath>

namespace vocab_opt::optimization {

KneeDetector::Result KneeDetector::elbow_distance(const std::vector<CandidateResult>& results) {
    auto k = metrics::detect_knee_elbow(results);
    return {k.knee_vocab, k.knee_index, k.max_curvature, "elbow_distance"};
}

KneeDetector::Result KneeDetector::max_marginal_drop(const std::vector<CandidateResult>& results) {
    if (results.size()<2) return {results.empty()?0:results[0].vocab_size,0,0.0,"marginal_drop"};
    double max_drop = -1;
    std::size_t best = 0;
    for (std::size_t i=1;i<results.size();++i) {
        double drop = results[i-1].delta_rho_per_mib - results[i].delta_rho_per_mib;
        if (drop > max_drop) { max_drop = drop; best = i; }
    }
    return {results[best].vocab_size, best, max_drop, "max_marginal_drop"};
}

KneeDetector::Result KneeDetector::threshold_crossing(const std::vector<CandidateResult>& results, double lambda_kappa) {
    auto k = metrics::detect_knee_threshold(results, lambda_kappa);
    return {k.knee_vocab, k.knee_index, k.max_curvature, "threshold_crossing (Delta rho < lambda*kappa)"};
}

} // namespace vocab_opt::optimization

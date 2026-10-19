
#include "metrics/marginal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vocab_opt::metrics {

void compute_marginals(std::vector<CandidateResult>& results) {
    if (results.empty()) return;
    std::sort(results.begin(), results.end(), [](auto& a, auto& b){ return a.vocab_size < b.vocab_size; });
    for (std::size_t i=0;i<results.size();++i) {
        if (i==0) {
            results[i].delta_rho = results[i].compression.rho;
            double mib = results[i].memory.total_mib;
            results[i].delta_rho_per_mib = mib>0 ? results[i].delta_rho / mib : 0.0;
        } else {
            double drho = results[i].compression.rho - results[i-1].compression.rho;
            double dmib = results[i].memory.total_mib - results[i-1].memory.total_mib;
            results[i].delta_rho = drho;
            results[i].delta_rho_per_mib = dmib>0 ? drho/dmib : 0.0;
        }
    }
}

KneeResult detect_knee_elbow(const std::vector<CandidateResult>& results) {
    if (results.size()<3) return {results.empty()?0:results[0].vocab_size,0,0.0};
    // Elbow method: max distance from line between first and last point (rho vs MiB)
    double x0 = results.front().memory.total_mib;
    double y0 = results.front().compression.rho;
    double x1 = results.back().memory.total_mib;
    double y1 = results.back().compression.rho;
    double max_dist = -1;
    std::size_t best_idx = 0;
    double line_len = std::hypot(x1-x0, y1-y0);
    if (line_len==0) return {results[0].vocab_size,0,0.0};
    for (std::size_t i=1;i+1<results.size();++i) {
        double x = results[i].memory.total_mib;
        double y = results[i].compression.rho;
        // distance point to line
        double num = std::abs((y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0);
        double dist = num / line_len;
        if (dist > max_dist) { max_dist = dist; best_idx = i; }
    }
    return {results[best_idx].vocab_size, best_idx, max_dist};
}

KneeResult detect_knee_threshold(const std::vector<CandidateResult>& results, double lambda_kappa) {
    // Find where delta_rho < lambda*kappa (penalty dominates)
    for (std::size_t i=1;i<results.size();++i) {
        if (results[i].delta_rho < lambda_kappa) {
            return {results[i].vocab_size, i, results[i].delta_rho};
        }
    }
    return {results.back().vocab_size, results.size()-1, 0.0};
}

} // namespace vocab_opt::metrics

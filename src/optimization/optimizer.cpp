
#include "optimization/optimizer.hpp"
#include <algorithm>
#include <limits>

namespace vocab_opt::optimization {

double VocabOptimizer::compute_utility(const CompressionResult& comp, const MemoryEstimate& mem) const {
    // U(V) = rho(V) - lambda * M(V)
    // lambda must have units that make compression and memory commensurable
    return comp.rho - cfg_.lambda * mem.total_mib;
}

CandidateResult VocabOptimizer::find_best(const std::vector<CandidateResult>& results) const {
    if (results.empty()) throw std::runtime_error("No results");
    CandidateResult best = results[0];
    double best_u = compute_utility(best.compression, best.memory);
    best.utility = best_u;
    for (auto& r: results) {
        double u = compute_utility(r.compression, r.memory);
        if (u > best_u) { best_u = u; best = r; best.utility = u; }
    }
    return best;
}

bool VocabOptimizer::check_first_order(const std::vector<CandidateResult>& sorted, std::size_t idx) const {
    // For interior global maximizer: Delta rho(V*-1) >= lambda*kappa and Delta rho(V*) <= lambda*kappa
    if (sorted.empty() || idx>=sorted.size()) return false;
    if (idx==0 || idx+1>=sorted.size()) return false; // boundary not interior
    double kappa_mib = sorted[idx].memory.kappa_mib_per_token; // lambda * kappa_mib is threshold
    double threshold = cfg_.lambda * kappa_mib; // Wait: Actually utility uses lambda * M_total, so Delta U = Delta rho - lambda*Delta M
    // So condition Delta rho >= lambda*Delta M and Delta rho_next <= lambda*Delta M_next
    double delta_prev = sorted[idx].compression.rho - sorted[idx-1].compression.rho;
    double dM_prev = sorted[idx].memory.total_mib - sorted[idx-1].memory.total_mib;
    double delta_next = sorted[idx+1].compression.rho - sorted[idx].compression.rho;
    double dM_next = sorted[idx+1].memory.total_mib - sorted[idx].memory.total_mib;
    return delta_prev >= cfg_.lambda * dM_prev && delta_next <= cfg_.lambda * dM_next;
}

} // namespace vocab_opt::optimization

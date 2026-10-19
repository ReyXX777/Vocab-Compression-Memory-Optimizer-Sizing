
#include "reporting/table_printer.hpp"
#include <iostream>
#include <iomanip>

namespace vocab_opt::reporting {

void print_table(const std::vector<CandidateResult>& results, const CandidateResult& best) {
    std::cout << "\n=== Vocabulary Sizing – Compression vs Memory Trade-off ===\n";
    std::cout << std::left << std::setw(8) << "V"
              << std::setw(12) << "T_tokens"
              << std::setw(12) << "rho(B/T)"
              << std::setw(12) << "tau(T/B)"
              << std::setw(14) << "M_total_MiB"
              << std::setw(12) << "U(V)"
              << std::setw(14) << "d_rho"
              << std::setw(18) << "d_rho/MiB"
              << "note\n";
    std::cout << std::string(100,'-') << "\n";
    for (auto& r: results) {
        std::cout << std::left << std::setw(8) << r.vocab_size
                  << std::setw(12) << r.compression.tokens
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.compression.rho
                  << std::setw(12) << r.compression.tau
                  << std::setw(14) << r.memory.total_mib
                  << std::setw(12) << r.utility
                  << std::setw(14) << r.delta_rho
                  << std::setw(18) << r.delta_rho_per_mib;
        if (r.vocab_size==best.vocab_size) std::cout << " <-- BEST (max U)";
        if (r.is_knee) std::cout << " [knee]";
        std::cout << "\n";
    }
    std::cout << "\nBest V = " << best.vocab_size << " with U=" << best.utility
              << " rho=" << best.compression.rho << " M_total=" << best.memory.total_mib << " MiB\n";
}

void print_knee(const std::vector<CandidateResult>& results) {
    if (results.empty()) return;
    std::cout << "\n--- Marginal Analysis (diminishing returns) ---\n";
    for (auto& r: results) {
        if (r.is_knee) {
            std::cout << "Knee detected at V=" << r.vocab_size << " (rho=" << r.compression.rho << ")\n";
        }
    }
}

} // namespace vocab_opt::reporting

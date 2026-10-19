
#include "reporting/csv_reporter.hpp"
#include <fstream>
#include <iomanip>

namespace vocab_opt::reporting {

void write_csv(const std::filesystem::path& path, const std::vector<CandidateResult>& results) {
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Cannot open CSV for writing: " + path.string());
    ofs << "V,B_bytes,T_tokens,rho_B_per_T,tau_T_per_B,M_embed_MiB,M_total_MiB,kappa_MiB_per_V,U,delta_rho,delta_rho_per_MiB,is_knee
";
    ofs << std::setprecision(8) << std::fixed;
    for (auto& r: results) {
        ofs << r.vocab_size << "," << r.compression.bytes << "," << r.compression.tokens << ","
            << r.compression.rho << "," << r.compression.tau << ","
            << r.memory.embed_mib << "," << r.memory.total_mib << ","
            << r.memory.kappa_mib_per_token << ","
            << r.utility << "," << r.delta_rho << "," << r.delta_rho_per_mib << ","
            << (r.is_knee?1:0) << "\n";
    }
}

} // namespace vocab_opt::reporting

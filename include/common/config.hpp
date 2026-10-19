
#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace vocab_opt {

struct ModelConfig {
    int embedding_dim = 4096;      // d
    int bytes_per_param = 2;       // b: 2 for FP16/BF16
    bool tied_embeddings = false;  // if false, double for output projection
};

struct ExperimentConfig {
    std::vector<int> candidate_vs = {32000, 48000, 64000, 96000, 128000};
    ModelConfig model;
    double lambda = 0.0008; // penalty per MiB: must make rho and MiB commensurable
    std::string train_path;
    std::string eval_path;
    std::string output_csv = "results.csv";
    double train_ratio = 0.9;
    int min_merge_freq = 2;
    bool use_shared_trajectory = true; // reuse single large vocab trajectory (nested BPE) for monotonic guarantee
    bool lowercase = false;
};

struct MemoryEstimate {
    int64_t vocab_size = 0;
    int64_t embed_params = 0;
    double embed_mib = 0.0;
    double total_mib = 0.0;
    double kappa_mib_per_token = 0.0; // delta M per vocab entry
};

struct CompressionResult {
    ByteCount bytes = 0;
    TokenCount tokens = 0;
    double rho = 0.0; // B/T
    double tau = 0.0; // T/B
};

struct CandidateResult {
    int vocab_size = 0;
    CompressionResult compression;
    MemoryEstimate memory;
    double utility = 0.0;
    double delta_rho = 0.0;
    double delta_rho_per_mib = 0.0;
    bool is_knee = false;
};

} // namespace vocab_opt

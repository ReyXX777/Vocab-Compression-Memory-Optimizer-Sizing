
#include "metrics/memory.hpp"
#include <cmath>

namespace vocab_opt::metrics {

double mib_from_bytes(int64_t bytes) {
    return static_cast<double>(bytes) / (1024.0*1024.0);
}

MemoryEstimate estimate_memory(int vocab_size, const ModelConfig& cfg) {
    MemoryEstimate m;
    m.vocab_size = vocab_size;
    int64_t embed_params = static_cast<int64_t>(vocab_size) * cfg.embedding_dim;
    m.embed_params = embed_params;
    int64_t embed_bytes = embed_params * cfg.bytes_per_param;
    m.embed_mib = mib_from_bytes(embed_bytes);
    int64_t total_params = cfg.tied_embeddings ? embed_params : embed_params*2;
    int64_t total_bytes = total_params * cfg.bytes_per_param;
    m.total_mib = mib_from_bytes(total_bytes);
    // kappa = db in bytes, converted to MiB
    int64_t kappa_bytes = static_cast<int64_t>(cfg.embedding_dim) * cfg.bytes_per_param;
    if (!cfg.tied_embeddings) kappa_bytes *= 2;
    m.kappa_mib_per_token = mib_from_bytes(kappa_bytes);
    return m;
}

} // namespace vocab_opt::metrics

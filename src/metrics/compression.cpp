
#include "metrics/compression.hpp"
#include <execution>
#include <numeric>
#include <mutex>

namespace vocab_opt::metrics {

CompressionResult evaluate(const tokenizer::BpeTokenizer& tok, const std::vector<std::string>& eval_corpus) {
    ByteCount B = 0;
    TokenCount T = 0;
    for (auto& s: eval_corpus) {
        B += s.size();
        T += tok.count_tokens(std::string_view(s));
    }
    double rho = T==0 ? 0.0 : static_cast<double>(B)/static_cast<double>(T);
    double tau = B==0 ? 0.0 : static_cast<double>(T)/static_cast<double>(B);
    return {B,T,rho,tau};
}

CompressionResult evaluate(const tokenizer::BpeTokenizer& tok, const std::vector<std::string_view>& eval_corpus) {
    ByteCount B=0;
    TokenCount T=0;
    for (auto sv: eval_corpus) {
        B += sv.size();
        T += tok.count_tokens(sv);
    }
    double rho = T==0?0.0: double(B)/double(T);
    double tau = B==0?0.0: double(T)/double(B);
    return {B,T,rho,tau};
}

CompressionResult evaluate_parallel(const tokenizer::BpeTokenizer& tok, const std::vector<std::string>& eval_corpus) {
    // Parallelize corpus evaluation using C++17 execution policies
    // Eliminate temporary object creation in scoring loops: use local accumulators
    std::vector<std::size_t> token_counts(eval_corpus.size());
    std::transform(std::execution::par_unseq, eval_corpus.begin(), eval_corpus.end(), token_counts.begin(),
        [&](const std::string& s){ return tok.count_tokens(std::string_view(s)); });

    ByteCount B = 0;
    TokenCount T = 0;
    for (std::size_t i=0;i<eval_corpus.size();++i) {
        B += eval_corpus[i].size();
        T += token_counts[i];
    }
    double rho = T==0?0.0: double(B)/double(T);
    double tau = B==0?0.0: double(T)/double(B);
    return {B,T,rho,tau};
}

} // namespace vocab_opt::metrics


#pragma once
#include "common/config.hpp"
#include "tokenizer/bpe_tokenizer.hpp"
#include <vector>
#include <string_view>

namespace vocab_opt::metrics {

CompressionResult evaluate(
    const tokenizer::BpeTokenizer& tok,
    const std::vector<std::string>& eval_corpus
);

// Overload for string_view corpus (zero-copy lines)
CompressionResult evaluate(
    const tokenizer::BpeTokenizer& tok,
    const std::vector<std::string_view>& eval_corpus
);

// Parallel version using C++17 execution policies
CompressionResult evaluate_parallel(
    const tokenizer::BpeTokenizer& tok,
    const std::vector<std::string>& eval_corpus
);

} // namespace vocab_opt::metrics

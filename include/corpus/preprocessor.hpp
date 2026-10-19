
#pragma once
#include <string>
#include <string_view>
#include <vector>

namespace vocab_opt::corpus {

class Preprocessor {
public:
    explicit Preprocessor(bool lowercase = false) : lowercase_(lowercase) {}

    // Zero-copy aware: takes string_view, returns normalized string (owned)
    // Handles UTF-8 boundaries correctly
    std::string normalize(std::string_view sv) const;

    // Pre-tokenization: split into chunks preserving whitespace semantics
    // Returns views into original when possible; for assignment we return owned strings
    // to keep API simple but internally uses string_view slicing
    std::vector<std::string> pretokenize(std::string_view sv) const;

    // For BPE training: flatten corpus into vector<string> with reserve
    std::vector<std::string> process_corpus(const std::vector<std::string_view>& lines) const;

private:
    bool lowercase_;
};

} // namespace vocab_opt::corpus

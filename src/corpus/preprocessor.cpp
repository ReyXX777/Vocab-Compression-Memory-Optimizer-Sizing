
#include "corpus/preprocessor.hpp"
#include "common/utf8.hpp"
#include <cctype>
#include <algorithm>

namespace vocab_opt::corpus {

std::string Preprocessor::normalize(std::string_view sv) const {
    // Zero-copy scan + reserve
    std::string tmp = utf8::normalize_utf8(sv);
    if (lowercase_) {
        // simple ascii lowercase; preserve UTF-8 multibyte as-is (would need ICU for full)
        for (auto& c: tmp) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    // trim? Keep as is to preserve tokenization behavior – note in analysis
    return tmp;
}

std::vector<std::string> Preprocessor::pretokenize(std::string_view sv) const {
    // Split on whitespace using string_view slicing, then own
    std::vector<std::string> out;
    out.reserve(32);
    std::size_t i=0;
    while (i < sv.size()) {
        while (i < sv.size() && std::isspace(static_cast<unsigned char>(sv[i]))) ++i;
        if (i>=sv.size()) break;
        std::size_t j=i;
        while (j < sv.size() && !std::isspace(static_cast<unsigned char>(sv[j]))) ++j;
        // slice zero-copy then normalize
        std::string_view slice(sv.data()+i, j-i);
        out.emplace_back(normalize(slice));
        i=j;
    }
    if (out.empty() && !sv.empty()) out.emplace_back(normalize(sv));
    return out;
}

std::vector<std::string> Preprocessor::process_corpus(const std::vector<std::string_view>& lines) const {
    std::vector<std::string> all;
    all.reserve(lines.size()*2);
    for (auto sv: lines) {
        if (sv.empty()) continue;
        auto toks = pretokenize(sv);
        for (auto& t: toks) if (!t.empty()) all.emplace_back(std::move(t));
    }
    return all;
}

} // namespace vocab_opt::corpus

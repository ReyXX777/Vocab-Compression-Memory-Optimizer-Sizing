
#pragma once
#include "common/config.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <filesystem>

namespace vocab_opt::corpus {

// Master std::string_view for Zero-Copy Slicing
class CorpusLoader {
public:
    // Loads entire file into owned buffer, returns views into it
    // Aggressively reserve memory for dynamic buffers
    struct LoadedCorpus {
        std::string buffer; // owns bytes
        std::vector<std::string_view> lines; // zero-copy slices
        ByteCount total_bytes() const { return buffer.size(); }
    };

    static LoadedCorpus load_file(const std::filesystem::path& path);
    static LoadedCorpus load_files(const std::vector<std::filesystem::path>& paths);

    // Train/eval split without copying: returns two LoadedCorpus that share slices logic
    // but each owns its buffer for simplicity; views remain zero-copy inside each
    struct Split {
        LoadedCorpus train;
        LoadedCorpus eval;
    };
    static Split split_train_eval(LoadedCorpus&& corpus, double train_ratio = 0.9);

    static std::vector<std::filesystem::path> discover(const std::string& pattern_or_path);
};

} // namespace vocab_opt::corpus

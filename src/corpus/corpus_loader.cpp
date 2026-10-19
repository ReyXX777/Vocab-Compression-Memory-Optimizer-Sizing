
#include "corpus/corpus_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

namespace vocab_opt::corpus {

CorpusLoader::LoadedCorpus CorpusLoader::load_file(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path.string());
    // reserve aggressively
    ifs.seekg(0, std::ios::end);
    std::size_t sz = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::string buffer;
    buffer.reserve(sz + 1024);
    buffer.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    LoadedCorpus lc;
    lc.buffer = std::move(buffer);
    // zero-copy slicing into lines
    lc.lines.reserve(std::count(lc.buffer.begin(), lc.buffer.end(), '\n') + 1);
    std::string_view bv(lc.buffer);
    std::size_t start = 0;
    for (std::size_t i = 0; i < bv.size(); ++i) {
        if (bv[i] == '\n') {
            lc.lines.emplace_back(bv.data() + start, i - start);
            start = i + 1;
        }
    }
    if (start <= bv.size()) lc.lines.emplace_back(bv.data() + start, bv.size() - start);
    return lc;
}

CorpusLoader::LoadedCorpus CorpusLoader::load_files(const std::vector<std::filesystem::path>& paths) {
    std::size_t total = 0;
    for (auto& p: paths) {
        if (std::filesystem::exists(p)) total += std::filesystem::file_size(p);
    }
    std::string buffer;
    buffer.reserve(total + 1024);
    for (auto& p: paths) {
        std::ifstream ifs(p, std::ios::binary);
        if (!ifs) continue;
        std::string tmp((std::istreambuf_iterator<char>(ifs)), {});
        buffer += tmp;
        buffer.push_back('\n');
    }
    LoadedCorpus lc;
    lc.buffer = std::move(buffer);
    std::string_view bv(lc.buffer);
    lc.lines.reserve(std::count(bv.begin(), bv.end(), '\n') + 1);
    std::size_t start = 0;
    for (std::size_t i=0;i<bv.size();++i) if (bv[i]=='\n') { lc.lines.emplace_back(bv.data()+start, i-start); start=i+1; }
    if (start <= bv.size()) lc.lines.emplace_back(bv.data()+start, bv.size()-start);
    return lc;
}

CorpusLoader::Split CorpusLoader::split_train_eval(LoadedCorpus&& corpus, double train_ratio) {
    // Deterministic shuffle with fixed seed to avoid leakage
    std::vector<std::string_view> lines = std::move(corpus.lines);
    // Keep buffer ownership separate for split: we will create new buffers for each split
    // but preserve zero-copy within each split
    std::mt19937 rng(42);
    std::shuffle(lines.begin(), lines.end(), rng);
    std::size_t n_train = static_cast<std::size_t>(lines.size() * train_ratio);

    Split sp;
    // Build train buffer
    std::size_t train_bytes = 0;
    for (std::size_t i=0;i<n_train;++i) train_bytes += lines[i].size() + 1;
    sp.train.buffer.reserve(train_bytes);
    for (std::size_t i=0;i<n_train;++i) { sp.train.buffer.append(lines[i].data(), lines[i].size()); sp.train.buffer.push_back('\n'); }

    std::size_t eval_bytes = 0;
    for (std::size_t i=n_train;i<lines.size();++i) eval_bytes += lines[i].size() + 1;
    sp.eval.buffer.reserve(eval_bytes);
    for (std::size_t i=n_train;i<lines.size();++i) { sp.eval.buffer.append(lines[i].data(), lines[i].size()); sp.eval.buffer.push_back('\n'); }

    // Re-slice zero-copy inside each
    auto slice = [](LoadedCorpus& lc){
        std::string_view bv(lc.buffer);
        lc.lines.reserve(std::count(bv.begin(), bv.end(), '\n') + 1);
        std::size_t s=0;
        for (std::size_t i=0;i<bv.size();++i) if (bv[i]=='\n') { lc.lines.emplace_back(bv.data()+s, i-s); s=i+1; }
        if (s <= bv.size()) lc.lines.emplace_back(bv.data()+s, bv.size()-s);
    };
    slice(sp.train);
    slice(sp.eval);
    return sp;
}

std::vector<std::filesystem::path> CorpusLoader::discover(const std::string& pattern_or_path) {
    std::filesystem::path p(pattern_or_path);
    if (std::filesystem::is_regular_file(p)) return {p};
    if (std::filesystem::is_directory(p)) {
        std::vector<std::filesystem::path> out;
        for (auto& e: std::filesystem::recursive_directory_iterator(p)) if (e.is_regular_file()) out.push_back(e.path());
        return out;
    }
    return {p};
}

} // namespace vocab_opt::corpus
